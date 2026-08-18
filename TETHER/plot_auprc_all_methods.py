import sys
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker, HPacker, DrawingArea
from matplotlib.patches import Rectangle

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve,
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER")
DATA_DIR = PROJECT_DIR / "cached_data"
CHKPT_DIR = PROJECT_DIR / "checkpoints"
CHKPT_COPY_DIR = PROJECT_DIR / "checkpoints copy"
RESULT_DIR = PROJECT_DIR / "testing_results"

sys.path.append(str(PROJECT_DIR))

import models.tf_to_tg as tf_to_tg_module
import models.tf_to_dna as tf_to_dna_module
import scripts.build_tf_to_tg_train_data as tf_tg_data_builder
from scripts.train_tf_to_tg_model import TFTGEdgeBagDataset, collate_tftg_edge_bags
import utils
import config
import warnings
import plotting_utils
import argparse

warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)

tf_tg_input_cache_dir = DATA_DIR / "tf_tg_training_cache"

all_evaluation_plot_dir = PROJECT_DIR / "plots"
all_evaluation_plot_dir.mkdir(exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# This script loads its models with compile_model=True. TFTGRegulationModel crops the TF
# embedding to a per-chunk width, so it compiles one graph per distinct (crop width, chunk
# count) pair -- around 8-12 of them. The default limit of 8 sits underneath that, and when
# it is exceeded the graphs evict each other and every batch recompiles: throughput drops
# to seconds per batch and nothing is logged to say why.
#
# On a V100 this is currently masked, because the bfloat16 autocast below makes Inductor
# skip compilation entirely. On an A100 (compute capability 8.0+) bf16 compiles for real
# and the limit would bite.
torch._dynamo.config.cache_size_limit = 128

TF_TG_MODEL_CHECKPOINTS = {
    "mESC": {
        "E7.5_rep1": CHKPT_DIR / "mESC" / "E7.5_rep1" / "tf_tg_train_E7.5_rep1_3675131" / "epoch_11_best_model.ckpt",
        # "E7.5_rep1": utils.find_latest_checkpoint(CHKPT_DIR, "mESC", "E7.5_rep1"),
        "E7.5_rep2": utils.find_latest_checkpoint(CHKPT_DIR, "mESC", "E7.5_rep2"),
        "E8.5_rep1": utils.find_latest_checkpoint(CHKPT_DIR, "mESC", "E8.5_rep1", training_number="3691937"),
        "E8.5_rep2": utils.find_latest_checkpoint(CHKPT_DIR, "mESC", "E8.5_rep2", training_number="3691937"),
    },
    "iPSC": {
        "WT_D13_rep1": utils.find_latest_checkpoint(CHKPT_DIR, "iPSC", "WT_D13_rep1"),
    },
    "Macrophage": {
        "buffer_1": utils.find_latest_checkpoint(CHKPT_DIR, "Macrophage", "buffer_1", training_number="3685893"),
        "buffer_2": utils.find_latest_checkpoint(CHKPT_DIR, "Macrophage", "buffer_2", training_number="3713132"),
        "buffer_3": utils.find_latest_checkpoint(CHKPT_DIR, "Macrophage", "buffer_3"),
        "buffer_4": utils.find_latest_checkpoint(CHKPT_DIR, "Macrophage", "buffer_4"),
    },
    "K562": {
        "sample_1": utils.find_latest_checkpoint(CHKPT_DIR, "K562", "sample_1", training_number="3692409"),
    },
    "mouse_liver": {
        "liver_1": utils.find_latest_checkpoint(CHKPT_DIR, "mouse_liver", "liver_1"),
        "liver_3": utils.find_latest_checkpoint(CHKPT_DIR, "mouse_liver", "liver_3")
    },
    "mouse_hepatocytes": {
        "hepatocytes_1": utils.find_latest_checkpoint(CHKPT_DIR, "mouse_hepatocytes", "hepatocytes_1"),
        "hepatocytes_3": utils.find_latest_checkpoint(CHKPT_DIR, "mouse_hepatocytes", "hepatocytes_3"),
    }
}

def generate_model_predictions(model, data_loader, device, tf_idx_to_name, tg_idx_to_name):
    pooling_mode = "lse"
    pooling_temperature = 1.0

    model = model.to(device)
    model.eval()
    
    if device.type == "cuda":
        model = torch.compile(model, mode="reduce-overhead")

    tf_indices_list = []
    tg_indices_list = []
    all_scores = []

    with torch.inference_mode():
        for batch in tqdm(data_loader, desc="Evaluating", ncols=100):
            tf_indices = batch["tf_idx"].detach().cpu().numpy().ravel()
            tg_indices = batch["tg_idx"].detach().cpu().numpy().ravel()

            batch = tf_to_tg_module.move_batch_to_device(batch, device)

            # Scores are ranked, and bf16's 8-bit mantissa collapses nearby logits onto
            # identical values -- ties, which AUROC and especially AUPRC read as lost
            # ranking. Measured on run 3793729's checkpoints, bf16 vs fp32 on the same
            # weights cost 0.009-0.042 pooled AUROC, and swung AUPRC in BOTH directions
            # (0.1779 vs 0.1588 at epoch 0; 0.1436 vs 0.1670 at epoch 5).
            #
            # That matters more here than in training, because this script compares TETHER
            # against SCENIC+/LINGER/CellOracle/Pando/FigR/GRaNIE whose scores are read
            # from files (already computed, unquantized). Running only TETHER's forward in
            # bf16 penalises TETHER alone -- an asymmetry in the benchmark's favour of the
            # baselines. Inference here is one-off and not throughput-bound, so fp32 costs
            # little. Pass --eval_precision bf16 to reproduce pre-fix cached results.
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda" and EVAL_PRECISION == "bf16"),
            ):
                edge_logits, _ = model(
                    tf_embedding=batch["tf_embedding"],
                    tf_mask=batch["tf_mask"],
                    peak_sequences=batch["peak_sequences"],
                    peak_accessibility=batch["peak_accessibility"],
                    peak_distance=batch["peak_distance"],
                    tf_expression=batch["tf_expression"],
                    tg_expression=batch["tg_expression"],
                    peak_mask=batch.get("peak_mask", None),
                    cell_mask=batch["cell_mask"],
                    pooling_mode=pooling_mode,
                    pooling_temperature=pooling_temperature,
                )

            scores = torch.sigmoid(edge_logits.float())

            tf_indices_list.append(tf_indices)
            tg_indices_list.append(tg_indices)
            all_scores.append(scores.detach().cpu().numpy().ravel())

    all_tf_indices_flat = np.concatenate(tf_indices_list)
    all_tg_indices_flat = np.concatenate(tg_indices_list)
    all_scores_flat = np.concatenate(all_scores)

    tf_names = [tf_idx_to_name[int(idx)].upper() for idx in all_tf_indices_flat]
    tg_names = [tg_idx_to_name[int(idx)].upper() for idx in all_tg_indices_flat]

    prediction_df = pd.DataFrame({
        "Source": tf_names,
        "Target": tg_names,
        "Score": all_scores_flat,
    })

    prediction_df = (
        prediction_df.groupby(["Source", "Target"], as_index=False)["Score"]
        .median()
    )

    return prediction_df

def create_ground_truth_comparison_df(score_df, ground_truth_lookup, ground_truth_name):
    gt_tfs, gt_tgs, gt_pairs_set = ground_truth_lookup

    src = score_df["Source"].str.upper()
    tgt = score_df["Target"].str.upper()
    
    # Subset the score_df to only include edges where both TF and TG are in the ground truth test set
    mask = src.isin(gt_tfs) & tgt.isin(gt_tgs)
    df = score_df.loc[mask].copy()
    
    # Build the ground truth labels for the subsetted DataFrame
    df["Source"] = src.loc[mask].values
    df["Target"] = tgt.loc[mask].values

    key = df["Source"] + "\t" + df["Target"]
    df["_in_gt"] = key.isin(gt_pairs_set).astype("int8")
    df["ground_truth_name"] = ground_truth_name

    return df

def create_tf_tg_index_to_name_mappings(metadata):
    tf_idx_to_name = {idx: name for name, idx in metadata["tf_name_to_idx"].items()}
    tg_idx_to_name = {idx: name for name, idx in metadata["tg_id_to_idx"].items()}
    return tf_idx_to_name, tg_idx_to_name

def load_and_standardize_method(name: str, info: dict) -> pd.DataFrame:
    """
    Load a GRN CSV and rename tf_col/target_col/score_col -> Source/Target/Score.
    Extra columns are preserved.
    """
    if info["path"].suffix == ".tsv":
        sep = "\t"
    elif info["path"].suffix == ".csv":
        sep = ","
    
    df = pd.read_csv(info["path"], sep=sep, header=0, index_col=None)

    tf_col     = info["tf_col"]
    target_col = info["target_col"]
    score_col  = info["score_col"]

    rename_map = {
        tf_col: "Source",
        target_col: "Target",
        score_col: "Score",
    }

    missing = [c for c in rename_map if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing expected columns: {missing}. Got: {list(df.columns)}")

    df = df.rename(columns=rename_map)

    df = df[["Source", "Target", "Score"]]
    df["Source"] = df["Source"].astype(str).str.upper()
    df["Target"] = df["Target"].astype(str).str.upper()

    return df

def convert_labeled_dataframe_to_indices(true_interactions, false_interactions, tf_name_to_idx, tg_id_to_idx):
    rows = []
    for tf, tg in true_interactions:
        rows.append((tf, tg, 1))
    for tf, tg in false_interactions:
        rows.append((tf, tg, 0))

    df = pd.DataFrame(rows, columns=["tf_name", "tg_id", "label"])
    df["tf_idx"] = df["tf_name"].str.upper().map(tf_name_to_idx)
    df["tg_idx"] = df["tg_id"].str.upper().map(tg_id_to_idx)

    missing_mask = df["tf_idx"].isna() | df["tg_idx"].isna()
    if missing_mask.any():
        n_missing = missing_mask.sum()
        logging.info(f"Dropping {n_missing} interactions with missing TF or TG indices.")
        df = df.loc[~missing_mask].copy()

    df["tf_idx"] = df["tf_idx"].astype(np.int64)
    df["tg_idx"] = df["tg_idx"].astype(np.int64)
    df["label"] = df["label"].astype(np.float32)

    return df

def sample_auprc_10x_negatives(full_universe, random_state=42):
    positives = full_universe[full_universe["_in_gt"] == 1]
    negatives = full_universe[full_universe["_in_gt"] == 0]

    n_pos = len(positives)
    n_neg_sample = min(n_pos * 10, len(negatives))

    if n_pos == 0 or n_neg_sample == 0:
        return full_universe.iloc[0:0].copy()

    neg_sampled = negatives.sample(
        n=n_neg_sample,
        replace=False,
        random_state=random_state,
    )

    auprc_df = pd.concat([positives, neg_sampled], axis=0)

    # Optional: shuffle, not required for sklearn metrics
    auprc_df = auprc_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return auprc_df

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate TF-TG model on multiple datasets and plot AUPRC.")
    parser.add_argument("--output_dir", type=str, default=str(all_evaluation_plot_dir), help="Directory to save evaluation plots.")
    parser.add_argument("--species", type=str, help="Species to evaluate.")
    parser.add_argument("--cell_type", type=str, help="Cell type to evaluate.")
    parser.add_argument("--sample_name", type=str, help="Sample name to evaluate.")
    parser.add_argument("--cross_model_cell_type", type=str, help="Cell type for cross-model evaluation.")
    parser.add_argument("--cross_model_sample_name", type=str, help="Sample name for cross-model evaluation.")

    parser.add_argument("--force_reload", action="store_true", help="Force reload of data and models.")
    parser.add_argument(
        "--eval_precision",
        choices=["fp32", "bf16"],
        default="fp32",
        help=(
            "Arithmetic for the TF-TG forward pass. Default fp32. 'bf16' reproduces the "
            "pre-2026-08-18 cached predictions, whose ties depress TETHER's AUPRC/AUROC "
            "relative to the file-loaded external methods."
        ),
    )

    return parser.parse_args()

args = parse_arguments()
species = args.species
cell_type = args.cell_type
sample_name = args.sample_name
force_reload = args.force_reload
# Read by generate_model_predictions() at call time (it is defined above this point).
EVAL_PRECISION = args.eval_precision
logging.info(
    f"TF-TG forward pass will run in {EVAL_PRECISION}."
    + ("" if EVAL_PRECISION == "fp32" else
       " WARNING: bf16 ties depress TETHER's scores relative to the file-loaded"
       " external methods, which are not quantised.")
)

cross_model_cell_type = args.cross_model_cell_type
cross_model_sample_name = args.cross_model_sample_name
cross_model_chkpt = TF_TG_MODEL_CHECKPOINTS[cross_model_cell_type][cross_model_sample_name]

sample_to_title_map = {
    "E7.5_rep1": "mESC-1",
    "E8.5_rep1": "mESC-2",
    "buffer_1": "Macrophage-1",
    "buffer_2": "Macrophage-2",
    "sample_1": "K562",
    "hepatocytes_1": "Hepatocytes-1",
    "hepatocytes_3": "Hepatocytes-3"
}

OWN_MODEL_METHOD = "TF-TG Model (own test set)"
CROSS_MODEL_METHOD = "TF-TG Model (cross-trained)"

TFTG_MODEL_METHODS = [
    OWN_MODEL_METHOD,
    CROSS_MODEL_METHOD,
]

model_sample_title = sample_to_title_map.get(sample_name, sample_name)
cross_model_sample_title = sample_to_title_map.get(cross_model_sample_name, cross_model_sample_name)

project_data_dir = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data")

# Load the genome information for the sample
auprc_all_method_dfs = {}
logging.info(f"Evaluating on {cell_type} {sample_name} ({species})")
if species == "mm10":
    gene_ref_file = project_data_dir / "genome_data" / "genome_annotation" / "mm10" / "Mus_musculus.GRCm39.115.gtf.gz"
elif species == "hg38":
    gene_ref_file = project_data_dir / "genome_data" / "genome_annotation" / "hg38" / "Homo_sapiens.GRCh38.113.gtf.gz"

genome_fasta_path = project_data_dir / "genome_data" / "reference_genome" / f"{species}" / f"{species}.fa"
chrom_sizes_path = project_data_dir / "genome_data" / "reference_genome" / f"{species}" / f"{species}.chrom.sizes"

# Specify the train/val/test splits, stratified by chromosome
if species == "mm10":
    train_chroms = [str(i) for i in range(1, 16)]
    val_chroms = [ str(i) for i in range(16, 18)]
    test_chroms = [str(i) for i in range(18, 20)]
elif species == "hg38":
    train_chroms = [str(i) for i in range(1, 18)]
    val_chroms = [str(i) for i in range(18, 20)]
    test_chroms = [str(i) for i in range(20, 23)]

sample_input_data_dir = PROJECT_DIR.parent / "data" / "sample_input_data" / cell_type / sample_name

# Load in the ATAC pseudobulk and filter to only include peaks on the test chromosomes
atac_pseudobulk = pd.read_parquet(sample_input_data_dir / "RE_pseudobulk.parquet")
dataset_peaks = atac_pseudobulk.index.to_list()
dataset_peaks = [peak for peak in dataset_peaks if peak.split(":", 1)[0].replace("chr", "") in test_chroms]

# Create a peak to index map for the peaks on the test chromosomes
atac_peak_map = {peak: idx for idx, peak in enumerate(dataset_peaks)}

# Load in the RNA pseudobulk
rna_pseudobulk = pd.read_parquet(sample_input_data_dir / "TG_pseudobulk.parquet")
rna_pseudobulk_norm = rna_pseudobulk.copy()
rna_pseudobulk_norm.index = rna_pseudobulk_norm.index.str.upper()

# Load in the peak to gene distance
peak_to_gene_distance = pd.read_parquet(sample_input_data_dir / "peak_to_gene_dist.parquet")
peak_to_gene = peak_to_gene_distance.copy()
peak_to_gene["target_id_norm"] = peak_to_gene["target_id"].str.upper()

common_cells = sorted(set(rna_pseudobulk_norm.columns) & set(atac_pseudobulk.columns))

# Load the merged ground truth
cell_type_cache_dir = DATA_DIR / f"{cell_type}_cache"
merged_ground_truth_df = pd.read_parquet(cell_type_cache_dir / f"{cell_type}_merged_ground_truth.parquet")

# Filter the ground truth to only include TFs and TGs that are present in the RNA pseudobulk
gt_tfs_in_rna = set(merged_ground_truth_df["Source"]).intersection(rna_pseudobulk_norm.index)
gt_tgs_in_rna = set(merged_ground_truth_df["Target"]).intersection(rna_pseudobulk_norm.index)
logging.info(f"Ground truth TFs in RNA pseudobulk: {len(gt_tfs_in_rna)} (Example: {list(gt_tfs_in_rna)[:5]})")
logging.info(f"Ground truth TGs in RNA pseudobulk: {len(gt_tgs_in_rna)} (Example: {list(gt_tgs_in_rna)[:5]})")

n_before_rna_filter = len(merged_ground_truth_df)

merged_ground_truth_df = merged_ground_truth_df[
    merged_ground_truth_df["Source"].isin(gt_tfs_in_rna) &
    merged_ground_truth_df["Target"].isin(gt_tgs_in_rna)
].copy()

logging.info(
    f"Ground truth edges after RNA TF/TG filtering: "
    f"{len(merged_ground_truth_df):,} / {n_before_rna_filter:,}"
)

tf_name_to_idx_cache_path = cell_type_cache_dir / "tf_name_to_idx.csv"

# Get the map of the TF names to their indices in the TF-DNA model training data
tf_name_to_idx = pd.read_csv(tf_name_to_idx_cache_path)
tf_name_to_idx["tf_name"] = tf_name_to_idx["tf_name"].str.upper()
tf_name_to_idx = tf_name_to_idx.set_index("tf_name")["tf_idx"].to_dict()

# Only keep ground truth TFs that have embeddings (i.e. were present in the TF-DNA model training data)
gt_tfs_in_embeddings = set(tf_name_to_idx.keys()).intersection(gt_tfs_in_rna)
logging.info(f"Ground truth TFs with embeddings: {len(gt_tfs_in_embeddings)} (Example: {list(gt_tfs_in_embeddings)[:5]})")

n_before_tf_embedding_filter = len(merged_ground_truth_df)

# Filter the ground truth to only include TFs that have embeddings in the TF-DNA model training data
merged_ground_truth_df = merged_ground_truth_df[
    merged_ground_truth_df["Source"].isin(gt_tfs_in_embeddings)
].copy()

logging.info(
    f"Ground truth edges after filtering to TFs with embeddings: "
    f"{len(merged_ground_truth_df):,} / {n_before_tf_embedding_filter:,}"
)

# Split the target genes into train/val/test based on chromosome
train_genes, val_genes, test_genes = tf_tg_data_builder.split_genes_by_chromosome(
    gene_ref_file,
    train_chroms=train_chroms,
    val_chroms=val_chroms,
    test_chroms=test_chroms
    )

# Create the train/val/test splits of the ground truth based on the gene splits
gt_train_df, gt_val_df, gt_test_df = tf_tg_data_builder.create_train_val_test_splits(
    merged_ground_truth_df, train_genes, val_genes, test_genes
)
gt_test_df["Source"] = gt_test_df["Source"].astype(str).str.upper()
gt_test_df["Target"] = gt_test_df["Target"].astype(str).str.upper()

logging.info(f"After subsetting to TFs with embeddings and TGs in RNA pseudobulk:")
logging.info(f"  - Train interactions: {len(gt_train_df)} (TFs: {gt_train_df['Source'].nunique()}, TGs: {gt_train_df['Target'].nunique()})")
logging.info(f"  - Val interactions: {len(gt_val_df)} (TFs: {gt_val_df['Source'].nunique()}, TGs: {gt_val_df['Target'].nunique()})")
logging.info(f"  - Test interactions: {len(gt_test_df)} (TFs: {gt_test_df['Source'].nunique()}, TGs: {gt_test_df['Target'].nunique()})")

# Build TF, TG, and edge sets for quick lookup later
gt_test_df = gt_test_df[["Source", "Target"]].dropna()
gt_tfs = set(gt_test_df["Source"].unique())
gt_tgs = set(gt_test_df["Target"].unique())
gt_pairs = (gt_test_df["Source"] + "\t" + gt_test_df["Target"]).drop_duplicates()

gt_lookup = (gt_tfs, gt_tgs, set(gt_pairs))

# Construct the full universe of possible TF-TG pairs for the test set, 
# label them based on presence in the ground truth, and merge with method predictions
full_universe = (
    pd.MultiIndex
    .from_product([gt_tfs, gt_tgs], names=["Source", "Target"])
    .to_frame(index=False)
)

# Create a list of true and false interactions based on whether the candidate edge is in the ground truth
full_universe["_in_gt"] = (full_universe["Source"] + "\t" + full_universe["Target"]).isin(gt_pairs).astype("int8")

# Sample a subset of the full universe for AUPRC evaluation (all positives + 10x negatives) up front,
# so inference only needs to score the edges actually used for evaluation instead of the full dense grid
full_universe_10x_negatives = sample_auprc_10x_negatives(full_universe, random_state=42)

true_df = full_universe_10x_negatives[full_universe_10x_negatives["_in_gt"] == 1]
false_df = full_universe_10x_negatives[full_universe_10x_negatives["_in_gt"] == 0]

true_interactions = zip(true_df["Source"], true_df["Target"])
false_interactions = zip(false_df["Source"], false_df["Target"])

sample_full_grn_dir = RESULT_DIR / "full_test_grns" / cell_type / sample_name
sample_full_grn_dir.mkdir(parents=True, exist_ok=True)

cross_tf_tg_df_file = sample_full_grn_dir / f"tf_tg_cross_model_predictions_{cross_model_cell_type}_{cross_model_sample_name}.tsv"
sample_full_grn_file = sample_full_grn_dir / f"tf_tg_predictions_{cell_type}_{sample_name}.tsv"

if not sample_full_grn_file.exists() or not cross_tf_tg_df_file.exists() or force_reload == True:

    # === CREATE FULL SET OF TF-TG INPUTS FOR ALL POSSIBLE TF-TG PAIRS IN THE TEST SET ===
    # Load the TF and TG name to index mappings from the training cache metadata
    with open(cell_type_cache_dir / "tf_tg_training_cache" / sample_name / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Load the TF and TG name to index mappings from the metadata
    tf_name_to_idx = metadata["tf_name_to_idx"]
    tg_id_to_idx = metadata["tg_id_to_idx"]

    # Create the reverse mappings from index to TF/TG name
    tf_idx_to_name, tg_idx_to_name = create_tf_tg_index_to_name_mappings(metadata)

    # Use the TF and TG name to index mappings to convert the labeled DataFrame of all 
    # possible TF-TG pairs into a labeled DataFrame with TF and TG indices
    labeled_df = convert_labeled_dataframe_to_indices(
        true_interactions, 
        false_interactions, 
        tf_name_to_idx, 
        tg_id_to_idx
    )

    # Create the centered one-hot encoded ATAC peak array
    atac_peak_array = utils.create_centered_peak_onehot_array(
        peak_ids=dataset_peaks,
        genome_fasta=genome_fasta_path,
        chrom_sizes=utils.load_chrom_sizes(chrom_sizes_path),
        peak_id_to_idx=atac_peak_map,
        flank_size=128,
        dtype=np.uint8,
        pad_out_of_bounds=True,
        num_workers=10,
        show_progress=False,
        chunk_size=10000,
    )
    atac_peak_tensor = torch.as_tensor(atac_peak_array, dtype=torch.uint8).float()

    # Prepare the lookup tables needed to build the TF-TG input dataset for the test set
    tg_to_peak_info, cell_to_idx, atac_mat, rna_mat, gene_to_rna_idx = tf_tg_data_builder.prepare_tftg_lookup_tables(
        peak_to_gene=peak_to_gene,
        atac_peak_map=atac_peak_map,
        atac_pseudobulk=atac_pseudobulk,
        rna_pseudobulk_norm=rna_pseudobulk_norm,
        dataset_peaks=dataset_peaks,
        common_cells=common_cells,
        max_precompute_peaks=8,
    )
    
    # Get the max number of peaks within 100Kb of any TG in the test set
    max_peaks_real = max(
        len(tg_to_peak_info.get(tg_name, {}).get("peak_indices", []))
        for tg_name in labeled_df["tg_id"]
    )

    # Build the compact TF-TG input dataset for the test set
    common_build_kwargs = dict(
        max_peaks_per_tg=8,
        max_cells_per_pair=16,
        tg_to_peak_info=tg_to_peak_info,
        cell_to_idx=cell_to_idx,
        atac_mat=atac_mat,
        rna_mat=rna_mat,
        gene_to_rna_idx=gene_to_rna_idx,
        common_cells=common_cells,
        tf_name_to_idx=tf_name_to_idx,
        tg_id_to_idx=tg_id_to_idx,
        max_peaks_real=max_peaks_real,
    )

    tftg_inputs_test = tf_tg_data_builder.build_tftg_inputs(
        labeled_df,
        seed=125,
        silence=True,
        **common_build_kwargs,
    )

    # Load the lookup tensors
    tf_embeddings_tensor = torch.load(
        cell_type_cache_dir / "tf_embeddings.pt",
        weights_only=True,
    )
    tf_mask_tensor = torch.load(
        cell_type_cache_dir / "tf_masks.pt",
        weights_only=True,
    )
    
    # Create the PyTorch dataset for the test set
    dataset = TFTGEdgeBagDataset(
        tftg_inputs_test,
        tf_embeddings_tensor=tf_embeddings_tensor,
        tf_mask_tensor=tf_mask_tensor,
        atac_peak_tensor=atac_peak_tensor
    )

    # Create the PyTorch DataLoader for the test set
    num_workers = 8
    loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_tftg_edge_bags,
    )

    tf_dna_model_chkpt = config.tf_dna_model_checkpoints[cell_type]
    tf_tg_model_chkpt = TF_TG_MODEL_CHECKPOINTS[cell_type][sample_name]

    # Generate the model predictions for the test set and create a DataFrame with TF names, TG names, and predicted scores
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not sample_full_grn_file.exists() or force_reload:
        # Load the TF→TG model
        tf_tg_model = utils.load_tf_tg_regulation_model(
            tf_dna_model_chkpt, 
            tf_tg_model_chkpt, 
            tf_embeddings_tensor, 
            tf_mask_tensor,
            compile_model=True,
            device=device
            )
        
        # Run the model on the test set and generate the predictions DataFrame
        prediction_df = generate_model_predictions(tf_tg_model.model, loader, device, tf_idx_to_name, tg_idx_to_name)
        
        prediction_df.to_csv(sample_full_grn_file, sep="\t", index=False)
    else:
        prediction_df = pd.read_csv(sample_full_grn_file, sep="\t", header=0)
    
    if not cross_tf_tg_df_file.exists() or force_reload:
        # Load the TF→TG model trained on the cross-model cell type and sample
        cross_tf_tg_model = utils.load_tf_tg_regulation_model(
            tf_dna_model_chkpt,
            cross_model_chkpt,
            tf_embeddings_tensor,
            tf_mask_tensor,
            compile_model=True,
            device=device
        )
        
        # Run the cross-trained model on the test set and generate the predictions DataFrame
        cross_model_prediction_df = generate_model_predictions(cross_tf_tg_model.model, loader, device, tf_idx_to_name, tg_idx_to_name)

        cross_model_prediction_df.to_csv(cross_tf_tg_df_file, sep="\t", index=False)
    else:
        cross_model_prediction_df = pd.read_csv(cross_tf_tg_df_file, sep="\t", header=0)
    
else:
    prediction_df = pd.read_csv(sample_full_grn_file, sep="\t", header=0)
    cross_model_prediction_df = pd.read_csv(cross_tf_tg_df_file, sep="\t", header=0)

OTHER_METHOD_MUON_DIR = Path("/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/other_method_grns")

other_method_grns = {}

linger_path       = OTHER_METHOD_MUON_DIR / "LINGER_muon" / f"linger_{cell_type}_{sample_name}.tsv"
scenic_plus_path  = OTHER_METHOD_MUON_DIR / "SCENIC_muon" / f"scenicplus_{cell_type}_{sample_name}.tsv"
cell_oracle_path  = OTHER_METHOD_MUON_DIR / "CellOracle_muon" / f"celloracle_{cell_type}_{sample_name}.tsv"
pando_path        = OTHER_METHOD_MUON_DIR / "Pando_muon" / f"pando_{cell_type}_{sample_name}.tsv"
figr_path         = OTHER_METHOD_MUON_DIR / "FigR_muon" / f"figr_{cell_type}_{sample_name}.tsv"

method_info = {
    "SCENIC+":    {"path": scenic_plus_path, "tf_col": "Source",    "target_col": "Target",    "score_col": "Score"},
    "LINGER":     {"path": linger_path,      "tf_col": "Source",    "target_col": "Target",    "score_col": "Score"},
    "CellOracle": {"path": cell_oracle_path, "tf_col": "Source",    "target_col": "Target",    "score_col": "Score"},
    "Pando":      {"path": pando_path,       "tf_col": "Source",    "target_col": "Target",    "score_col": "Score"},
    "FigR":       {"path": figr_path,        "tf_col": "Source",    "target_col": "Target",    "score_col": "Score"},
}

# Load and standardize the predictions from each method, filtering to only include TFs and TGs present in the ground truth
standardized_method_dfs = {}
for method_name, info in method_info.items():
    df_std = load_and_standardize_method(method_name, info)
    
    mask = df_std["Source"].isin(gt_tfs) & df_std["Target"].isin(gt_tgs)
    df_filtered: pd.DataFrame = df_std.loc[mask]
    
    standardized_method_dfs[method_name] = df_filtered
    
# Add the TF-TG model predictions to the standardized_method_dfs for metric computation
standardized_method_dfs[OWN_MODEL_METHOD] = prediction_df
standardized_method_dfs[CROSS_MODEL_METHOD] = cross_model_prediction_df

auprc_all_method_dfs[sample_name] = {}

labeled_grn_dir = RESULT_DIR / "labeled_auprc_grns" / cell_type / sample_name
labeled_grn_dir.mkdir(parents=True, exist_ok=True)

# Compute metrics for each inference method
for method_name, df_std in standardized_method_dfs.items():
    
    method_grn_file = labeled_grn_dir / f"{method_name.lower().replace('+','')}.tsv"
    if method_grn_file.exists() and not force_reload:
        auprc_df = pd.read_csv(method_grn_file, sep="\t", header=0)
    else:
        # Create a labeled DataFrame of the predicted scores for the method vs the test set ground truth
        method_labeled_df = create_ground_truth_comparison_df(df_std, gt_lookup, "test_chrom_gt")

        y = method_labeled_df["_in_gt"].fillna(0).astype(int).to_numpy()
        s = method_labeled_df["Score"].to_numpy()
        
        # Only keep the edges that are in the full universe of TF-TG pairs for the test set
        eval_df = full_universe_10x_negatives[["Source", "Target", "_in_gt"]].copy()

        auprc_df = eval_df.merge(
            method_labeled_df[["Source", "Target", "Score"]],
            on=["Source", "Target"],
            how="left",
        )

        auprc_df["Score"] = auprc_df["Score"].fillna(0)
    
    auprc_all_method_dfs[sample_name][method_name] = auprc_df
    
    # Save the labeled DataFrame for the method to a TSV file for future reference
    auprc_df.to_csv(labeled_grn_dir / f"{method_name.lower().replace('+','')}.tsv", sep="\t", index=False)


# ===== PLOTTING AUPRC FOR ALL METHODS =====
method_color_dict = {
  OWN_MODEL_METHOD: "#4195df",
  CROSS_MODEL_METHOD: "#86C7E7",
  "LINGER": "#EF767A",
  "CellOracle": "#F9C60D",
  "Pando": "#EF9CFA",
  "SCENIC+": "#82EC32",
  "FigR": "#FDA7BB",
  "GRaNIE": "#F98637"
}

sample_title = sample_to_title_map.get(sample_name, sample_name)

combined_fig, ax = plt.subplots(
    nrows=1,
    ncols=1,
    figsize=(6, 6),
    sharex=True,
    sharey=True,
)

ax.set_box_aspect(1)

cell_type_method_auprc = {
    "sample_name": [],
    "method": [],
    "auprc": [],
    "rand_auprc": [],
}

if sample_name not in auprc_all_method_dfs:
    raise KeyError(f"Sample {sample_name} not found in auprc_all_method_dfs")

auprc_text_lines = []
auprc_metric_rows = []

for method in auprc_all_method_dfs[sample_name].keys():
    print(method)

    if method not in method_color_dict:
        continue
    
    auprc_df = auprc_all_method_dfs[sample_name][method]

    y_auprc = auprc_df["_in_gt"].astype(int).to_numpy()
    s_auprc = auprc_df["Score"].astype(float).to_numpy()

    if len(np.unique(y_auprc)) < 2:
        auprc = np.nan
        rand_auprc = np.nan
        continue

    auprc = average_precision_score(y_auprc, s_auprc)
    prec, rec, _ = precision_recall_curve(y_auprc, s_auprc)

    rand_scores = plotting_utils._create_random_distribution(s_auprc)
    rand_prec, rand_rec, _ = precision_recall_curve(y_auprc, rand_scores)
    rand_auprc = average_precision_score(y_auprc, rand_scores)

    auprc_metric_rows.append({
        "sample_name": sample_name,
        "method": method,
        "auprc": auprc,
        "rand_auprc": rand_auprc,
    })

    method_color = method_color_dict.get(method, "#747474")
    auprc_text_lines.append((method, auprc, method_color))

    line_weight = 3 if method in [OWN_MODEL_METHOD, CROSS_MODEL_METHOD] else 2

    ax.step(
        rec,
        prec,
        where="post",
        lw=line_weight,
        color=method_color,
        label="",
        zorder=3,
    )

    ax.step(
        rand_rec,
        rand_prec,
        where="post",
        lw=1,
        linestyle="--",
        color=method_color,
        label="",
        zorder=3,
        alpha=0.75,
    )

auprc_text_lines_sorted = sorted(
    auprc_text_lines,
    key=lambda x: x[1],
    reverse=True,
)


ax.set_title(sample_title, fontsize=26)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

ax.tick_params(labelsize=18)

legend_rows = []

for method, auprc, method_color in auprc_text_lines_sorted:
    color_box = DrawingArea(16, 16, 0, 0)
    color_box.add_artist(
        Rectangle(
            (0, 1),
            18,
            18,
            facecolor=method_color,
            edgecolor="black",
            linewidth=1.5,
        )
    )

    label_text = TextArea(
        f"{method} = {auprc:.3f}",
        textprops=dict(
            color="black",
            fontsize=18,
        ),
    )

    row = HPacker(
        children=[color_box, label_text],
        align="center",
        pad=0.1,
        sep=6,
    )

    legend_rows.append(row)

packed_legend = VPacker(
    children=legend_rows,
    align="left",
    pad=0,
    sep=8,
)

anchored_text = AnchoredOffsetbox(
    loc="upper center",
    child=packed_legend,
    pad=0.5,
    frameon=False,
    bbox_to_anchor=(0.5, -0.22),
    bbox_transform=ax.transAxes,
    borderpad=0.4,
)

ax.add_artist(anchored_text)

combined_fig.text(
    0.5,
    -0.05,
    "Recall",
    ha="center",
    fontsize=24,
)

combined_fig.text(
    -0.05,
    0.45,
    "Precision",
    va="center",
    rotation="vertical",
    fontsize=24,
)

combined_fig.subplots_adjust(
    left=0.04,
    right=0.98,
    bottom=0.10,
    top=0.88,
    wspace=0.08,
)
auprc_plot_dir = all_evaluation_plot_dir / "auprc_plots"
auprc_plot_dir.mkdir(parents=True, exist_ok=True)

combined_fig.savefig(
    auprc_plot_dir / f"{sample_name}_auprc.png",
    dpi=300,
    bbox_inches="tight",
)

# Save the AUPRC metrics for each method to a CSV file
auprc_metric_dir = all_evaluation_plot_dir / "auprc_metrics"
auprc_metric_dir.mkdir(parents=True, exist_ok=True)

auprc_metrics_df = pd.DataFrame(auprc_metric_rows)
auprc_metrics_df.to_csv(
    auprc_metric_dir / f"{sample_name}_auprc_metrics.tsv",
    sep="\t",
    index=False,
)
