# %%
import json
import os
import sys
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from tqdm import tqdm
import logging
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.strategies import DDPStrategy

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/notebooks/simple_model_testing")
DATA_DIR = PROJECT_DIR / "data"
CHKPT_DIR = PROJECT_DIR / "checkpoints"
CHKPT_COPY_DIR = PROJECT_DIR / "checkpoints copy"
RESULT_DIR = PROJECT_DIR / "testing_results"

sys.path.append(str(PROJECT_DIR))

import models.tf_to_tg_simple as tf_to_tg_module
import models.tf_to_dna as tf_to_dna_module
import scripts.build_tf_to_tg_train_data as tf_tg_data_builder
import utils
import config
import warnings
import argparse

warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)

tf_tg_input_cache_dir = DATA_DIR / "tf_tg_training_cache"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def create_new_tf_tg_regulation_model(
    tf_bind_model_path: Path,
    tf_embeddings_tensor: torch.Tensor,
    tf_mask_tensor: torch.Tensor,
    checkpoint_path: Path | None = None,
) -> tf_to_tg_module.TFTGRegulationModel:

    # 1) Recreate the base TF→DNA model with the same hyperparameters
    base_model = tf_to_dna_module.TFPeakBindingModel(
        tf_embedding_dim=128,
        hidden_dim=128,
        dropout=0.3,
        num_layers=4,
        num_heads=4,
        dim_head=32,
    )

    # 2) Wrap in Lightning module and load checkpoint
    lit_model = tf_to_dna_module.LitTFPeakBindingModel.load_from_checkpoint(
        checkpoint_path=tf_bind_model_path,
        model=base_model,
        tf_embeddings_tensor=tf_embeddings_tensor,
        tf_mask_tensor=tf_mask_tensor,
        lr=1e-4,
        weight_decay=1e-4,
        pos_weight=None,
    )

    # 3) Get the trained base model and freeze it
    trained_tf_peak_model = lit_model.model

    trained_tf_peak_model.eval()

    for p in trained_tf_peak_model.parameters():
        p.requires_grad = False

    trained_tf_peak_model = torch.compile(
        trained_tf_peak_model,
        mode="reduce-overhead",
        fullgraph=False,
    )

    # 4) Inject into your TF→TG model
    tf_tg_model = tf_to_tg_module.TFTGRegulationModel(
        pretrained_tf_peak_model=trained_tf_peak_model,
        d_model=128,
        tf_peak_chunk_size=128,
    )

    # 5) Optionally load TF→TG checkpoint
    if checkpoint_path is not None:
        logging.info(f"Loading TF→TG model weights from checkpoint: {checkpoint_path}")

        tf_tg_ckpt = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        fixed = {}

        for key, value in tf_tg_ckpt["state_dict"].items():
            if key.startswith("model."):
                key = key[len("model."):]
            fixed[key] = value

        tf_tg_model.load_state_dict(fixed, strict=True)

    return tf_tg_model

def create_tf_tg_index_to_name_mappings(tf_name_to_idx, tg_id_to_idx):
    tf_idx_to_name = {idx: name for name, idx in tf_name_to_idx.items()}
    tg_idx_to_name = {idx: name for name, idx in tg_id_to_idx.items()}
    return tf_idx_to_name, tg_idx_to_name

def prepare_tftg_lookup_tables(
    peak_to_gene,
    atac_peak_map,
    atac_pseudobulk,
    rna_pseudobulk_norm,
    dataset_peaks,
    common_cells,
    max_precompute_peaks=None,
):
    valid_peak_set = set(atac_peak_map.keys())

    peak_to_gene_valid = peak_to_gene[
        peak_to_gene["peak_id"].isin(valid_peak_set)
    ].copy()

    peak_to_gene_valid["abs_dist"] = peak_to_gene_valid["TSS_dist"].abs()

    tg_to_peak_info = {}

    # Subset to only peaks within 100kb of the TG TSS and sort by distance
    for tg_norm, sub in peak_to_gene_valid.groupby("target_id_norm", sort=False):
        sub = sub[sub["abs_dist"] <= 100_000].sort_values("abs_dist")

        if sub.empty:
            continue

        # Optional cap to only use the closest N peaks per TG
        if max_precompute_peaks is not None:
            sub = sub.head(max_precompute_peaks)

        peak_ids = sub["peak_id"].tolist()
        peak_indices = np.asarray(
            [atac_peak_map[p] for p in peak_ids],
            dtype=np.int64,
        )
        peak_distances = sub["TSS_dist"].to_numpy(dtype=np.float32)

        tg_to_peak_info[tg_norm] = {
            "peak_ids": peak_ids,
            "peak_indices": peak_indices,
            "peak_distances": peak_distances,
        }

    cell_to_idx = {cell: i for i, cell in enumerate(common_cells)}

    atac_mat = (
        atac_pseudobulk
        .reindex(index=dataset_peaks, columns=common_cells)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    rna_mat = (
        rna_pseudobulk_norm
        .reindex(columns=common_cells)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    gene_to_rna_idx = {gene: i for i, gene in enumerate(rna_pseudobulk_norm.index)}

    return tg_to_peak_info, cell_to_idx, atac_mat, rna_mat, gene_to_rna_idx

def build_tftg_inputs(
    tf_tg_df,
    max_cells_per_pair=8,
    seed=123,
    silence=False,
    *,
    tg_to_peak_info,
    cell_to_idx,
    atac_mat,
    rna_mat,
    gene_to_rna_idx,
    common_cells,
    tf_name_to_idx,
    tg_id_to_idx,
    max_peaks_real,
):
    """
    Build one compact item per TF-TG edge.

    Output shapes:
        label:              [E]
        tf_idx:             [E]
        tg_idx:             [E]
        peak_indices:       [E, P]
        peak_distance:      [E, P]
        peak_mask:          [E, P]
        peak_accessibility: [E, C, P]
        tf_expression:      [E, C]
        tg_expression:      [E, C]
    """

    rng = np.random.default_rng(seed)

    tf_names = []
    tg_names = []
    cell_ids_all = []
    labels = []

    tf_indices = []
    tg_indices = []
    peak_indices_all = []
    peak_access_all = []
    peak_masks_all = []
    tf_expr_all = []
    tg_expr_all = []

    common_cells = list(common_cells)
    n_common_cells = len(common_cells)

    n_total = len(tf_tg_df)
    log_every = max(1, n_total // 50)

    for i, row in enumerate(tf_tg_df.itertuples(index=False), start=1):
        if silence == False:
            if i == 1 or i % log_every == 0 or i == n_total:
                logging.info(f"Building compact TF-TG edges: {100 * i / n_total:.1f}% ({i:,}/{n_total:,})")

        tf_name = row.tf_name
        tg_name = row.tg_id
        label = float(row.label)

        tf_idx = tf_name_to_idx.get(tf_name)
        tg_idx = tg_id_to_idx.get(tg_name)

        if tf_idx is None or tg_idx is None:
            continue

        peak_info = tg_to_peak_info.get(tg_name)
        if peak_info is None:
            continue

        peak_indices_real = list(peak_info["peak_indices"])

        n_peaks = len(peak_indices_real)
        if n_peaks == 0:
            continue

        peak_indices = np.asarray(peak_indices_real, dtype=np.int64)
        peak_mask = np.ones(n_peaks, dtype=bool)

        if n_peaks < max_peaks_real:
            pad_len = max_peaks_real - n_peaks

            peak_indices = np.pad(
                peak_indices,
                (0, pad_len),
                constant_values=0,
            )

            peak_mask = np.pad(
                peak_mask,
                (0, pad_len),
                constant_values=False,
            )

        # Sample cells
        if max_cells_per_pair is None or max_cells_per_pair >= n_common_cells:
            sampled_cells = common_cells
        else:
            sampled_cells = rng.choice(
                common_cells,
                size=max_cells_per_pair,
                replace=False,
            ).tolist()

        sampled_cell_indices = np.asarray(
            [cell_to_idx[c] for c in sampled_cells],
            dtype=np.int64,
        )

        C = len(sampled_cell_indices)
        P = max_peaks_real

        # ATAC accessibility: [C, P]
        peak_acc_matrix = np.zeros((C, P), dtype=np.float32)
        peak_acc_matrix[:, :n_peaks] = atac_mat[
            np.ix_(peak_indices_real, sampled_cell_indices)
        ].T

        # RNA expression: [C]
        tf_rna_idx = gene_to_rna_idx.get(tf_name)
        tg_rna_idx = gene_to_rna_idx.get(tg_name)

        if tf_rna_idx is None or tg_rna_idx is None:
            raise ValueError(
                f"TF or TG missing from RNA matrix after filtering: "
                f"tf_name={tf_name}, tg_name={tg_name}, "
                f"tf_rna_idx={tf_rna_idx}, tg_rna_idx={tg_rna_idx}"
            )

        tf_expr_vals = np.asarray(
            rna_mat[tf_rna_idx, sampled_cell_indices],
            dtype=np.float32,
        ).reshape(-1)

        tg_expr_vals = np.asarray(
            rna_mat[tg_rna_idx, sampled_cell_indices],
            dtype=np.float32,
        ).reshape(-1)

        # Append once per TF-TG edge
        tf_names.append(tf_name)
        tg_names.append(tg_name)
        cell_ids_all.append(sampled_cells)
        labels.append(label)

        tf_indices.append(tf_idx)
        tg_indices.append(tg_idx)
        peak_indices_all.append(peak_indices)
        peak_access_all.append(peak_acc_matrix)
        peak_masks_all.append(peak_mask)
        tf_expr_all.append(tf_expr_vals)
        tg_expr_all.append(tg_expr_vals)

    if len(labels) == 0:
        raise ValueError(
            "No TF-TG examples were created. Check TF/TG IDs, peak-to-gene mapping, "
            "and overlap with ATAC/RNA matrices."
        )

    return {
        "tf_name": tf_names,
        "tg_name": tg_names,
        "cell_ids": cell_ids_all,

        "label": torch.tensor(labels, dtype=torch.float32),

        "tf_idx": torch.tensor(tf_indices, dtype=torch.long),
        "tg_idx": torch.tensor(tg_indices, dtype=torch.long),

        "peak_indices": torch.tensor(np.stack(peak_indices_all), dtype=torch.long),
        "peak_accessibility": torch.tensor(np.stack(peak_access_all), dtype=torch.float32),
        "peak_mask": torch.tensor(np.stack(peak_masks_all), dtype=torch.bool),

        "tf_expression": torch.tensor(np.stack(tf_expr_all), dtype=torch.float32),
        "tg_expression": torch.tensor(np.stack(tg_expr_all), dtype=torch.float32),
    }

parser = argparse.ArgumentParser()

parser.add_argument("--species", type=str, help="Species for training")
parser.add_argument("--cell_type", type=str, help="Cell type for training")
parser.add_argument("--sample_name", type=str, help="Sample name for training")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPU devices to use for training")
parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes to use for training")
parser.add_argument("--job_id", type=str, help="SLURM job ID for this training run")
parser.add_argument("--sample_pairs", type=int, default=None, help="Number of TF-TG pairs to sample for training (default: use all)")
parser.add_argument("--max_peaks_per_tg", type=int, required=False, default=None, help="Maximum number of peaks to consider per TG (default: 64)")
parser.add_argument("--max_cells_per_pair", type=int, default=8, help="Maximum number of cells to sample per TF-TG pair (default: 8)")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
parser.add_argument("--pct_true_edges", type=float, default=0.15, help="Percentage of true edges to include in the training set (default: 0.15)")
parser.add_argument("--true_false_ratio", type=float, default=2.0, help="Ratio of true to false edges in the training set (default: 2.0)")
parser.add_argument("--peak_flank_size", type=int, default=128, help="Size of the flank region around peaks (default: 128)")
parser.add_argument("--checkpoint_path", type=str, required=False, help="Path to a model checkpoint to resume training from")
parser.add_argument("--force_reload", action="store_true", help="Whether to force reload cached data instead of using existing cache files")
args = parser.parse_args()


species = args.species
cell_type = args.cell_type
sample_name = args.sample_name
epochs = args.epochs
num_gpus = args.num_gpus
num_nodes = args.num_nodes
job_id = args.job_id
checkpoint_path = args.checkpoint_path
force_reload = args.force_reload
batch_size = args.batch_size
sample_pairs = args.sample_pairs
max_peaks_per_tg = args.max_peaks_per_tg
max_cells_per_pair = args.max_cells_per_pair
pct_true_edges = args.pct_true_edges
true_false_ratio = args.true_false_ratio
peak_flank_size = args.peak_flank_size

mm10_tf_dna_path = CHKPT_DIR / "tf_dna_mm10_3671604" / "epoch=08-val_auroc=0.9177-val_loss=0.2783.ckpt"
hg38_tf_dna_path = CHKPT_DIR / "tf_dna_hg38_3683606" / "epoch=13-val_auroc=0.9566-val_loss=0.2042.ckpt"

tf_dna_model_checkpoints = {
    "mESC": mm10_tf_dna_path,
    "iPSC": hg38_tf_dna_path,
    "Macrophage": hg38_tf_dna_path,
    "K562": hg38_tf_dna_path
}

project_data_dir = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data")

if species == "mm10":
    gene_ref_file = project_data_dir / "genome_data" / "genome_annotation" / "mm10" / "Mus_musculus.GRCm39.115.gtf.gz"
elif species == "hg38":
    gene_ref_file = project_data_dir / "genome_data" / "genome_annotation" / "hg38" / "Homo_sapiens.GRCh38.113.gtf.gz"

genome_fasta_path = project_data_dir / "genome_data" / "reference_genome" / f"{species}" / f"{species}.fa"
chrom_sizes_path = project_data_dir / "genome_data" / "reference_genome" / f"{species}" / f"{species}.chrom.sizes"
chrom_sizes_path = project_data_dir / "genome_data" / "reference_genome" / f"{species}" / f"{species}.chrom.sizes"

if species == "mm10":
    train_chroms = [str(i) for i in range(1, 16)]
    val_chroms = [ str(i) for i in range(16, 18)]
    test_chroms = [str(i) for i in range(18, 20)]
elif species == "hg38":
    train_chroms = [str(i) for i in range(1, 18)]
    val_chroms = [str(i) for i in range(18, 20)]
    test_chroms = [str(i) for i in range(20, 23)]

sample_input_data_dir = PROJECT_DIR / "data" / "sample_input_data" / cell_type / sample_name

output_dir = PROJECT_DIR / "data" / "checkpoints" / "simplified_model" / f"{cell_type}_{sample_name}_simplified_model_test_{job_id}"

run_name = f"simplified_model_test_{sample_name}_{job_id}"

cell_type_cache_dir = DATA_DIR / f"{cell_type}_cache"

# Save in the cache, but under a simplified model directory name
tf_tg_input_cache_dir = cell_type_cache_dir / f"simplified_model_cache"

tf_tg_atac_peak_cache_path = tf_tg_input_cache_dir / "atac_peak_tensor.pt"
tf_tg_metadata_cache_path = tf_tg_input_cache_dir / "metadata.json"
tf_tg_manifest_cache_path = tf_tg_input_cache_dir / "manifest.json"
tf_tg_train_cache_path = tf_tg_input_cache_dir / "tftg_inputs_train.pt"
tf_tg_val_cache_path = tf_tg_input_cache_dir / "tftg_inputs_val.pt"
tf_tg_test_cache_path = tf_tg_input_cache_dir / "tftg_inputs_test.pt"

# Read in the ATAC and RNA pseudobulk data, and the peak-to-gene distance file
atac_pseudobulk = pd.read_parquet(sample_input_data_dir / "RE_pseudobulk.parquet")
peak_to_gene_distance = pd.read_parquet(sample_input_data_dir / "peak_to_gene_dist.parquet")
rna_pseudobulk = pd.read_parquet(sample_input_data_dir / "TG_pseudobulk.parquet")

logging.info(f"ATAC peaks BEFORE peak-to-gene filtering: {atac_pseudobulk.shape[0]:,}")
# Keep only ATAC peaks that are present in the peak-to-gene distance table
valid_peak_ids = set(peak_to_gene_distance["peak_id"])

atac_pseudobulk = atac_pseudobulk.loc[
    atac_pseudobulk.index.isin(valid_peak_ids)
].copy()
logging.info(f"ATAC peaks AFTER peak-to-gene filtering: {atac_pseudobulk.shape[0]:,}")

rna_pseudobulk_norm = rna_pseudobulk.copy()
rna_pseudobulk_norm.index = rna_pseudobulk_norm.index.str.upper()

common_cells = sorted(set(rna_pseudobulk_norm.columns) & set(atac_pseudobulk.columns))

if len(common_cells) == 0:
    raise ValueError(
        "No common pseudobulk cell columns between RNA and ATAC matrices."
    )

logging.info(f"Common RNA/ATAC pseudobulk columns: {len(common_cells):,}")
    
peak_to_gene = peak_to_gene_distance.copy()
peak_to_gene["target_id_norm"] = peak_to_gene["target_id"].str.upper()

# Load and merge the ground truth files, or load from cache if already merged
merged_ground_truth_path = sample_input_data_dir / "merged_ground_truth.parquet"
if not merged_ground_truth_path.exists():
    merged_ground_truth_df = utils.load_ground_truth_files(
        config.gt_by_dataset_dict[cell_type]
    )
else:
    merged_ground_truth_df = pd.read_parquet(merged_ground_truth_path)
    
merged_ground_truth_df["Source"] = merged_ground_truth_df["Source"].str.upper()
merged_ground_truth_df["Target"] = merged_ground_truth_df["Target"].str.upper()

gt_tfs_in_rna = set(merged_ground_truth_df["Source"]).intersection(rna_pseudobulk_norm.index)
gt_tgs_in_rna = set(merged_ground_truth_df["Target"]).intersection(rna_pseudobulk_norm.index)
logging.info(f"Ground truth TFs in RNA pseudobulk: {len(gt_tfs_in_rna)} (Example: {list(gt_tfs_in_rna)[:5]})")
logging.info(f"Ground truth TGs in RNA pseudobulk: {len(gt_tgs_in_rna)} (Example: {list(gt_tgs_in_rna)[:5]})")

n_before_rna_filter = len(merged_ground_truth_df)

# Subset the ground truth to only TFs and TGs present in the rna_pseudobulk 
merged_ground_truth_df = merged_ground_truth_df[
    merged_ground_truth_df["Source"].isin(gt_tfs_in_rna) &
    merged_ground_truth_df["Target"].isin(gt_tgs_in_rna)
].copy()

logging.info(
    f"Ground truth edges after RNA TF/TG filtering: "
    f"{len(merged_ground_truth_df):,} / {n_before_rna_filter:,}"
)

# Get the map of TF name to index
tf_name_to_idx = pd.read_csv(config.tf_name_to_idx_cache_path)
tf_name_to_idx["tf_name"] = tf_name_to_idx["tf_name"].str.upper()
tf_name_to_idx = tf_name_to_idx.set_index("tf_name")["tf_idx"].to_dict()

# Only keep ground truth TFs that have embeddings (i.e. were present in the TF-DNA model training data)
gt_tfs_in_embeddings = set(tf_name_to_idx.keys()).intersection(gt_tfs_in_rna)
logging.info(f"Ground truth TFs with embeddings: {len(gt_tfs_in_embeddings)} (Example: {list(gt_tfs_in_embeddings)[:5]})")

n_before_tf_embedding_filter = len(merged_ground_truth_df)

merged_ground_truth_df = merged_ground_truth_df[
    merged_ground_truth_df["Source"].isin(gt_tfs_in_embeddings)
].copy()

logging.info(
    f"Ground truth edges after filtering to TFs with embeddings: "
    f"{len(merged_ground_truth_df):,} / {n_before_tf_embedding_filter:,}"
)

# Create a map of TG name to index for TGs present in the ground truth (and RNA pseudobulk)
tg_id_to_idx = {tg: idx for idx, tg in enumerate(merged_ground_truth_df["Target"].unique())}
    
# Split genes into train/val/test based on chromosome using the GTF reference file
train_genes, val_genes, test_genes = tf_tg_data_builder.split_genes_by_chromosome(
    gene_ref_file,
    train_chroms=train_chroms,
    val_chroms=val_chroms,
    test_chroms=test_chroms
    )

# Subset the ground truth to create train/val/test splits based on the target gene chromosome splits
# (Only keeps TFs and TGs present in the ground truth and RNA pseudobulk, and only keeps TFs with embeddings)
gt_train_df, gt_val_df, gt_test_df = tf_tg_data_builder.create_train_val_test_splits(
    merged_ground_truth_df, train_genes, val_genes, test_genes
)
logging.info(f"After subsetting to TFs with embeddings and TGs in RNA pseudobulk:")
logging.info(f"  - Train interactions: {len(gt_train_df)} (TFs: {gt_train_df['Source'].nunique()}, TGs: {gt_train_df['Target'].nunique()})")
logging.info(f"  - Val interactions: {len(gt_val_df)} (TFs: {gt_val_df['Source'].nunique()}, TGs: {gt_val_df['Target'].nunique()})")
logging.info(f"  - Test interactions: {len(gt_test_df)} (TFs: {gt_test_df['Source'].nunique()}, TGs: {gt_test_df['Target'].nunique()})")

# Create labeled TF-TG datasets for train/val/test splits
# (samples true and false edges according to pct_true_edges and true_false_ratio)
tf_tg_labeled_train_df = tf_tg_data_builder._create_labeled_df(
    gt_train_df,
    pct_true_edges,
    true_false_ratio,
    seed=123,
    tf_name_to_idx=tf_name_to_idx,
    tg_id_to_idx=tg_id_to_idx,
)
tf_tg_labeled_val_df = tf_tg_data_builder._create_labeled_df(
    gt_val_df,
    pct_true_edges,
    true_false_ratio,
    seed=124,
    tf_name_to_idx=tf_name_to_idx,
    tg_id_to_idx=tg_id_to_idx,
)
tf_tg_labeled_test_df = tf_tg_data_builder._create_labeled_df(
    gt_test_df,
    pct_true_edges,
    true_false_ratio,
    seed=125,
    tf_name_to_idx=tf_name_to_idx,
    tg_id_to_idx=tg_id_to_idx,
)

tf_idx_to_name, tg_idx_to_name = create_tf_tg_index_to_name_mappings(tf_name_to_idx, tg_id_to_idx)

tf_names = tf_name_to_idx.keys()
tg_names = tg_id_to_idx.keys()

# Create the peak input data for the TF-DNA model

# Create a map of ATAC peaks to indices in the pseudobulk matrix, filtering to valid chromosomes
dataset_peaks = atac_pseudobulk.index.to_list()

# Only use peaks from standard chromosomes (chr1-chr19 for mm10, chr1-chr22 for hg38) to avoid issues with 
# non-standard chromosomes and contigs
if config.species == "mm10":
    valid_chroms = {f"chr{i}" for i in range(1, 20)}
elif config.species == "hg38":
    valid_chroms = {f"chr{i}" for i in range(1, 23)}

dataset_peaks = [peak for peak in dataset_peaks if peak.split(":", 1)[0] in valid_chroms]
atac_peak_map = {peak: idx for idx, peak in enumerate(dataset_peaks)}

# Create the centered one-hot encoded ATAC peak array for the test set
logging.info("Creating centered one-hot encoded ATAC peak array for the test set...")
atac_peak_array = utils.create_centered_peak_onehot_array(
    peak_ids=dataset_peaks,
    genome_fasta=genome_fasta_path,
    chrom_sizes=utils.load_chrom_sizes(chrom_sizes_path),
    peak_id_to_idx=atac_peak_map,
    flank_size=128,
    dtype=np.uint8,
    pad_out_of_bounds=True,
    num_workers=8,
    show_progress=False,
    chunk_size=10000,
)
atac_peak_tensor = torch.as_tensor(atac_peak_array, dtype=torch.uint8).float()
logging.info(f"ATAC peak tensor shape: {atac_peak_tensor.shape}")

logging.info(f"Constructing TF-TG lookup tables for test set evaluation")
tg_to_peak_info, cell_to_idx, atac_mat, rna_mat, gene_to_rna_idx = prepare_tftg_lookup_tables(
    peak_to_gene=peak_to_gene,
    atac_peak_map=atac_peak_map,
    atac_pseudobulk=atac_pseudobulk,
    rna_pseudobulk_norm=rna_pseudobulk_norm,
    dataset_peaks=dataset_peaks,
    common_cells=common_cells,
    max_precompute_peaks=max_peaks_per_tg,
)
logging.info(f"Number of TGs with at least one peak: {len(tg_to_peak_info)}")
logging.info(f"Example TG to peak info entry: {next(iter(tg_to_peak_info.items()))}")
logging.info(f"Number of common cells: {len(common_cells)}")
logging.info(f"ATAC matrix shape: {atac_mat.shape}, RNA matrix shape: {rna_mat.shape}")

# Determine the maximum number of peaks to consider across all TGs in the dataset 
# to ensure consistent tensor shapes
tf_tg_df = pd.concat([tf_tg_labeled_train_df, tf_tg_labeled_val_df, tf_tg_labeled_test_df], ignore_index=True)

max_peaks_real = max(
    len(tg_to_peak_info.get(tg_name, {}).get("peak_indices", []))
    for tg_name in tf_tg_df["tg_id"]
)

# Check that at least some TGs have peaks within 100kb, otherwise the model will have no signal to learn from
n_tgs_with_peaks = sum(
    len(tg_to_peak_info.get(tg, {}).get("peak_indices", [])) > 0
    for tg in tf_tg_df["tg_id"].unique()
)
    
logging.info(f"TGs with at least one peak within 100kb: {n_tgs_with_peaks:,} / {tf_tg_df['tg_id'].nunique():,}")
logging.info(f"Max peaks per TG after filtering/capping: {max_peaks_real:,}")

# Build the compact TF-TG input dataset for the test set
common_build_kwargs = dict(
    max_cells_per_pair=max_cells_per_pair,
    tg_to_peak_info=tg_to_peak_info,
    cell_to_idx=cell_to_idx,
    atac_mat=atac_mat,
    rna_mat=rna_mat,
    gene_to_rna_idx=gene_to_rna_idx,
    common_cells=common_cells,
    tf_name_to_idx=tf_name_to_idx,
    tg_id_to_idx=tg_id_to_idx,
    max_peaks_real=max_peaks_real
)

logging.info("Building TF-TG input dataset...")
tftg_inputs_train = build_tftg_inputs(
    tf_tg_labeled_train_df,
    seed=125,
    silence=False,
    **common_build_kwargs,
)

logging.info("\nBuilding validation inputs")
tftg_inputs_val = build_tftg_inputs(
    tf_tg_labeled_val_df,
    seed=124,
    **common_build_kwargs,
)

logging.info("\nBuilding test inputs")
tftg_inputs_test = build_tftg_inputs(
    tf_tg_labeled_test_df,
    seed=125,
    **common_build_kwargs,
)

logging.info(f"Wrote training data and metadata to {tf_tg_input_cache_dir}")

class TFTGEdgeBagDataset(Dataset):
    def __init__(
        self,
        inputs,
        *,
        tf_embeddings_tensor,
        tf_mask_tensor,
        atac_peak_tensor,
    ):
        self.inputs = inputs
        self.tf_embeddings_tensor = tf_embeddings_tensor
        self.tf_mask_tensor = tf_mask_tensor
        self.atac_peak_tensor = atac_peak_tensor

    def __len__(self):
        return len(self.inputs["label"])

    def __getitem__(self, idx):
        tf_idx = self.inputs["tf_idx"][idx]
        tg_idx = self.inputs["tg_idx"][idx]

        peak_indices = self.inputs["peak_indices"][idx]          # [P]
        peak_sequences = self.atac_peak_tensor[peak_indices]     # [P, L, 4]

        tf_embedding = self.tf_embeddings_tensor[tf_idx]         # [T, D]
        tf_mask = self.tf_mask_tensor[tf_idx]                    # [T]

        return {
            "label": self.inputs["label"][idx],
            "tf_name": self.inputs["tf_name"][idx],
            "tg_name": self.inputs["tg_name"][idx],
            "cell_ids": self.inputs["cell_ids"][idx],
            "tf_idx": tf_idx,
            "tg_idx": tg_idx,
            "tf_embedding": tf_embedding.float(),
            "tf_mask": tf_mask.bool(),
            "peak_indices": peak_indices,
            "peak_sequences": peak_sequences,
            "peak_mask": self.inputs["peak_mask"][idx].bool(),
            "peak_accessibility": self.inputs["peak_accessibility"][idx].float(),
            "tf_expression": self.inputs["tf_expression"][idx].float(),
            "tg_expression": self.inputs["tg_expression"][idx].float(),
        }
        
def collate_tftg_edge_bags(batch):
    output = {
        "label": torch.stack([b["label"] for b in batch]).float(),

        "tf_idx": torch.stack([b["tf_idx"] for b in batch]).long(),
        "tg_idx": torch.stack([b["tg_idx"] for b in batch]).long(),

        "tf_embedding": torch.stack([b["tf_embedding"] for b in batch]),
        "tf_mask": torch.stack([b["tf_mask"] for b in batch]),

        "peak_indices": torch.stack([b["peak_indices"] for b in batch]),
        "peak_sequences": torch.stack([b["peak_sequences"] for b in batch]),
        "peak_mask": torch.stack([b["peak_mask"] for b in batch]),

        "peak_accessibility": torch.stack([b["peak_accessibility"] for b in batch]),
        "tf_expression": torch.stack([b["tf_expression"] for b in batch]),
        "tg_expression": torch.stack([b["tg_expression"] for b in batch]),

        "tf_name": [b["tf_name"] for b in batch],
        "tg_name": [b["tg_name"] for b in batch],
        "cell_ids": [b["cell_ids"] for b in batch],
    }

    E, C = output["tf_expression"].shape
    output["cell_mask"] = torch.ones(E, C, dtype=torch.bool)

    return output

# Load the lookup tensors
tf_embeddings_tensor = torch.load(
    cell_type_cache_dir / "tf_embeddings.pt",
    weights_only=True,
)
tf_mask_tensor = torch.load(
    cell_type_cache_dir / "tf_masks.pt",
    weights_only=True,
)

# Save compact split inputs
torch.save(tftg_inputs_train, tf_tg_train_cache_path)
torch.save(tftg_inputs_val, tf_tg_val_cache_path)
torch.save(tftg_inputs_test, tf_tg_test_cache_path)

# Save mapping dictionaries and metadata
metadata = {
    "tf_name_to_idx": tf_name_to_idx,
    "tg_id_to_idx": tg_id_to_idx,
    "gene_to_rna_idx": gene_to_rna_idx,
    "cell_to_idx": cell_to_idx,
    "max_peaks_per_tg": max_peaks_per_tg,
    "max_cells_per_pair": max_cells_per_pair,
    "flank_size": peak_flank_size,
    "peak_dtype": "uint8",
    "max_peaks_real": max_peaks_real,
}
with open(tf_tg_metadata_cache_path, "w") as f:
    json.dump(metadata, f, indent=4)

# Save a manifest to keep track of model settings and dataset versions
manifest = {
    "max_peaks_per_tg": max_peaks_per_tg,
    "max_cells_per_pair": max_cells_per_pair,
    "flank_size": peak_flank_size,
    "atac_peak_tensor_dtype": str(atac_peak_tensor.dtype),
    "atac_peak_tensor_shape": list(atac_peak_tensor.shape),
    "tf_embeddings_tensor_shape": list(tf_embeddings_tensor.shape),
    "tf_mask_tensor_shape": list(tf_mask_tensor.shape),
    "n_train_rows": int(len(tftg_inputs_train["label"])),
    "n_val_rows": int(len(tftg_inputs_val["label"])),
    "n_test_rows": int(len(tftg_inputs_test["label"])),
}

with open(tf_tg_manifest_cache_path, "w") as f:
    json.dump(manifest, f, indent=2)

train_dataset = TFTGEdgeBagDataset(
    tftg_inputs_train,
    tf_embeddings_tensor=tf_embeddings_tensor,
    tf_mask_tensor=tf_mask_tensor,
    atac_peak_tensor=atac_peak_tensor
)

val_dataset = TFTGEdgeBagDataset(
    tftg_inputs_val,
    tf_embeddings_tensor=tf_embeddings_tensor,
    tf_mask_tensor=tf_mask_tensor,
    atac_peak_tensor=atac_peak_tensor

)

test_dataset = TFTGEdgeBagDataset(
    tftg_inputs_test,
    tf_embeddings_tensor=tf_embeddings_tensor,
    tf_mask_tensor=tf_mask_tensor,
    atac_peak_tensor=atac_peak_tensor
)

# Create the DataLoaders with the custom collate function for batching edge bags
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=6,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    collate_fn=collate_tftg_edge_bags,
    )

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=6,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    collate_fn=collate_tftg_edge_bags,
    )

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=6,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    collate_fn=collate_tftg_edge_bags,
    )

logging.info(f"Created DataLoader with {len(train_dataset):,} samples and batch size {train_loader.batch_size} ({len(train_loader):,} batches)")

tf_dna_model_chkpt = tf_dna_model_checkpoints[cell_type]

# Load the TF→TG model
tf_tg_model = create_new_tf_tg_regulation_model(
    tf_bind_model_path=tf_dna_model_chkpt,
    tf_embeddings_tensor=tf_embeddings_tensor,
    tf_mask_tensor=tf_mask_tensor
)

# Generate the model predictions for the test set and create a DataFrame with TF names, TG names, and predicted scores
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = torch.nn.BCEWithLogitsLoss()

score_threshold = 0.5
pooling_mode = "lse"
pooling_temperature = 1.0

logging.info("\nStarting Lightning training...")

lit_model = tf_to_tg_module.LitTFTGRegulationModel(
    model=tf_tg_model,
    lr=1e-4,
    weight_decay=1e-4,
    pos_weight=None,
    pooling_mode=pooling_mode,
    pooling_temperature=pooling_temperature,
    enable_timing_sync=False,
)

checkpoint_callback = ModelCheckpoint(
    dirpath=output_dir,
    filename="epoch={epoch:02d}-val_auroc={val/auroc:.4f}-val_loss={val/loss:.4f}",
    monitor="val/auroc",
    mode="max",
    save_top_k=500,
    save_last=True,
    auto_insert_metric_name=False,
)

early_stopping_callback = EarlyStopping(
    monitor="val/loss",
    mode="min",
    patience=15,
)

lr_monitor = LearningRateMonitor(logging_interval="epoch")

wandb_logger = WandbLogger(
    project="tf_tg_regulation_prediction",
    name=run_name,
    save_dir=output_dir,
)

wandb_logger.log_hyperparams({
    "sample_name": sample_name,
    "epochs": epochs,
    "batch_size": batch_size,
    "num_batches": len(train_loader),
    "num_gpus": num_gpus,
    "num_nodes": num_nodes,
    "job_id": job_id,
    "run_name": run_name,
    "sample_pairs": len(train_dataset),
    "max_peaks_per_tg": max_peaks_per_tg,
    "max_cells_per_pair": max_cells_per_pair,
    "pct_true_edges": pct_true_edges,
    "true_false_ratio": true_false_ratio,
    "pooling_mode": pooling_mode,
    "pooling_temperature": pooling_temperature,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "flank_size": 128,
    "max_precompute_peaks": max_peaks_per_tg,
    "persistent_workers": True,
    "tf_bind_model_path": str(tf_dna_model_chkpt),
})

world_size = int(
    os.environ.get(
        "WORLD_SIZE",
        os.environ.get("SLURM_NTASKS", "1"),
    )
)

use_ddp = world_size > 1

logging.info(f"Num GPUs: {world_size} | Batch size: {batch_size}")
logging.info(f"Num steps per epoch: {len(train_loader)}")

strategy=DDPStrategy(
    process_group_backend="nccl",
    find_unused_parameters=False,
) if use_ddp else "auto"

trainer = pl.Trainer(
    max_epochs=epochs,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    num_nodes=1,
    strategy=strategy,
    precision="16-mixed",
    logger=wandb_logger,
    callbacks=[
        TQDMProgressBar(refresh_rate=25),
        checkpoint_callback,
        early_stopping_callback,
        lr_monitor,
    ],
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
    log_every_n_steps=10,
    default_root_dir=output_dir,
    enable_progress_bar=True,
    enable_checkpointing=True,
    check_val_every_n_epoch=1,
)

trainer.fit(
    lit_model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader,
)


