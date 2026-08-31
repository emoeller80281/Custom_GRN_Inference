
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import seaborn as sns
import scanpy as sc
import muon as mu
import time
import gtfparse

import argparse

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER")
DATA_DIR = PROJECT_DIR / "cached_data"
CHKPT_DIR = PROJECT_DIR / "checkpoints"
RESULT_DIR = PROJECT_DIR / "testing_results"

output_dir = PROJECT_DIR / "mouse_development_testing"

sys.path.append(str(PROJECT_DIR))

import warnings

from tqdm import tqdm
from torch.utils.data import DataLoader

import models.tf_to_tg as tf_to_tg_module
from scripts.train_tf_to_tg_model import TFTGEdgeBagDataset, collate_tftg_edge_bags
import config
import utils

if torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.cache_size_limit = 128
else:
    device = torch.device("cpu")
    logging.info("Using CPU")

warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`.*",
    category=FutureWarning,
)

font_path = PROJECT_DIR / "fonts" / "Arial.ttf"

fm.fontManager.addfont(font_path)
arial_font = fm.FontProperties(fname=font_path).get_name()

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = [arial_font, "Helvetica", "Liberation Sans", "DejaVu Sans"]

logging.info(f"Using font: {arial_font}")

rng = np.random.default_rng()

def load_pseudobulk_data(cell_type, sample_name):
    input_data_dir = Path(f"/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/sample_input_data/{cell_type}/{sample_name}")
    
    # Read in the ATAC and RNA pseudobulk data, and the peak-to-gene distance file
    atac_pseudobulk = pd.read_parquet(input_data_dir / "RE_pseudobulk.parquet")
    peak_to_gene_distance = pd.read_parquet(input_data_dir / "peak_to_gene_dist.parquet")
    rna_pseudobulk = pd.read_parquet(input_data_dir / "TG_pseudobulk.parquet")
    
    # Keep only ATAC peaks that are present in the peak-to-gene distance table
    valid_peak_ids = set(peak_to_gene_distance["peak_id"])

    atac_pseudobulk = atac_pseudobulk.loc[
        atac_pseudobulk.index.isin(valid_peak_ids)
    ].copy()
    
    rna_pseudobulk_norm = rna_pseudobulk.copy()
    rna_pseudobulk_norm.index = rna_pseudobulk_norm.index.str.upper()

    common_cells = sorted(set(rna_pseudobulk_norm.columns) & set(atac_pseudobulk.columns))
    
    if len(common_cells) == 0:
        raise ValueError(
            "No common pseudobulk cell columns between RNA and ATAC matrices."
        )
        
    peak_to_gene = peak_to_gene_distance.copy()
    peak_to_gene["target_id_norm"] = peak_to_gene["target_id"].str.upper()
    
    return atac_pseudobulk, peak_to_gene, rna_pseudobulk_norm

def generate_atac_peak_tensor(atac_pseudobulk, species="mm10"):
    dataset_peaks = atac_pseudobulk.index.to_list()
    atac_peak_map = {peak: idx for idx, peak in enumerate(dataset_peaks)}

    PROJECT_DATA_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data")
    
    genome_fasta_path = PROJECT_DATA_DIR / "genome_data" / "reference_genome" / species / f"{species}.fa"
    chrom_sizes_path = PROJECT_DATA_DIR / "genome_data" / "reference_genome" / species / f"{species}.chrom.sizes"

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
    atac_peak_tensor = torch.as_tensor(atac_peak_array, dtype=torch.uint8)
    atac_peak_tensor = atac_peak_tensor.float()
    
    return atac_peak_tensor

# prepare_tftg_lookup_tables/build_tftg_inputs used to be duplicated here (drifted from
# utils.py's versions -- e.g. this file's build used row.Source/row.Target and a
# placeholder all-zero label, since there's no ground truth for the all-pairs prediction
# universe). Now calling utils.prepare_tftg_lookup_tables/utils.build_tftg_inputs directly
# (see the call site below) so the compact edge-bag format change lands here too instead
# of needing a second, easily-forgotten fix.
parser = argparse.ArgumentParser(
    description=(
        "Score TF-TG edges for a chosen mESC sample's metacells using the "
        "hepatocytes_1-trained TF-TG model."
    )
)
parser.add_argument(
    "--sample_name",
    type=str,
    default="E8.5_rep1",
    help=(
        "mESC sample directory under data/sample_input_data/mESC/ to load "
        "pseudobulk RNA/ATAC data from, e.g. E8.5_rep1, E8.5_CRISPR_T_KO, "
        "E8.5_CRISPR_T_WT. Default: E8.5_rep1."
    ),
)
parser.add_argument(
    "--metacell_file",
    type=str,
    default=None,
    help=(
        "Path to a comma-separated file of metacell/barcode IDs (a subset "
        "of --sample_name's own pseudobulk columns) to restrict scoring "
        "to. Relative paths are resolved against mouse_development_testing/. "
        "Defaults to 'E8.5_rep1_NMP_metacells.txt' only when "
        "--sample_name=E8.5_rep1; required otherwise, since each sample's "
        "metacell list lives in its own barcode namespace."
    ),
)
parser.add_argument(
    "--prediction_output_file",
    type=str,
    default=None,
    help=(
        "Output CSV path for the predicted TF-TG edge scores. Relative "
        "paths are resolved against mouse_development_testing/. Defaults "
        "to 'hepatocyte_model_vs_{sample_name}_NMP_metacell_GRN.csv'."
    ),
)
args = parser.parse_args()

sample_name = args.sample_name

if args.metacell_file is not None:
    metacell_file = Path(args.metacell_file)
    if not metacell_file.is_absolute():
        metacell_file = output_dir / metacell_file
elif sample_name == "E8.5_rep1":
    metacell_file = output_dir / "E8.5_rep1_NMP_metacells.txt"
else:
    parser.error(
        f"--metacell_file is required when --sample_name is not "
        f"E8.5_rep1 (got --sample_name={sample_name!r})"
    )

if args.prediction_output_file is not None:
    prediction_output_file = Path(args.prediction_output_file)
    if not prediction_output_file.is_absolute():
        prediction_output_file = output_dir / prediction_output_file
else:
    prediction_output_file = (
        output_dir / f"hepatocyte_model_vs_{sample_name}_NMP_metacell_GRN.csv"
    )

hepatocyte_tf_tg_model = utils.find_latest_checkpoint(CHKPT_DIR, "mouse_hepatocytes", "hepatocytes_1", training_number="3709466")

logging.info(f"loading sample data for {sample_name}")
atac_pseudobulk, peak_to_gene, rna_pseudobulk_norm = load_pseudobulk_data("mESC", sample_name)

logging.info(f"Loading NMP metacell barcodes from {metacell_file}")
metacell_df = pd.read_csv(metacell_file, header=None, index_col=None)
nmp_metacell_list = metacell_df.iloc[0, :].to_list()

# Remove NaNs and make sure IDs are strings
nmp_metacell_list = [
    str(cell)
    for cell in nmp_metacell_list
    if pd.notna(cell)
]

logging.info(f"Number of NMP cells: {len(nmp_metacell_list)}")
logging.info(f"RNA pseudobulk cells: {rna_pseudobulk_norm.shape[1]}")
logging.info(f"ATAC pseudobulk cells: {atac_pseudobulk.shape[1]}")

# Find NMP cells present in BOTH RNA and ATAC
common_cells = sorted(
    set(nmp_metacell_list)
    & set(rna_pseudobulk_norm.columns)
    & set(atac_pseudobulk.columns)
)

logging.info(f"NMP cells present in both RNA and ATAC: {len(common_cells)}")

# Subset both datasets
rna_pseudobulk_norm = rna_pseudobulk_norm.loc[:, common_cells]
atac_pseudobulk = atac_pseudobulk.loc[:, common_cells]

logging.info("generating ATAC peak tensor")
atac_peak_tensor = generate_atac_peak_tensor(atac_pseudobulk, species="mm10")

logging.info("Creating maps")
# NOTE: must preserve list order (not a set) -- atac_peak_map below has to match the
# row order generate_atac_peak_tensor() used to build atac_peak_tensor, since peak_indices
# computed from atac_peak_map are used to index directly into atac_peak_tensor.
dataset_tgs = list(dict.fromkeys(rna_pseudobulk_norm.index.to_list()))
dataset_peaks = atac_pseudobulk.index.to_list()

tg_id_to_idx = {tg: idx for idx, tg in enumerate(dataset_tgs)}
atac_peak_map = {peak: idx for idx, peak in enumerate(dataset_peaks)}

logging.info("Loading cached TF name to index map")
training_cache_dir = config.cell_type_cache_dir("mESC")
tf_dna_cache_dir = config.tf_dna_cache_dir_for_cell_type("mESC")
tf_name_to_idx_cache_path = tf_dna_cache_dir / "tf_name_to_idx.csv"

tf_name_to_idx = pd.read_csv(tf_name_to_idx_cache_path)
tf_name_to_idx = tf_name_to_idx[tf_name_to_idx["tf_name"].str.upper().isin(tg_id_to_idx.keys())]
tf_name_to_idx["tf_name"] = tf_name_to_idx["tf_name"].str.upper()
tf_name_to_idx = tf_name_to_idx.set_index("tf_name")["tf_idx"].to_dict()

candidate_tfs = sorted(tf_name_to_idx.keys())
candidate_items = sorted(tg_id_to_idx.keys())

logging.info("Creating all edge combo DataFrame")
all_edge_combo_df = (
    pd.MultiIndex
    .from_product([candidate_tfs, candidate_items], names=["Source", "Target"])
    .to_frame(index=False)
)

logging.info("Preparing TF/TG lookup tables")
tg_to_peak_info, cell_to_idx, atac_mat, rna_mat, gene_to_rna_idx = utils.prepare_tftg_lookup_tables(
    peak_to_gene=peak_to_gene,
    atac_peak_map=atac_peak_map,
    atac_pseudobulk=atac_pseudobulk,
    rna_pseudobulk_norm=rna_pseudobulk_norm,
    dataset_peaks=dataset_peaks,
    common_cells=common_cells,
    max_precompute_peaks=25,
)

logging.info("Determining the max number of peaks per TG")
max_peaks_real = max(
    len(tg_to_peak_info.get(tg_name, {}).get("peak_indices", []))
    for tg_name in all_edge_combo_df["Target"]
)

logging.info("Building TF-TG model inputs")
common_build_kwargs = dict(
    max_peaks_per_tg=25,
    max_cells_per_pair=50,
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

tftg_inputs_path = output_dir / f"{sample_name}_NMP_lineage_tftg_inputs.pt"
tftg_inputs = (
    torch.load(tftg_inputs_path, weights_only=True) if tftg_inputs_path.is_file() else None
)

# Rebuild if missing, or if it predates the compact edge-bag format (peak_accessibility/
# tf_expression/tg_expression are now gathered from atac_mat/rna_mat at read time instead
# of stored per-edge -- see utils.build_tftg_inputs; "cell_indices" is that format's
# signature key). Replaces the old missing-"label" patch with the same idea: don't try to
# patch a stale cache, just rebuild it.
if tftg_inputs is None or "cell_indices" not in tftg_inputs:
    logging.info("Building TF-TG model inputs")
    # utils.build_tftg_inputs expects tf_name/tg_id/label columns. No ground truth exists
    # for this all-pairs inference universe, so label is a placeholder never read
    # downstream -- predictions come from the model's output scores, not batch["label"].
    edge_df = all_edge_combo_df.rename(columns={"Source": "tf_name", "Target": "tg_id"})
    edge_df["label"] = 0.0

    tftg_inputs = utils.build_tftg_inputs(
        edge_df,
        seed=123,
        **common_build_kwargs,
    )
    torch.save(tftg_inputs, tftg_inputs_path)
logging.info("Done!")



def generate_model_predictions(model, data_loader, device, tf_idx_to_name, tg_idx_to_name):
    pooling_mode = "lse"
    pooling_temperature = 1.0

    model = model.to(device)
    model.eval()

    # The compute-capability autocast-dtype picker that used to live here is gone:
    # predictions are scored in fp32 now, so there is no dtype to choose. Kept for the
    # record because the reasoning was subtle -- torch.cuda.is_bf16_supported() returns
    # True on a V100 (7.0) by counting emulated bf16, so it selected bf16 on hardware
    # with no bf16 tensor cores, and Inductor then skipped compiling the model entirely.
    # That trade-off no longer applies.

    tf_indices_list = []
    tg_indices_list = []
    all_scores = []

    n_batches = len(data_loader)

    with torch.inference_mode():
        iterator = iter(data_loader)
        # Time-based throttling (mininterval), not count-based (miniters): torch.compile
        # being active anywhere in this loop defeats miniters entirely -- confirmed by
        # direct reproduction, even with a compile that never recompiles after its first
        # call, so it isn't about graph breaks/recompiles specifically. mininterval
        # survives it. This also degrades better than a fixed update-every-N-batches
        # count would given the per-batch cost here is wildly non-uniform (compile
        # warm-up, GPU contention), where a positive time interval still spaces prints
        # out sensibly and count-based miniters would not, even if it did throttle.
        pbar = tqdm(
            total=n_batches,
            desc="Evaluating",
            ncols=100,
            mininterval=2.0,
        )
        while True:
            try:
                batch = next(iterator)
            except StopIteration:
                break

            tf_indices = batch["tf_idx"].detach().cpu().numpy().ravel()
            tg_indices = batch["tg_idx"].detach().cpu().numpy().ravel()

            batch = tf_to_tg_module.move_batch_to_device(batch, device)

            # Score in fp32. enabled=False forces fp32 regardless of any ambient
            # autocast, which is equivalent to removing this block here (no caller
            # currently wraps it) but stays correct if one ever does.
            #
            # Measured on this model: bf16 vs fp32 predictions for the same
            # checkpoint on mESC/E7.5_rep1 correlate only 0.516 (max score diff
            # 0.816) and cost 0.031-0.036 AUPRC against external methods that are
            # loaded from file and therefore unaffected. Quantising only TETHER
            # while its competitors are exact makes every such comparison unfair.
            with torch.autocast(device_type="cuda", enabled=False):
                edge_logits, _ = model(
                    tf_embedding=batch.get("tf_embedding", None),
                    tf_mask=batch.get("tf_mask", None),
                    peak_sequences=batch["peak_sequences"],
                    peak_accessibility=batch["peak_accessibility"],
                    peak_distance=batch["peak_distance"],
                    tf_expression=batch["tf_expression"],
                    tg_expression=batch["tg_expression"],
                    peak_mask=batch.get("peak_mask", None),
                    cell_mask=batch["cell_mask"],
                    pooling_mode=pooling_mode,
                    pooling_temperature=pooling_temperature,
                    tf_idx=batch.get("tf_idx", None),
                )

            scores = torch.sigmoid(edge_logits.float())

            scores_host = scores.detach().cpu().numpy().ravel()

            tf_indices_list.append(tf_indices)
            tg_indices_list.append(tg_indices)
            all_scores.append(scores_host)

            pbar.update(1)

        pbar.close()

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

def attach_tf_embedding_table(lit_model, tf_embeddings_device, tf_mask_device):
    """
    Give a loaded model its device-resident TF embedding table.

    torch.compile wraps the core model in an OptimizedModule, so reach through
    _orig_mod when present to register the buffers on the real module. The tensors
    are passed in already on-device so the own and cross models share one copy
    rather than each holding their own ~2 GB.
    """
    core_model = getattr(lit_model.model, "_orig_mod", lit_model.model)
    core_model.set_tf_embedding_table(tf_embeddings_device, tf_mask_device)
    return lit_model


tf_dna_model_chkpt = config.tf_dna_model_checkpoints["mESC"]

# Load the lookup tensors
tf_embeddings_tensor = torch.load(
    tf_dna_cache_dir / "tf_embeddings.pt",
    weights_only=True,
)
tf_mask_tensor = torch.load(
    tf_dna_cache_dir / "tf_masks.pt",
    weights_only=True,
)

dataset = TFTGEdgeBagDataset(
    tftg_inputs,
    tf_embeddings_tensor=tf_embeddings_tensor,
    tf_mask_tensor=tf_mask_tensor,
    atac_peak_tensor=atac_peak_tensor,
    atac_mat=atac_mat,
    rna_mat=rna_mat,
    gene_to_rna_idx=gene_to_rna_idx,
    return_tf_indices=True,
)

# Create the PyTorch DataLoader for the test set
num_workers = 8
loader = DataLoader(
    dataset,
    batch_size=1024,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
    collate_fn=collate_tftg_edge_bags,
)

tf_embeddings_device = tf_embeddings_tensor.to(device).float()
tf_mask_device = tf_mask_tensor.to(device).bool()

tf_tg_model = utils.load_tf_tg_regulation_model(
    tf_dna_model_chkpt,
    hepatocyte_tf_tg_model,
    tf_embeddings_tensor,
    tf_mask_tensor,
    tf_peak_chunk_size=1024,
    compile_model=True,
    device=device
    )

attach_tf_embedding_table(tf_tg_model, tf_embeddings_device, tf_mask_device)

tf_idx_to_name = {idx: name for name, idx in tf_name_to_idx.items()}
tg_idx_to_name = {idx: name for name, idx in tg_id_to_idx.items()}

prediction_df = generate_model_predictions(tf_tg_model.model, loader, device, tf_idx_to_name, tg_idx_to_name)

logging.info(f"Saving predictions to {prediction_output_file}")
prediction_df.to_csv(prediction_output_file)