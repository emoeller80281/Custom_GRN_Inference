
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
    max_peaks_per_tg=None,
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

    tf_indices = []
    tg_indices = []
    peak_indices_all = []
    peak_access_all = []
    peak_dist_all = []
    peak_masks_all = []
    tf_expr_all = []
    tg_expr_all = []

    common_cells = list(common_cells)
    n_common_cells = len(common_cells)

    # rng.choice() re-runs np.asarray() on its first argument every call, so passing a
    # Python list of thousands of cell-name strings re-converted the whole list per edge:
    # measured 417 us/edge against 8.3 us when drawing positions from an int, which was
    # ~77% of the entire build. Draw positions and index once instead. This is the same
    # draw, not an approximation -- Generator.choice picks positions and then indexes, so
    # rng.choice(names, k) == names[rng.choice(len(names), k)] for a given seed (verified
    # bit-identical over 2,000 consecutive draws).
    common_cells_arr = np.asarray(common_cells)
    common_cell_rows = np.asarray(
        [cell_to_idx[c] for c in common_cells], dtype=np.int64
    )
    take_all_cells = max_cells_per_pair is None or max_cells_per_pair >= n_common_cells

    # Per-TG padded peak bags. Every TF is paired with every TG, so without this the same
    # three np.pad calls repeat once per TF for each TG -- another ~40 us/edge.
    tg_bag_cache = {}

    n_total = len(tf_tg_df)
    log_every = max(1, n_total // 50)

    build_started = time.time()

    for i, row in enumerate(tf_tg_df.itertuples(index=False), start=1):
        if silence == False:
            if i == 1 or i % log_every == 0 or i == n_total:
                # Rate and ETA
                elapsed = time.time() - build_started
                rate = i / elapsed if elapsed > 0 else 0.0
                eta = (n_total - i) / rate if rate > 0 else 0.0
                logging.info(
                    f"Building compact TF-TG edges: {100 * i / n_total:.1f}% "
                    f"({i:,}/{n_total:,}) at {rate:,.0f} edges/s, "
                    f"elapsed {elapsed / 60:.1f}m, ETA {eta / 60:.1f}m"
                )

        tf_name = row.Source
        tg_name = row.Target

        tf_idx = tf_name_to_idx.get(tf_name)
        tg_idx = tg_id_to_idx.get(tg_name)

        if tf_idx is None or tg_idx is None:
            continue

        bag = tg_bag_cache.get(tg_name, 0)
        if bag == 0:
            peak_info = tg_to_peak_info.get(tg_name)
            if peak_info is None:
                tg_bag_cache[tg_name] = None
                continue

            peak_indices_real = list(peak_info["peak_indices"])
            peak_dst_real = list(peak_info["peak_distances"])

            n_peaks = len(peak_indices_real)
            if n_peaks == 0:
                tg_bag_cache[tg_name] = None
                continue

            peak_indices = np.asarray(peak_indices_real, dtype=np.int64)
            peak_dst = np.asarray(peak_dst_real, dtype=np.float32)
            peak_mask = np.ones(n_peaks, dtype=bool)

            if n_peaks < max_peaks_real:
                pad_len = max_peaks_real - n_peaks
                peak_indices = np.pad(peak_indices, (0, pad_len), constant_values=0)
                peak_dst = np.pad(peak_dst, (0, pad_len), constant_values=0.0)
                peak_mask = np.pad(peak_mask, (0, pad_len), constant_values=False)

            # Shared by every edge on this TG from here on, and np.stack copies them into
            # the output, so freeze them rather than trust that nobody writes through.
            for arr in (peak_indices, peak_dst, peak_mask):
                arr.flags.writeable = False

            bag = (peak_indices, peak_dst, peak_mask, n_peaks,
                   np.asarray(peak_indices_real, dtype=np.int64))
            tg_bag_cache[tg_name] = bag
        elif bag is None:
            continue

        peak_indices, peak_dst, peak_mask, n_peaks, peak_rows = bag

        # Sample cells
        if take_all_cells:
            sampled_cells = common_cells
            sampled_cell_indices = common_cell_rows
        else:
            sampled_positions = rng.choice(
                n_common_cells,
                size=max_cells_per_pair,
                replace=False,
            )
            sampled_cells = common_cells_arr[sampled_positions].tolist()
            sampled_cell_indices = common_cell_rows[sampled_positions]

        C = len(sampled_cell_indices)
        P = max_peaks_real

        # ATAC accessibility: [C, P]. This is what np.ix_ builds internally, without
        # re-deriving the open mesh (and the peak index array) on every edge.
        peak_acc_matrix = np.zeros((C, P), dtype=np.float32)
        peak_acc_matrix[:, :n_peaks] = atac_mat[
            peak_rows[:, None], sampled_cell_indices[None, :]
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

        tf_indices.append(tf_idx)
        tg_indices.append(tg_idx)
        peak_indices_all.append(peak_indices)
        peak_access_all.append(peak_acc_matrix)
        peak_dist_all.append(peak_dst)
        peak_masks_all.append(peak_mask)
        tf_expr_all.append(tf_expr_vals)
        tg_expr_all.append(tg_expr_vals)

    return {
        "tf_name": tf_names,
        "tg_name": tg_names,
        "cell_ids": cell_ids_all,

        # No ground truth exists for this all-pairs inference set. TFTGEdgeBagDataset
        # (shared with training/eval, where real labels matter) still expects the key,
        # so fill it with a placeholder that is never read downstream -- predictions
        # come from the model's output scores, not batch["label"].
        "label": torch.zeros(len(tf_indices), dtype=torch.float32),

        "tf_idx": torch.tensor(tf_indices, dtype=torch.long),
        "tg_idx": torch.tensor(tg_indices, dtype=torch.long),

        "peak_indices": torch.tensor(np.stack(peak_indices_all), dtype=torch.long),
        "peak_accessibility": torch.tensor(np.stack(peak_access_all), dtype=torch.float32),
        "peak_mask": torch.tensor(np.stack(peak_masks_all), dtype=torch.bool),
        "peak_distance": torch.tensor(np.stack(peak_dist_all), dtype=torch.float32),

        "tf_expression": torch.tensor(np.stack(tf_expr_all), dtype=torch.float32),
        "tg_expression": torch.tensor(np.stack(tg_expr_all), dtype=torch.float32),
    }

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
tg_to_peak_info, cell_to_idx, atac_mat, rna_mat, gene_to_rna_idx = prepare_tftg_lookup_tables(
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
if not tftg_inputs_path.is_file():
    tftg_inputs = build_tftg_inputs(
        all_edge_combo_df,
        seed=123,
        **common_build_kwargs,
    )
    torch.save(tftg_inputs, tftg_inputs_path)
else:
    tftg_inputs = torch.load(tftg_inputs_path, weights_only=True)
    if "label" not in tftg_inputs:
        # Cache predates the placeholder "label" field -- patch it in rather than
        # forcing a rebuild of the (very large) cached tensor.
        tftg_inputs["label"] = torch.zeros(len(tftg_inputs["tf_idx"]), dtype=torch.float32)
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