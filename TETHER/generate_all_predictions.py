import sys
import time
import resource
import threading
import psutil
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER")
DATA_DIR = PROJECT_DIR / "cached_data"
CHKPT_DIR = PROJECT_DIR / "checkpoints"
CHKPT_COPY_DIR = PROJECT_DIR / "checkpoints copy"
RESULT_DIR = PROJECT_DIR / "testing_results"

sys.path.append(str(PROJECT_DIR))

import models.tf_to_tg as tf_to_tg_module
import scripts.build_tf_to_tg_train_data as tf_tg_data_builder
from scripts.train_tf_to_tg_model import TFTGEdgeBagDataset, collate_tftg_edge_bags
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

all_evaluation_plot_dir = PROJECT_DIR / "plots"
all_evaluation_plot_dir.mkdir(exist_ok=True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

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

def _process_tree():
    """This process plus any live children (the DataLoader workers)."""
    proc = psutil.Process()
    try:
        return [proc] + proc.children(recursive=True)
    except psutil.Error:
        return [proc]


def _tree_cpu_snapshot():
    """Per-PID cumulative (user, system) CPU seconds across the process tree."""
    snapshot = {}
    for proc in _process_tree():
        try:
            times = proc.cpu_times()
            snapshot[proc.pid] = (times.user, times.system)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return snapshot


class _TreeSampler(threading.Thread):
    """Poll RSS and CPU across the process tree for the duration of a phase.

    Sampling rather than reading endpoints, for two reasons:
      * ru_maxrss is a monotonic high-water mark for this process alone, so a per-phase
        peak cannot be obtained by differencing it, and it cannot see child processes.
      * A child that exits before the phase ends disappears from the tree, taking its
        CPU time with it. Keeping the last reading seen per PID retains it.
    """

    def __init__(self, interval=0.2):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak_rss_bytes = 0
        self.used_pss = False
        # pid -> most recent (user, system) cumulative CPU seconds observed
        self.cpu_last_seen = {}
        self._stop_event = threading.Event()

    def poll(self):
        total_rss = 0
        for proc in _process_tree():
            try:
                # PSS, not RSS: DataLoader workers are forked and share most of their
                # pages with the parent copy-on-write, so summing RSS across the tree
                # counts those shared pages once per process. PSS divides each shared
                # page by the number of processes mapping it, so the tree total is a
                # real memory figure rather than one that can exceed the node's RAM.
                try:
                    total_rss += proc.memory_full_info().pss
                    self.used_pss = True
                except (psutil.AccessDenied, AttributeError):
                    total_rss += proc.memory_info().rss
                times = proc.cpu_times()
                self.cpu_last_seen[proc.pid] = (times.user, times.system)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        self.peak_rss_bytes = max(self.peak_rss_bytes, total_rss)

    def run(self):
        while not self._stop_event.is_set():
            self.poll()
            self._stop_event.wait(self.interval)

    def stop(self):
        self._stop_event.set()
        self.join(timeout=5)
        # Final reading so work since the last poll is not lost
        self.poll()

    def cpu_since(self, baseline):
        """Total tree CPU consumed relative to a per-PID baseline snapshot."""
        user = system = 0.0
        for pid, (last_user, last_system) in self.cpu_last_seen.items():
            base_user, base_system = baseline.get(pid, (0.0, 0.0))
            user += max(0.0, last_user - base_user)
            system += max(0.0, last_system - base_system)
        return user, system


class ResourceProbe:
    """Capture `/usr/bin/time -v` style resource usage for one phase of work.

    /usr/bin/time is not installed on this cluster, so the equivalent counters are
    read in-process from getrusage(RUSAGE_SELF). Two things getrusage cannot give us
    are filled in separately:

      * DataLoader workers are children, and RUSAGE_CHILDREN only counts children that
        have been reaped -- with persistent_workers=True they stay alive, so their CPU
        time is invisible to rusage. psutil reads the whole tree instead.
      * ru_maxrss only ever rises, so a per-phase peak needs sampling, not differencing.
        The cumulative high-water mark is still reported, clearly labelled.
    """

    def __init__(self, label, device, sample_interval=0.2):
        self.label = label
        self.device = device
        self.sample_interval = sample_interval
        self.stats = {}

    def __enter__(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self._rusage_start = resource.getrusage(resource.RUSAGE_SELF)
        self._tree_cpu_baseline = _tree_cpu_snapshot()
        self._sampler = _TreeSampler(self.sample_interval)
        self._sampler.start()
        self._wall_start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # CUDA work is queued asynchronously; without this we would time the queue,
        # not the compute.
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - self._wall_start
        rusage_end = resource.getrusage(resource.RUSAGE_SELF)
        self._sampler.stop()

        user_seconds = rusage_end.ru_utime - self._rusage_start.ru_utime
        system_seconds = rusage_end.ru_stime - self._rusage_start.ru_stime
        tree_user, tree_system = self._sampler.cpu_since(self._tree_cpu_baseline)

        self.stats = {
            "phase": self.label,
            "wall_seconds": round(wall_seconds, 3),
            "user_seconds": round(user_seconds, 3),
            "system_seconds": round(system_seconds, 3),
            "cpu_percent": round(100.0 * (user_seconds + system_seconds) / wall_seconds, 1) if wall_seconds > 0 else 0.0,
            "tree_user_seconds": round(tree_user, 3),
            "tree_system_seconds": round(tree_system, 3),
            "tree_cpu_percent": round(100.0 * (tree_user + tree_system) / wall_seconds, 1) if wall_seconds > 0 else 0.0,
            # True peak for this phase over main process + DataLoader workers, using
            # PSS so pages shared with forked workers are not counted more than once
            "peak_rss_tree_kb": self._sampler.peak_rss_bytes // 1024,
            "peak_rss_tree_metric": "pss" if self._sampler.used_pss else "rss",
            # Monotonic high-water mark for this process only, as /usr/bin/time reports
            "max_rss_self_kb": rusage_end.ru_maxrss,
            "major_page_faults": rusage_end.ru_majflt - self._rusage_start.ru_majflt,
            "minor_page_faults": rusage_end.ru_minflt - self._rusage_start.ru_minflt,
            "voluntary_ctx_switches": rusage_end.ru_nvcsw - self._rusage_start.ru_nvcsw,
            "involuntary_ctx_switches": rusage_end.ru_nivcsw - self._rusage_start.ru_nivcsw,
            "fs_inputs": rusage_end.ru_inblock - self._rusage_start.ru_inblock,
            "fs_outputs": rusage_end.ru_oublock - self._rusage_start.ru_oublock,
        }

        if self.device.type == "cuda":
            self.stats["gpu_peak_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            self.stats["gpu_peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 1024**2, 1)
        else:
            self.stats["gpu_peak_allocated_mb"] = np.nan
            self.stats["gpu_peak_reserved_mb"] = np.nan

        self.log()
        return False

    def log(self):
        s = self.stats
        logging.info(
            f"\n=== Resource usage: {self.label} ===\n"
            f"  Elapsed (wall clock) time (s):        {s['wall_seconds']}\n"
            f"  User time (s):                        {s['user_seconds']}\n"
            f"  System time (s):                      {s['system_seconds']}\n"
            f"  Percent of CPU this phase got:        {s['cpu_percent']}%\n"
            f"  Process-tree user/system time (s):    {s['tree_user_seconds']} / {s['tree_system_seconds']} "
            f"({s['tree_cpu_percent']}% -- includes DataLoader workers)\n"
            f"  Peak {s['peak_rss_tree_metric'].upper()}, process tree (kbytes):      {s['peak_rss_tree_kb']:,}\n"
            f"  Maximum resident set size (kbytes):   {s['max_rss_self_kb']:,} (cumulative high-water, this process)\n"
            f"  Major (requiring I/O) page faults:    {s['major_page_faults']:,}\n"
            f"  Minor (reclaiming a frame) faults:    {s['minor_page_faults']:,}\n"
            f"  Voluntary context switches:           {s['voluntary_ctx_switches']:,}\n"
            f"  Involuntary context switches:         {s['involuntary_ctx_switches']:,}\n"
            f"  File system inputs / outputs:         {s['fs_inputs']:,} / {s['fs_outputs']:,}\n"
            f"  GPU peak allocated / reserved (MB):   {s['gpu_peak_allocated_mb']} / {s['gpu_peak_reserved_mb']}"
        )


def generate_model_predictions(model, data_loader, device, tf_idx_to_name, tg_idx_to_name):
    pooling_mode = "lse"
    pooling_temperature = 1.0

    model = model.to(device)
    model.eval()

    # if device.type == "cuda":
    #     model = torch.compile(model, mode="reduce-overhead")

    # Pick the autocast dtype from the hardware. bf16 needs compute capability 8.0+;
    # on a V100 (7.0) Inductor logs "does not support bfloat16 compilation natively,
    # skipping" and silently gives up on compiling the model at all, so the run falls
    # back to eager after paying the full tracing cost. fp16 is also what these models
    # were trained under (Lightning precision="16-mixed").
    if device.type == "cuda":
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        logging.info(
            f"Autocast dtype: {autocast_dtype} on {torch.cuda.get_device_name(0)} "
            f"(compute capability {'.'.join(str(c) for c in torch.cuda.get_device_capability(0))})"
        )
    else:
        autocast_dtype = torch.float32

    tf_indices_list = []
    tg_indices_list = []
    all_scores = []

    with torch.inference_mode():
        for batch in tqdm(data_loader, desc="Evaluating", ncols=100):
            tf_indices = batch["tf_idx"].detach().cpu().numpy().ravel()
            tg_indices = batch["tg_idx"].detach().cpu().numpy().ravel()

            batch = tf_to_tg_module.move_batch_to_device(batch, device)

            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(device.type == "cuda")):
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

def create_tf_tg_index_to_name_mappings(metadata):
    tf_idx_to_name = {idx: name for name, idx in metadata["tf_name_to_idx"].items()}
    tg_idx_to_name = {idx: name for name, idx in metadata["tg_id_to_idx"].items()}
    return tf_idx_to_name, tg_idx_to_name

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


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate TF-TG model on multiple datasets and plot AUPRC.")
    parser.add_argument("--output_dir", type=str, default=str(all_evaluation_plot_dir), help="Directory to save evaluation plots.")
    parser.add_argument("--species", type=str, help="Species to evaluate.")
    parser.add_argument("--cell_type", type=str, help="Cell type to evaluate.")
    parser.add_argument("--sample_name", type=str, help="Sample name to evaluate.")
    parser.add_argument("--cross_model_cell_type", type=str, help="Cell type for cross-model evaluation.")
    parser.add_argument("--cross_model_sample_name", type=str, help="Sample name for cross-model evaluation.")
    # Defaults follow bash_scripts/03b_train_tf_to_tg_model.sh, which produced the
    # checkpoints in TF_TG_MODEL_CHECKPOINTS (max_peaks_per_tg=100, max_cells_per_pair=24).
    # run_stability.sh uses 25/25, but that trains a *different* set of checkpoints.
    # Bag geometry is not stored in the .ckpt files (the architecture is bag-size
    # agnostic) and the training logs for the older jobs are gone, so this cannot be
    # confirmed from the artefacts. Scoring with a much smaller bag than training changes
    # what the peak attention and the log-sum-exp pooling see -- override deliberately.
    parser.add_argument("--max_peaks_per_tg", type=int, default=100, help="Max peaks per TG in each edge bag.")
    parser.add_argument("--max_cells_per_pair", type=int, default=24, help="Max cells sampled per TF-TG pair.")
    parser.add_argument("--tf_peak_chunk_size", type=int, default=128, help="Chunk size for TF-peak pairs when running TF-DNA inference")
    parser.add_argument("--batch_size", type=int, default=512, help="Inference batch size.")

    parser.add_argument("--force_reload", action="store_true", help="Force reload of data and models.")

    return parser.parse_args()

args = parse_arguments()
species = args.species
cell_type = args.cell_type
sample_name = args.sample_name
force_reload = args.force_reload

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

tf_name_to_idx_cache_path = cell_type_cache_dir / "tf_name_to_idx.csv"

# Get the map of the TF names to their indices in the TF-DNA model training data.
# These indices address rows of tf_embeddings.pt / tf_masks.pt, so this mapping is the
# one the dataset must use -- it is not interchangeable with an arbitrary re-indexing.
tf_name_to_idx_df = pd.read_csv(tf_name_to_idx_cache_path)
tf_name_to_idx_df["tf_name"] = tf_name_to_idx_df["tf_name"].str.upper()
tf_name_to_idx = tf_name_to_idx_df.set_index("tf_name")["tf_idx"].to_dict()

# Split the target genes into train/val/test based on chromosome
train_genes, val_genes, test_genes = tf_tg_data_builder.split_genes_by_chromosome(
    gene_ref_file,
    train_chroms=train_chroms,
    val_chroms=val_chroms,
    test_chroms=test_chroms
    )

# Score the full cross product: every TF that has an embedding AND is measured in this
# dataset, against every target gene on the test chromosomes that is measured in this
# dataset. Genes with no peak within range are dropped later by build_tftg_inputs.
rna_genes = set(rna_pseudobulk_norm.index)

universe_tfs = sorted(set(tf_name_to_idx) & rna_genes)
universe_tgs = sorted({str(gene).upper() for gene in test_genes} & rna_genes)

if not universe_tfs or not universe_tgs:
    raise ValueError(
        f"Empty prediction universe for {cell_type}/{sample_name}: "
        f"{len(universe_tfs)} TFs, {len(universe_tgs)} TGs"
    )

logging.info(
    f"Prediction universe: {len(universe_tfs):,} TFs "
    f"(of {len(tf_name_to_idx):,} with embeddings) x {len(universe_tgs):,} test-chromosome TGs "
    f"(of {len(set(test_genes)):,} on chroms {test_chroms[0]}-{test_chroms[-1]}) "
    f"= {len(universe_tfs) * len(universe_tgs):,} edges"
)

# build_tftg_inputs reads row.tf_name / row.tg_id / row.label, so name the columns to match
gt_test_df = (
    pd.MultiIndex
    .from_product([universe_tfs, universe_tgs], names=["tf_name", "tg_id"])
    .to_frame(index=False)
)

# Placeholder label. These are unlabeled predictions -- nothing downstream of the model
# reads it, but build_tftg_inputs requires the column.
gt_test_df["label"] = 1.0

# TG indices are only carried through as labels for naming the output rows (target
# expression is looked up by name), so build a fresh mapping spanning the whole
# test-chromosome universe. The training cache's tg_id_to_idx covers only ground-truth
# targets and would silently drop most of the genes we want to score here.
tg_id_to_idx = {tg_name: idx for idx, tg_name in enumerate(universe_tgs)}

tf_idx_to_name = {idx: name for name, idx in tf_name_to_idx.items()}
tg_idx_to_name = {idx: name for name, idx in tg_id_to_idx.items()}

sample_full_grn_dir = RESULT_DIR / "full_test_grns"
sample_full_grn_dir.mkdir(parents=True, exist_ok=True)

cross_tf_tg_df_file = sample_full_grn_dir / f"{cross_model_sample_name}_model_vs_{sample_name}_test_set_grn.tsv"
sample_full_grn_file = sample_full_grn_dir / f"{sample_name}_model_vs_{sample_name}_test_set_grn.tsv"

if not sample_full_grn_file.exists() or not cross_tf_tg_df_file.exists() or force_reload == True:

    # === CREATE FULL SET OF TF-TG INPUTS FOR ALL POSSIBLE TF-TG PAIRS IN THE TEST SET ===
    # tf_name_to_idx / tg_id_to_idx and their reverse mappings are built above from the
    # full prediction universe. The training cache metadata is deliberately not used
    # here: its tg_id_to_idx spans only ground-truth targets.

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
    tg_to_peak_info, cell_to_idx, atac_mat, rna_mat, gene_to_rna_idx = utils.prepare_tftg_lookup_tables(
        peak_to_gene=peak_to_gene,
        atac_peak_map=atac_peak_map,
        atac_pseudobulk=atac_pseudobulk,
        rna_pseudobulk_norm=rna_pseudobulk_norm,
        dataset_peaks=dataset_peaks,
        common_cells=common_cells,
        max_precompute_peaks=args.max_peaks_per_tg,
    )

    n_tgs_with_peaks = sum(
        len(tg_to_peak_info.get(tg_name, {}).get("peak_indices", [])) > 0
        for tg_name in gt_test_df["tg_id"].unique()
    )
    logging.info(f"Target genes with at least one peak in range: {n_tgs_with_peaks:,} / {len(universe_tgs):,}")

    # Get the max number of peaks within 100Kb of any TG in the test set
    max_peaks_real = max(
        len(tg_to_peak_info.get(tg_name, {}).get("peak_indices", []))
        for tg_name in gt_test_df["tg_id"].unique()
    )

    # Build the compact TF-TG input dataset for the test set
    common_build_kwargs = dict(
        max_cells_per_pair=args.max_cells_per_pair,
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
        gt_test_df,
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
        batch_size=args.batch_size,
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
    
    # Resource usage is recorded per model. Note the two are not a clean A/B: whichever
    # model runs first pays torch.compile tracing and DataLoader worker startup, and the
    # second reuses the same persistent workers. Compare the `inference` phases, and read
    # the first model's `model_load` as including one-off warmup.
    resource_records = []
    n_edges_built = len(tftg_inputs_test["label"])

    def _record(probe, model_type, checkpoint_path):
        row = dict(probe.stats)
        row.update({
            "cell_type": cell_type,
            "sample_name": sample_name,
            "model_type": model_type,
            "model_cell_type": cell_type if model_type == "own" else cross_model_cell_type,
            "model_sample_name": sample_name if model_type == "own" else cross_model_sample_name,
            "checkpoint": Path(checkpoint_path).name if checkpoint_path else None,
            "n_edges": n_edges_built,
            "n_tfs": len(universe_tfs),
            "n_tgs": len(universe_tgs),
            "batch_size": args.batch_size,
            "max_peaks_per_tg": args.max_peaks_per_tg,
            "max_cells_per_pair": args.max_cells_per_pair,
            "num_workers": num_workers,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        })
        if row["phase"].startswith("inference") and row["wall_seconds"] > 0:
            row["edges_per_second"] = round(n_edges_built / row["wall_seconds"], 1)
        resource_records.append(row)

    if not sample_full_grn_file.exists() or force_reload:
        # Load the TF→TG model
        with ResourceProbe("model_load (own)", device) as probe:
            tf_tg_model = utils.load_tf_tg_regulation_model(
                tf_dna_model_chkpt,
                tf_tg_model_chkpt,
                tf_embeddings_tensor,
                tf_mask_tensor,
                tf_peak_chunk_size=args.tf_peak_chunk_size,
                compile_model=True,
                device=device
                )

        # Run the model on the test set and generate the predictions DataFrame
        with ResourceProbe("inference (own)", device) as probe:
            prediction_df = generate_model_predictions(tf_tg_model.model, loader, device, tf_idx_to_name, tg_idx_to_name)
        _record(probe, "own", tf_tg_model_chkpt)

        prediction_df.to_csv(sample_full_grn_file, sep="\t", index=False)
    else:
        logging.info(f"{sample_full_grn_file} exists; skipping own-model run (no resource usage recorded)")
        prediction_df = pd.read_csv(sample_full_grn_file, sep="\t", header=0)

    if not cross_tf_tg_df_file.exists() or force_reload:
        # Load the TF→TG model trained on the cross-model cell type and sample
        with ResourceProbe("model_load (cross)", device) as probe:
            cross_tf_tg_model = utils.load_tf_tg_regulation_model(
                tf_dna_model_chkpt,
                cross_model_chkpt,
                tf_embeddings_tensor,
                tf_mask_tensor,
                tf_peak_chunk_size=args.tf_peak_chunk_size,
                compile_model=True,
                device=device
            )

        # Run the cross-trained model on the test set and generate the predictions DataFrame
        with ResourceProbe("inference (cross)", device) as probe:
            cross_model_prediction_df = generate_model_predictions(cross_tf_tg_model.model, loader, device, tf_idx_to_name, tg_idx_to_name)
        _record(probe, "cross", cross_model_chkpt)

        cross_model_prediction_df.to_csv(cross_tf_tg_df_file, sep="\t", index=False)
    else:
        logging.info(f"{cross_tf_tg_df_file} exists; skipping cross-model run (no resource usage recorded)")
        cross_model_prediction_df = pd.read_csv(cross_tf_tg_df_file, sep="\t", header=0)

    if resource_records:
        resource_usage_file = sample_full_grn_dir / f"resource_usage_{cell_type}_{sample_name}.tsv"
        pd.DataFrame(resource_records).to_csv(resource_usage_file, sep="\t", index=False)
        logging.info(f"Wrote resource usage for {len(resource_records)} phases to {resource_usage_file}")

else:
    prediction_df = pd.read_csv(sample_full_grn_file, sep="\t", header=0)
    cross_model_prediction_df = pd.read_csv(cross_tf_tg_df_file, sep="\t", header=0)