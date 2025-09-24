import os
import torch
import pandas as pd
import logging
import pybedtools

from transformer_2 import MultiomicTransformer

logging.basicConfig(level=logging.INFO, format="%(message)s")

PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
RAW_MESC_DATA_DIR = "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/SINGLE_CELL_DATASETS/DS014_DOI496239_MOUSE_ESC_RAW_FILES"
MESC_PEAK_MATRIX_FILE = "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/SINGLE_CELL_DATASETS/DS014_DOI496239_MOUSE_ESCDAYS7AND8/scATAC_PeakMatrix.txt"

MM10_GENOME_DIR = os.path.join(PROJECT_DIR, "data/reference_genome/mm10")
MM10_CHROM_SIZES_FILE = os.path.join(MM10_GENOME_DIR, "chrom.sizes")
MM10_GENE_TSS_FILE = os.path.join(PROJECT_DIR, "data/genome_annotation/mm10/mm10_TSS.bed")
GROUND_TRUTH_DIR = os.path.join(PROJECT_DIR, "ground_truth_files")
SAMPLE_INPUT_DIR = os.path.join(PROJECT_DIR, "input/mESC/")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output/transformer_testing_output")

def load_homer_tf_to_peak_results():
    assert os.path.exists(os.path.join(OUTPUT_DIR, "homer_tf_to_peak.parquet")), \
        "ERROR: Homer TF to peak output parquet file required"
        
    homer_results = pd.read_parquet(os.path.join(OUTPUT_DIR, "homer_tf_to_peak.parquet"), engine="pyarrow")
    homer_results = homer_results.reset_index(drop=True)
    homer_results["source_id"] = homer_results["source_id"].str.capitalize()
    
    return homer_results

def create_or_load_genomic_windows(chrom_id, window_size, force_recalculate=False):
    genome_window_file = os.path.join(MM10_GENOME_DIR, f"mm10_{chrom_id}_windows_{window_size // 1000}kb.bed")
    if not os.path.exists(genome_window_file) or force_recalculate:
        
        logging.info("Creating genomic windows")
        mm10_genome_windows = pybedtools.bedtool.BedTool().window_maker(g=MM10_CHROM_SIZES_FILE, w=window_size)
        mm10_windows = (
            mm10_genome_windows
            .filter(lambda x: x.chrom == chrom_id)  # TEMPORARY Restrict to one chromosome for testing
            .saveas(genome_window_file)
            .to_dataframe()
        )
    else:
        
        logging.info("Loading existing genomic windows")
        mm10_windows = pybedtools.BedTool(genome_window_file).to_dataframe()
        
    return mm10_windows

def make_peak_to_window_map(peaks_bed: pd.DataFrame, windows_bed: pd.DataFrame) -> dict[str, int]:
    """
    peaks_bed: df with ['chrom','start','end','peak_id']
    windows_bed: df with ['chrom','start','end','win_idx']
    """
    bedtool_peaks = pybedtools.BedTool.from_dataframe(peaks_bed)
    bedtool_windows = pybedtools.BedTool.from_dataframe(windows_bed)
    
    mapping = {}
    for interval in bedtool_peaks.intersect(bedtool_windows, wa=True, wb=True):
        peak_id = interval.name  # the peak_id column from peaks_bed
        win_idx = int(interval.fields[-1])  # last column = win_idx
        mapping[peak_id] = win_idx
    return mapping

def prepare_inputs(TG_pseudobulk: pd.DataFrame,
                   RE_pseudobulk: pd.DataFrame,
                   tf_list: list[str],
                   window_map: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert pseudobulk matrices into model inputs.
    
    Args:
      TG_pseudobulk : genes x samples dataframe
      RE_pseudobulk : peaks x samples dataframe
      tf_list       : list of TF gene symbols to keep
      window_map    : dict mapping peaks -> window index (0..num_windows-1)
    
    Returns:
      tf_expr   : Tensor [B, num_tf]
      atac_wins : Tensor [B, num_windows, 1]
    """
    # 1. Extract TF expression
    tf_expr = TG_pseudobulk.loc[TG_pseudobulk.index.intersection(tf_list)].T
    tf_tensor = torch.tensor(tf_expr.values, dtype=torch.float32)   # [B, num_tf]

    # 2. Collapse peaks into windows
    num_windows = max(window_map.values()) + 1
    atac_wins = torch.zeros((RE_pseudobulk.shape[1], num_windows, 1), dtype=torch.float32)

    peak_idx = [RE_pseudobulk.index.get_loc(p) for p in window_map if p in RE_pseudobulk.index]
    win_idx = [window_map[p] for p in window_map if p in RE_pseudobulk.index]

    peak_tensor = torch.tensor(RE_pseudobulk.iloc[peak_idx].values.T, dtype=torch.float32)  # [B, num_peaks]
    win_idx_tensor = torch.tensor(win_idx, dtype=torch.long)

    atac_wins.index_add_(1, win_idx_tensor, peak_tensor.unsqueeze(-1))
    return tf_tensor, atac_wins

sample_name = "E7.5_rep1"
window_size = 25000

tf_list = list(load_homer_tf_to_peak_results()["source_id"].unique())
logging.info(f"TF List: {tf_list[:5]}, total {len(tf_list)} TFs")

sample_data_dir = os.path.join(SAMPLE_INPUT_DIR, sample_name)
TG_pseudobulk = pd.read_csv(os.path.join(sample_data_dir, "TG_pseudobulk.tsv"), sep="\t", index_col=0)
RE_pseudobulk = pd.read_csv(os.path.join(sample_data_dir, "RE_pseudobulk.tsv"), sep="\t", index_col=0)

peaks_df = (
    RE_pseudobulk.index.to_series()
    .str.split("[:-]", expand=True)
    .rename(columns={0: "chrom", 1: "start", 2: "end"})
)
peaks_df["start"] = peaks_df["start"].astype(int)
peaks_df["end"] = peaks_df["end"].astype(int)
peaks_df["peak_id"] = RE_pseudobulk.index

# Create genome windows and add index
mm10_windows = create_or_load_genomic_windows("chr19", window_size)
mm10_windows = mm10_windows.reset_index(drop=True)
mm10_windows["win_idx"] = mm10_windows.index

# Build peak -> window mapping
window_map = make_peak_to_window_map(peaks_df, mm10_windows)
logging.info(f"Mapped {len(window_map)} peaks to windows")

logging.info("TG Pseudobulk")
logging.info(f"\tTG_pseudobulk: {TG_pseudobulk.shape[0]:,} Genes x {TG_pseudobulk.shape[1]} metacells")

logging.info("\nRE Pseudobulk")
logging.info(f"\tRE_pseudobulk: {RE_pseudobulk.shape[0]:,} Peaks x {RE_pseudobulk.shape[1]} metacells")

# Example setup
d_model = 128
num_heads = 8
d_ff = 256
dropout = 0.1
num_tf = len(tf_list)
num_windows = max(window_map.values()) + 1
num_tg = TG_pseudobulk.shape[0]   # or restrict to TGs only

model = MultiomicTransformer(d_model, num_heads, d_ff, dropout, num_tf, num_windows, num_tg)

# Prepare inputs
logging.info("\nPreparing Info")
tf_tensor, atac_wins = prepare_inputs(TG_pseudobulk, RE_pseudobulk, tf_list, window_map)

# Forward pass
logging.info("\n ----- Training -----")
gene_logits = model(atac_wins, tf_tensor)   # [B, num_tg]


logging.info(gene_logits)
logging.info(gene_logits.shape)
