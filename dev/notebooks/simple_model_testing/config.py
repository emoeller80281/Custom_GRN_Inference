import sys
import argparse
from pathlib import Path

PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/notebooks/simple_model_testing")
sys.path.append(str(PROJECT_DIR))

DATA_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data")

# species = "hg38"
# cell_type="iPSC"
# sample_name="WT_D13_rep1"

species = "mm10"
cell_type="mESC"
sample_name="E7.5_rep1"

assert cell_type in {"Macrophage", "mESC", "K562", "iPSC"}, \
    f"Invalid cell type: {cell_type}. Select from: 'Macrophage', 'mESC', 'K562', 'iPSC'"
assert species in {"mm10", "hg38"}, \
    f"Invalid species: {species}. Select from: 'mm10', 'hg38'"

# Species-specific paths
genome_fasta_path = DATA_DIR / "genome_data" / "reference_genome" / species / f"{species}.fa"
chrom_sizes_path = DATA_DIR / "genome_data" / "reference_genome" / species / f"{species}.chrom.sizes"
embedding_dir = PROJECT_DIR / "data" / "tf_data" / species / "tf_embeddings"
chip_atlas_cache_dir = DATA_DIR / "ground_truth_files" / f"chip_atlas_{species}_all.csv"

if species == "mm10":
    gene_ref_file = DATA_DIR / "genome_data" / "genome_annotation" / "mm10" / "Mus_musculus.GRCm39.115.gtf.gz"
elif species == "hg38":
    gene_ref_file = DATA_DIR / "genome_data" / "genome_annotation" / "hg38" / "Homo_sapiens.GRCh38.113.gtf.gz"

# Cell type and sample-specific paths
sample_input_data_dir = PROJECT_DIR / "data" / "sample_input_data" / cell_type / sample_name

training_cache_dir = PROJECT_DIR / "data" / f"{cell_type}_cache"
tf_dna_input_cache_dir = training_cache_dir / "tf_dna_training_cache"
tf_tg_input_cache_dir = training_cache_dir / "tf_tg_training_cache" / sample_name

# Shared cache files for both TF-to-TG and TF-to-DNA training
tf_name_to_idx_cache_path = training_cache_dir / "tf_name_to_idx.csv"
tf_embedding_cache_path = training_cache_dir / "tf_embeddings.pt"
tf_mask_cache_path = training_cache_dir / "tf_masks.pt"

# Cache file for the merged ground truth dataset for the TF-TG true edges
merged_ground_truth_cache_path = training_cache_dir / f"{cell_type}_merged_ground_truth.parquet"

# TF-DNA training specific cache files
tf_dna_peak_id_to_idx_cache_path = tf_dna_input_cache_dir / "peak_id_to_idx.csv"
tf_dna_edge_tf_idx_cache_path = tf_dna_input_cache_dir / "edge_tf_idx.pt"
tf_dna_edge_peak_idx_cache_path = tf_dna_input_cache_dir / "edge_peak_idx.pt"
tf_dna_edge_labels_cache_path = tf_dna_input_cache_dir / "edge_labels.pt"
tf_dna_tf_lengths_cache_path = tf_dna_input_cache_dir / "tf_lengths.pt"
tf_dna_peak_onehot_cache_path = tf_dna_input_cache_dir / "peak_onehot_array.pt"
tf_dna_train_idx_cache_path = tf_dna_input_cache_dir / "train_idx.pt"
tf_dna_val_idx_cache_path = tf_dna_input_cache_dir / "val_idx.pt"
tf_dna_test_idx_cache_path = tf_dna_input_cache_dir / "test_idx.pt"

# TF-TG training specific cache files
tf_tg_atac_peak_cache_path = tf_tg_input_cache_dir / "atac_peak_tensor.pt"
tf_tg_metadata_cache_path = tf_tg_input_cache_dir / "metadata.json"
tf_tg_manifest_cache_path = tf_tg_input_cache_dir / "manifest.json"
tf_tg_train_cache_path = tf_tg_input_cache_dir / "tftg_inputs_train.pt"
tf_tg_val_cache_path = tf_tg_input_cache_dir / "tftg_inputs_val.pt"
tf_tg_test_cache_path = tf_tg_input_cache_dir / "tftg_inputs_test.pt"

# Ground truth files by cell type
gt_by_dataset_dict = {
    "Macrophage": [
        DATA_DIR / "ground_truth_files" / "chipatlas_macrophage.csv",
    ],
    "mESC": [
        DATA_DIR / "ground_truth_files" / "chip_atlas_tf_peak_tg_dist.csv",
        DATA_DIR / "ground_truth_files" / "RN111.tsv",
        DATA_DIR / "ground_truth_files" / "RN112.tsv",
        DATA_DIR / "ground_truth_files" / "RN114.tsv",
        DATA_DIR / "ground_truth_files" / "RN116.tsv",        
    ],
    "K562": [
        DATA_DIR / "ground_truth_files" / "chipatlas_K562.csv",
        DATA_DIR / "ground_truth_files" / "RN117.tsv",        
    ],
    "iPSC": [
        DATA_DIR / "ground_truth_files" / "chipatlas_iPSC_1mb.csv",
    ]
}