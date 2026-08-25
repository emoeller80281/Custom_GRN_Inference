import sys
import argparse
from pathlib import Path

PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER")
sys.path.append(str(PROJECT_DIR))

DATA_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data")
CHKPT_DIR = PROJECT_DIR / "checkpoints"

# species = "hg38"
# cell_type="iPSC"
# sample_name="WT_D13_rep1"

species = "hg38"
cell_type="Macrophage"
sample_name="buffer_2"

# species = "mm10"
# cell_type="mESC"
# sample_name="E7.5_rep1"

# Argelaguet et al. 2022 mouse organogenesis atlas, the paper's own SEACells metacells
# (1,896 metacells with matched RNA + ATAC, 9 wild-type timecourse libraries E7.5-E8.75,
# both CRISPR libraries excluded). Built by
# mouse_preprocessing_scripts/12_tether_training_data/.
# species = "mm10"
# cell_type="mESC"
# sample_name="WT_timecourse_metacells"

# species = "mm10"
# cell_type="mouse_liver"
# sample_name="liver_4"

# species = "mm10"
# cell_type="mouse_hepatocytes"
# sample_name="hepatocytes_3"

# species = "hg38"
# cell_type="K562"
# sample_name="sample_1"

assert cell_type in {"Macrophage", "mESC", "K562", "iPSC", "mouse_liver", "mouse_hepatocytes"}, \
    f"Invalid cell type: {cell_type}. Select from: 'Macrophage', 'mESC', 'K562', 'iPSC', 'mouse_liver', 'mouse_hepatocytes'"
assert species in {"mm10", "hg38"}, \
    f"Invalid species: {species}. Select from: 'mm10', 'hg38'"

# Species is a property of the cell type, not an independent choice. The CLI entry points
# take --cell_type without --species, so the cache layout below has to be derivable from
# the cell type alone.
cell_type_to_species = {
    "mESC": "mm10",
    "mouse_liver": "mm10",
    "mouse_hepatocytes": "mm10",
    "iPSC": "hg38",
    "Macrophage": "hg38",
    "K562": "hg38",
}

assert cell_type_to_species[cell_type] == species, \
    f"cell_type {cell_type!r} belongs to {cell_type_to_species[cell_type]}, but species is set to {species!r}."


def species_for_cell_type(cell_type_name: str) -> str:
    try:
        return cell_type_to_species[cell_type_name]
    except KeyError:
        raise ValueError(
            f"Unknown cell type {cell_type_name!r}. Add it to config.cell_type_to_species. "
            f"Known: {sorted(cell_type_to_species)}"
        ) from None


def species_cache_dir(species_name: str) -> Path:
    """cached_data/<species>/ -- the root everything cached now hangs off."""
    return PROJECT_DIR / "cached_data" / species_name


TF_DNA_CACHE_DIRNAME = "tf_dna_cache"


def tf_dna_cache_dir(species_name: str) -> Path:
    """TF-DNA training cache, one per species.

    Its contents -- the ChIP-Atlas edge set, the peak universe, the one-hot peak tensor --
    depend only on the genome, never on the cell type. Keeping it per cell type meant
    byte-identical 30 GB copies under iPSC_cache and K562_cache.
    """
    return species_cache_dir(species_name) / TF_DNA_CACHE_DIRNAME


def tf_dna_cache_dir_for_cell_type(cell_type_name: str) -> Path:
    """The species TF-DNA cache a given cell type draws its TF tables from."""
    return tf_dna_cache_dir(species_for_cell_type(cell_type_name))


def cell_type_cache_dir(cell_type_name: str, species_name: str | None = None) -> Path:
    """cached_data/<species>/<cell_type>_cache/ -- TF-TG caches and the TF embedding table."""
    species_name = species_name or species_for_cell_type(cell_type_name)
    return species_cache_dir(species_name) / f"{cell_type_name}_cache"


# TF-DNA model checkpoints for the different cell types
mm10_tf_dna_path = CHKPT_DIR / "tf_dna_mm10_3831017" / "epoch=05-val_auroc=0.9460-val_loss=0.1880.ckpt"
hg38_tf_dna_path = CHKPT_DIR / "tf_dna_hg38_3831693" / "epoch=02-val_auroc=0.9642-val_loss=0.1702.ckpt"

tf_dna_model_checkpoints = {
    "mESC": mm10_tf_dna_path,
    "mouse_liver": mm10_tf_dna_path,
    "mouse_hepatocytes": mm10_tf_dna_path,
    "iPSC": hg38_tf_dna_path,
    "Macrophage": hg38_tf_dna_path,
    "K562": hg38_tf_dna_path
}

# Species-specific paths
genome_fasta_path = DATA_DIR / "genome_data" / "reference_genome" / species / f"{species}.fa"
chrom_sizes_path = DATA_DIR / "genome_data" / "reference_genome" / species / f"{species}.chrom.sizes"
embedding_dir = DATA_DIR / "tf_data" / species / "tf_embeddings"
chip_atlas_cache_dir = DATA_DIR / "ground_truth_files" / f"chip_atlas_{species}_all.csv"

if species == "mm10":
    gene_ref_file = DATA_DIR / "genome_data" / "genome_annotation" / "mm10" / "Mus_musculus.GRCm39.115.gtf.gz"
elif species == "hg38":
    gene_ref_file = DATA_DIR / "genome_data" / "genome_annotation" / "hg38" / "Homo_sapiens.GRCh38.113.gtf.gz"

# Cell type and sample-specific paths
sample_input_data_dir = DATA_DIR / "sample_input_data" / cell_type / sample_name

training_cache_dir = PROJECT_DIR / "cached_data" / species / f"{cell_type}_tf_tg_cache"
tf_dna_input_cache_dir = PROJECT_DIR / "cached_data" / species / "tf_dna_cache"
tf_tg_input_cache_dir = training_cache_dir / sample_name

# Shared by TF-to-DNA and TF-to-TG training. These are species-level: the TF set comes from
# the species' ChIP-Atlas edges, so every cell type of a species had a byte-identical copy.
# They live with the TF-DNA cache, and tf_idx everywhere indexes rows of this one table.
tf_name_to_idx_cache_path = tf_dna_input_cache_dir / "tf_name_to_idx.csv"
tf_embedding_cache_path = tf_dna_input_cache_dir / "tf_embeddings.pt"
tf_mask_cache_path = tf_dna_input_cache_dir / "tf_masks.pt"

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
# Full peak x cell / gene x cell pseudobulk matrices -- not needed for the default (frozen
# per-edge cell bag) training path, only for --resample_cells_per_epoch in
# train_tf_to_tg_model.py. Built by build_tf_to_tg_train_data.py, optionally on their own via
# --build_resample_matrices_only for a cache that already has everything else.
tf_tg_atac_mat_cache_path = tf_tg_input_cache_dir / "atac_mat.pt"
tf_tg_rna_mat_cache_path = tf_tg_input_cache_dir / "rna_mat.pt"
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
    ],
    "mouse_liver": [
        DATA_DIR / "ground_truth_files" / "chipatlas_mouse_liver.csv",
        DATA_DIR / "ground_truth_files" / "KnockTF_mouse_liver.csv",
    ],
    "mouse_hepatocytes": [
        DATA_DIR / "ground_truth_files" / "chipatlas_mouse_hepatocytes.csv",
    ]
}

