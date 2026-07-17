import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/notebooks/simple_model_testing")
sys.path.append(str(PROJECT_DIR))

import utils

def parse_args():
    parser = argparse.ArgumentParser(description="Download ChIP-Atlas TF-DNA interaction data for a specified species and list of TFs.")
    parser.add_argument("--species", type=str, required=True, help="Species genome assembly (e.g., 'mm10' for mouse, 'hg38' for human).")
    parser.add_argument("--entrez_email", type=str, required=True, help="Email address for Entrez queries.")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of parallel workers to use for downloading data.")
    
    return parser.parse_args()

def create_organism_chip_atlas_file(
    species: str, 
    ground_truth_dir: Path, 
    tf_chip_seq_save_dir: Path,
    tf_names: np.ndarray,
    num_workers: int = 10
    ) -> pd.DataFrame:
    
    full_chip_atlas_path = ground_truth_dir / f"chip_atlas_{species}_all.parquet"
    
    if not Path(full_chip_atlas_path).exists():
        utils.fetch_chip_atlas_tf_list_to_parquet(
            tf_names, 
            genome=species, 
            out_dir=tf_chip_seq_save_dir,
            num_workers=num_workers
            )
        
        utils.build_chip_atlas_df_from_parquet(
            parquet_dir=tf_chip_seq_save_dir, 
            output_file=full_chip_atlas_path
        )
        
        chip_atlas_full_df: pd.DataFrame = pd.read_parquet(full_chip_atlas_path)
        
        logging.info(f"Fetched {len(chip_atlas_full_df)} TF-DNA interactions from ChIP-Atlas for {species}. Saving...")
        chip_atlas_full_df.to_parquet(full_chip_atlas_path, index=False)
    else:
        chip_atlas_full_df: pd.DataFrame = pd.read_parquet(full_chip_atlas_path)
        logging.info(f"Loaded {len(chip_atlas_full_df)} TF-DNA interactions from existing file for {species}.")

    return chip_atlas_full_df

def main():
    args = parse_args()

    species = args.species
    email = args.entrez_email
    num_workers = args.num_workers

    if species == "mm10":
        organism_name = "mouse"
    elif species == "hg38":
        organism_name = "human"

    DATA_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data")
    
    tf_sequences_dir = PROJECT_DIR / "data" / "tf_data" / species / "tf_sequences"
    tf_chip_seq_save_dir = PROJECT_DIR / "data" / "tf_data" / species / "chip_atlas_TF_files"
    
    tf_sequences_dir.mkdir(parents=True, exist_ok=True)
    tf_chip_seq_save_dir.mkdir(parents=True, exist_ok=True)

    # Load the TF names for the species from the motif information file
    tf_name_file = DATA_DIR / "databases" / "motif_information" / species / "TF_Information_all_motifs.txt"
    tf_name_df = pd.read_csv(tf_name_file, sep="\t")
    tf_names = tf_name_df["TF_Name"].unique()

    # Fetch all ChIP-Atlas TF-DNA interactions for the TFs in the species TF list
    ground_truth_dir = DATA_DIR / "ground_truth_files"
    chip_atlas_full_df = create_organism_chip_atlas_file(
        species=species,
        ground_truth_dir=ground_truth_dir,
        tf_chip_seq_save_dir=tf_chip_seq_save_dir,
        tf_names=tf_names,
        num_workers=num_workers
    )

    # Get the TF names from the ChIP-Atlas dataset
    chip_atlas_tfs = chip_atlas_full_df["source_id"].unique()

    # Download FASTA files for the TFs currently in the ChIP-Atlas database
    gene_names = chip_atlas_tfs

    logging.info(f"Downloading FASTA files for {len(gene_names)} TFs from ChIP-Atlas for {species}...")
    utils.download_gene_protein_fastas(
        gene_names=gene_names,
        organism=organism_name,
        output_dir=tf_sequences_dir,
        email=email,
    )

if __name__ == "__main__":
    main()