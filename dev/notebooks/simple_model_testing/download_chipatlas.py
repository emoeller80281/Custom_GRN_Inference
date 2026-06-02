import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

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
    save_dir: Path, 
    tf_names: np.ndarray,
    num_workers: int = 10
    ) -> pd.DataFrame:
    if not Path(save_dir / f"chip_atlas_{species}_all.csv").exists():
        chip_atlas_full_df: pd.DataFrame = utils.fetch_chip_atlas_tf_list(tf_names, species=species, num_workers=num_workers)
        chip_atlas_full_df.to_csv(save_dir / f"chip_atlas_{species}_all.csv", index=False)
    else:
        chip_atlas_full_df: pd.DataFrame = pd.read_csv(save_dir / f"chip_atlas_{species}_all.csv")
        
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

    # Load the TF names for the species from the motif information file
    tf_name_file = DATA_DIR / "databases" / "motif_information" / species / "TF_Information_all_motifs.txt"
    tf_name_df = pd.read_csv(tf_name_file, sep="\t")
    tf_names = tf_name_df["TF_Name"].unique()

    # Fetch all ChIP-Atlas TF-DNA interactions for the TFs in the species TF list
    ground_truth_dir = DATA_DIR / "ground_truth_files"
    chip_atlas_full_df = create_organism_chip_atlas_file(
        species=species,
        save_dir=ground_truth_dir,
        tf_names=tf_names,
        num_workers=num_workers
    )

    # Get the TF names from the ChIP-Atlas dataset
    chip_atlas_tfs = chip_atlas_full_df["TF"].unique()

    # Download FASTA files for the TFs currently in the ChIP-Atlas database
    gene_names = chip_atlas_tfs

    output_dir = PROJECT_DIR / "data" / "tf_data" / species / "tf_sequences"
    output_dir.mkdir(parents=True, exist_ok=True)

    utils.download_gene_protein_fastas(
        gene_names=gene_names,
        organism=organism_name,
        output_dir=output_dir,
        email=email,
    )

if __name__ == "__main__":
    main()