import sys
from pathlib import Path
import logging
import os
import pandas as pd
import numpy as np
import networkx as nx
from typing import Union

from mygene import MyGeneInfo
import pickle
from multiomic_transformer.data.networks import (
    trrust_pathway, string_pathway, kegg_pathways
)
from config.settings import *

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR)) 

def build_organism_pkns(
    organism_code: str,
    string_csv_file: Union[str, Path],
    trrust_csv_file: Union[str, Path],
    kegg_csv_file: Union[str, Path],
    ) -> pd.DataFrame:
    
    string_csv_file = Path(string_csv_file)
    trrust_csv_file = Path(trrust_csv_file)
    kegg_csv_file = Path(kegg_csv_file)

    for p in (string_csv_file.parent, trrust_csv_file.parent, kegg_csv_file.parent):
        p.mkdir(parents=True, exist_ok=True)

    string_gpickle_file = string_csv_file.with_suffix(".gpickle")
    trrust_gpickle_file = trrust_csv_file.with_suffix(".gpickle")
    kegg_gpickle_file = kegg_csv_file.with_suffix(".gpickle")

    # Check organism code
    if organism_code == "mm10":
        trrust_species = "mouse"
        kegg_organism = "mmu"
        string_org_code = "10090"
        
    elif organism_code == "hg38":
        trrust_species = "human"
        kegg_organism = "hsa"
        string_org_code = "9606"
    else:
        raise ValueError(f"Organism not recognized: {organism_code} (must be 'mm10' or 'hg38').")

    # ----- Build TRRUST Graph -----
    if not os.path.isfile(trrust_csv_file):
        logging.info("Building TRRUST prior knowledge network")
        trrust_pathway.build_trrust_pkn(
            species=trrust_species,
            out_csv=str(trrust_csv_file),
            out_gpickle = str(trrust_gpickle_file)
        )
        trrust_pkn = pd.read_csv(trrust_csv_file)
    else:
        logging.info("TRRUST CSV and Graphml files found, loading pkn csv")
        trrust_pkn = pd.read_csv(trrust_csv_file)

    # ----- Build KEGG Graph -----
    if not os.path.isfile(kegg_csv_file):
        logging.info("Building KEGG prior knowledge network")
        kegg_pathways.build_kegg_pkn(
            dataset_name=organism_code,
            output_path=str(KEGG_DIR),
            organism=kegg_organism,
            out_csv=str(kegg_csv_file),
            out_gpickle=str(kegg_gpickle_file)
        )
        kegg_pkn = pd.read_csv(kegg_csv_file)
    else:
        logging.info("KEGG CSV and Graphml files found, loading pkn csv")
        kegg_pkn = pd.read_csv(kegg_csv_file)
        
    # ----- Build STRING Graph -----
    if not os.path.isfile(string_csv_file):
        logging.info("Building STRING prior knowledge network")
        string_pathway.build_string_pkn(
            string_dir=str(string_csv_file.parent),
            string_org_code=string_org_code,
            min_combined_score=800,
            as_directed=True,
            out_csv=str(string_csv_file),
            out_gpickle=str(string_gpickle_file)
        )
        string_pkn = pd.read_csv(string_csv_file)
    else:
        logging.info("STRING CSV and Graphml files found, loading pkn csv")
        string_pkn = pd.read_csv(string_csv_file)

    # Add source database column
    trrust_pkn["source_db"] = "TRRUST"
    kegg_pkn["source_db"] = "KEGG"
    string_pkn["source_db"] = "STRING"
    
    # Convert Ensembl IDs or aliases in your PKN to HGNC symbols
    mg = MyGeneInfo()
    def normalize_genes(gene_list):
        query = mg.querymany(gene_list, scopes=["symbol", "alias", "ensembl.gene"], fields="symbol", species=trrust_species)
        mapping = {q["query"]: q.get("symbol", q["query"]) for q in query}
        return [mapping.get(g, g).upper() for g in gene_list]
    
    def print_network_info(df: pd.DataFrame, network_name: str):
        logging.info(f"\n{network_name}")
        logging.info(trrust_pkn.head())
        logging.info(f"TFs: {df['TF'].nunique()}")
        logging.info(f"TGs: {df['TG'].nunique()}")
        logging.info(f"Edges: {df.shape[0]}")
    
    # Case-normalize the gene names
    for df in [trrust_pkn, kegg_pkn, string_pkn]:
        network_name = str(df["source_db"].to_list()[0])
        df["TF"] = df["TF"].str.upper()
        df["TG"] = df["TG"].str.upper()
        df["TF"] = normalize_genes(df["TF"])
        df["TG"] = normalize_genes(df["TG"])
        print_network_info(df, network_name)
        
        # Save the normalized dataframe
        if network_name == "TRRUST":
            df.to_csv(trrust_csv_file, index=False)
        elif network_name == "KEGG":
            df.to_csv(kegg_csv_file, index=False)
        elif network_name == "STRING":
            df.to_csv(string_csv_file, index=False)
            
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("Building Prior Knowledge Network")
    
    string_csv_file = STRING_DIR / f"string_{ORGANISM_CODE}_pkn.csv"
    trrust_csv_file = TRRUST_DIR / f"trrust_{ORGANISM_CODE}_pkn.csv"
    kegg_csv_file = KEGG_DIR / f"kegg_{ORGANISM_CODE}_pkn.csv"
    
    build_organism_pkns(ORGANISM_CODE, string_csv_file, trrust_csv_file, kegg_csv_file)
