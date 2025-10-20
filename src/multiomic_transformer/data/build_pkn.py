import sys
from pathlib import Path
import logging
import os
import pandas as pd
import numpy as np
import networkx as nx

from mygene import MyGeneInfo
import pickle
from multiomic_transformer.data.networks import (
    trrust_pathway, string_pathway, kegg_pathways
)
from config.settings import *

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR)) 

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.info("Building Prior Knowledge Network")
    ORGANISM_CODE = "hg38"
    
    PKN_DIR = DATA_DIR / "prior_knowledge_network_data" / ORGANISM_CODE

    STRING_DIR = PKN_DIR / "STRING"
    TRRUST_DIR = PKN_DIR / "TRRUST"
    KEGG_DIR = PKN_DIR / "KEGG"

    STRING_DIR.mkdir(parents=True, exist_ok=True)
    TRRUST_DIR.mkdir(parents=True, exist_ok=True)
    KEGG_DIR.mkdir(parents=True, exist_ok=True)

    string_csv_file = STRING_DIR / "string_human_pkn.csv"
    string_gpickle_file = STRING_DIR / "string_human_pkn.gpickle"
    
    trrust_csv_file = TRRUST_DIR / "trrust_human_pkn.csv"
    trrust_gpickle_file = TRRUST_DIR / "trrust_human_pkn.gpickle"
    
    kegg_csv_file = KEGG_DIR / "kegg_human_pkn.csv"
    kegg_gpickle_file = KEGG_DIR / "kegg_human_pkn.gpickle"

    # ----- Build TRRUST Graph -----
    if not os.path.isfile(trrust_csv_file):
        logging.info("Building TRRUST prior knowledge network")
        trrust_pathway.build_trrust_pkn(
            species="human",
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
            dataset_name=ORGANISM_CODE,
            output_path=str(KEGG_DIR),
            organism="hsa",
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
            string_dir=str(STRING_DIR),
            string_org_code="9606",
            min_combined_score=800,
            as_directed=True,
            out_csv=str(string_csv_file),
            out_gpickle=str(string_gpickle_file)
        )
        string_pkn = pd.read_csv(string_csv_file)
    else:
        logging.info("STRING CSV and Graphml files found, loading pkn csv")
        string_pkn = pd.read_csv(string_csv_file)


    # --- Harmonize key columns ---
    # Ensure consistent TF/TG-style naming
    trrust_pkn.rename(columns={"source": "source_id", "target": "target_id"}, inplace=True)
    kegg_pkn.rename(columns={"source": "source_id", "target": "target_id"}, inplace=True)
    string_pkn.rename(columns={"protein1": "source_id", "protein2": "target_id"}, inplace=True)

    trrust_pkn["source_db"] = "TRRUST"
    kegg_pkn["source_db"] = "KEGG"
    string_pkn["source_db"] = "STRING"
    mg = MyGeneInfo()

    # Convert Ensembl IDs or aliases in your PKN to HGNC symbols
    def normalize_genes(gene_list):
        query = mg.querymany(gene_list, scopes=["symbol", "alias", "ensembl.gene"], fields="symbol", species="human")
        mapping = {q["query"]: q.get("symbol", q["query"]) for q in query}
        return [mapping.get(g, g).upper() for g in gene_list]
    
    # Case-normalize the gene names
    for df in [trrust_pkn, kegg_pkn, string_pkn]:
        df["source_id"] = df["source_id"].str.upper()
        df["target_id"] = df["target_id"].str.upper()
        df["TF"] = normalize_genes(df["TF"])
        df["TG"] = normalize_genes(df["TG"])

    def print_network_info(df: pd.DataFrame, network_name: str):
        logging.info(f"\n{network_name}")
        logging.info(trrust_pkn.head())
        logging.info(f"TFs: {df['source_id'].nunique()}")
        logging.info(f"TGs: {df['target_id'].nunique()}")
        logging.info(f"Edges: {df.shape[0]}")
        
    print_network_info(trrust_pkn, "TRRUST")
    print_network_info(kegg_pkn, "KEGG")
    print_network_info(string_pkn, "STRING")




    # # --- Merge all sources ---
    # logging.info("\nMerging all PKNs")
    # merged_df = pd.concat([trrust_pkn, kegg_pkn, string_pkn], ignore_index=True)
    # logging.info("Done!")
    
    # print_network_info(merged_df, "Merged DataFrame")

    # # Drop perfect duplicates (same source-target pair + identical source_db)
    # merged_df.drop_duplicates(subset=["source_id", "target_id", "source_db"], inplace=True)

    # logging.info(f"\nUnified PKN: {len(merged_df):,} edges across {merged_df['source_db'].nunique()} sources")

    # # --- Save outputs ---
    # merged_df_path_prefix = PKN_DIR / f"{ORGANISM_CODE}_merged_pkn"
    # os.makedirs(os.path.dirname(merged_df_path_prefix), exist_ok=True)
    
    # merged_df.to_csv(f"{merged_df_path_prefix}.csv", index=False)
    # G_merged = nx.from_pandas_edgelist(merged_df, source="source_id", target="target_id", edge_attr=True, create_using=nx.DiGraph())
    # with open(f"{merged_df_path_prefix}.gpickle", 'wb') as f:
    #     pickle.dump(G_merged, f, pickle.HIGHEST_PROTOCOL)
