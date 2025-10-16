import sys
from pathlib import Path
import logging
import os
import pandas as pd
import numpy as np
import networkx as nx
from multiomic_transformer.data.networks import (
    trrust_pathway, string_pathway, kegg_pathways
)
from config.settings import *

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR)) 

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    STRING_DIR = PKN_DIR / "STRING"
    TRRUST_DIR = PKN_DIR / "TRRUST"
    KEGG_DIR = PKN_DIR / "KEGG"

    STRING_DIR.mkdir(parents=True, exist_ok=True)
    TRRUST_DIR.mkdir(parents=True, exist_ok=True)
    KEGG_DIR.mkdir(parents=True, exist_ok=True)

    string_csv_file = STRING_DIR / "string_mouse_pkn.csv"
    string_graphml_file = STRING_DIR / "string_mouse_pkn.graphml"
    
    trrust_csv_file = TRRUST_DIR / "trrust_mouse_pkn.csv"
    trrust_graphml_file = TRRUST_DIR / "trrust_mouse_pkn.graphml"
    
    kegg_csv_file = KEGG_DIR / "kegg_mouse_pkn.csv"
    kegg_graphml_file = KEGG_DIR / "kegg_mouse_pkn.graphml"

    if not os.path.isfile(string_csv_file):
        logging.info("Building STRING prior knowledge network")
        string_pathway.build_string_pkn(
            string_dir=str(STRING_DIR),
            string_org_code="10090",
            as_directed=True,
            out_csv=str(string_csv_file)
            # out_graphml=str(string_graphml_file)
        )
        string_pkn = pd.read_csv(string_csv_file)
    else:
        logging.info("STRING CSV and Graphml files found, loading pkn csv")
        string_pkn = pd.read_csv(string_csv_file)

    if not os.path.isfile(trrust_csv_file):
        logging.info("Building TRRUST prior knowledge network")
        trrust_pathway.build_trrust_pkn(
            species="mouse",
            out_csv=str(trrust_csv_file),
            out_graphml = str(trrust_graphml_file)
        )
        trrust_pkn = pd.read_csv(trrust_csv_file)
    else:
        logging.info("TRRUST CSV and Graphml files found, loading pkn csv")
        trrust_pkn = pd.read_csv(trrust_csv_file)

    if not os.path.isfile(kegg_csv_file):
        logging.info("Building KEGG prior knowledge network")
        kegg_pathways.build_kegg_pkn(
            dataset_name=DATASET_NAME,
            output_path=str(KEGG_DIR),
            organism="mmu",
            out_csv=str(kegg_csv_file),
            out_graphml=str(kegg_graphml_file)
        )
        kegg_pkn = pd.read_csv(kegg_csv_file)
    else:
        logging.info("KEGG CSV and Graphml files found, loading pkn csv")
        kegg_pkn = pd.read_csv(kegg_csv_file)

    # --- Harmonize key columns ---
    # Ensure consistent TF/TG-style naming
    trrust_pkn.rename(columns={"source": "source_id", "target": "target_id"}, inplace=True)
    kegg_pkn.rename(columns={"source": "source_id", "target": "target_id"}, inplace=True)
    string_pkn.rename(columns={"source": "source_id", "target": "target_id"}, inplace=True)

    trrust_pkn["source_db"] = "TRRUST"
    kegg_pkn["source_db"] = "KEGG"
    string_pkn["source_db"] = "STRING"

    # Optional: case-normalize
    for df in [trrust_pkn, kegg_pkn, string_pkn]:
        df["source_id"] = df["source_id"].str.upper()
        df["target_id"] = df["target_id"].str.upper()

    # --- Select canonical columns ---
    def select_common_columns(df):
        keep = [c for c in df.columns if c in {"source_id", "target_id", "source_db"} or c.endswith("_sign") or c.endswith("_score") or c in {"signal"}]
        return df[keep]

    trrust_pkn = select_common_columns(trrust_pkn)
    kegg_pkn   = select_common_columns(kegg_pkn)
    string_pkn = select_common_columns(string_pkn)

    # --- Merge all sources ---
    logging.info("\nMerging all PKNs")
    merged_df = pd.concat([trrust_pkn, kegg_pkn, string_pkn], ignore_index=True)
    logging.info("Done!")

    # Drop perfect duplicates (same source-target pair + identical source_db)
    merged_df.drop_duplicates(subset=["source_id", "target_id", "source_db"], inplace=True)

    logging.info(f"\nUnified PKN: {len(merged_df):,} edges across {merged_df['source_db'].nunique()} sources")

    # --- Save outputs ---
    merged_df_path_prefix = PKN_DIR / f"{ORGANISM_CODE}_merged_pkn"
    os.makedirs(os.path.dirname(merged_df_path_prefix), exist_ok=True)
    
    merged_df.to_csv(f"{merged_df_path_prefix}.csv", index=False)
    G_merged = nx.from_pandas_edgelist(merged_df, source="source_id", target="target_id", edge_attr=True, create_using=nx.DiGraph())
    nx.write_graphml(G_merged, f"{merged_df_path_prefix}.graphml")