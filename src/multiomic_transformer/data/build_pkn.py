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

    string_file = STRING_DIR / "string_mouse_pkn_directed.csv"
    trrust_file = TRRUST_DIR / "trrust_mouse_pkn.csv"
    kegg_file = KEGG_DIR / "kegg_mouse_pkn.csv"

    # if not os.path.isfile(string_file):
    #     print("Building STRING prior knowledge network")
    #     string_pkn = string_pathway.build_string_pkn(
    #         string_dir=str(STRING_DIR),
    #         string_org_code="10090",
    #         as_directed=True,
    #         out_csv=str(string_file)
    #     )
    # else:
    #     string_pkn = pd.read_csv(string_file)

    # if not os.path.isfile(trrust_file):
    #     print("Building TRRUST prior knowledge network")
    #     trrust_pkn = trrust_pathway.build_trrust_pkn(
    #         species="mouse",
    #         out_csv=str(TRRUST_DIR / "trrust_mouse_pkn.csv")
    #     )
    # else:
    #     trrust_pkn = pd.read_csv(trrust_file)

    if not os.path.isfile(kegg_file):
        print("Building KEGG prior knowledge network")
        kegg_pkn = kegg_pathways.build_kegg_pkn(
            dataset_name=DATASET_NAME,
            output_path=str(KEGG_DIR),
            organism="mmu",
            out_csv=str(kegg_file)
        )
    else:
        
        kegg_pkn = pd.read_csv(kegg_file)
