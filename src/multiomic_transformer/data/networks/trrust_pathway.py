# trrust_pathway.py

import os
import io
import logging
import requests
from typing import Union
import pandas as pd
import networkx as nx
import pickle

TRRUST_MOUSE_URL = "https://www.grnpedia.org/trrust/data/trrust_rawdata.mouse.tsv"
TRRUST_HUMAN_URL = "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv"

def _load_trrust(path_or_url: str) -> pd.DataFrame:
    if path_or_url.startswith(("http://", "https://")):
        r = requests.get(path_or_url, timeout=60)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), sep="\t", header=None)
    else:
        df = pd.read_csv(path_or_url, sep="\t", header=None)
    if df.shape[1] < 3:
        raise ValueError("Unexpected TRRUST format.")
    df = df.iloc[:, :4]
    df.columns = ["TF", "TG", "Regulation", "PMIDs"][:df.shape[1]]
    return df

def build_trrust_pkn(
    species: str = "mouse",              # "mouse" | "human"
    *,
    trrust_path_or_url: Union[str, None] = None,  # if None, uses default URL for species
    normalize_case: str = "upper",       # "upper" | "lower" | None
    out_csv: Union[str, None] = None,
    out_gpickle: Union[str, None] = None
):
    """
    Build the FULL TRRUST PKN (directed, signed) for a species.

    Output columns:
      TF, TG, trrust_sign (-1/0/+1), trrust_regulation, trrust_pmids, trrust_support_n
    """
    if trrust_path_or_url is None:
        if species.lower().startswith("mouse"):
            trrust_path_or_url = TRRUST_MOUSE_URL
        elif species.lower().startswith("human"):
            trrust_path_or_url = TRRUST_HUMAN_URL
        else:
            raise ValueError("species must be 'mouse' or 'human' (or provide trrust_path_or_url).")

    logging.info(f"Loading TRRUST from {trrust_path_or_url}")
    trrust = _load_trrust(trrust_path_or_url)

    def _canon(s: pd.Series) -> pd.Series:
        s = s.astype(str)
        if normalize_case == "upper":
            return s.str.upper()
        if normalize_case == "lower":
            return s.str.lower()
        return s

    trrust["TF"] = _canon(trrust["TF"])
    trrust["TG"] = _canon(trrust["TG"])

    reg_map = {"Activation": 1, "Repression": -1, "Unknown": 0,
               "ACTIVATION": 1, "REPRESSION": -1, "UNKNOWN": 0}
    trrust["trrust_sign"] = trrust["Regulation"].map(reg_map).fillna(0).astype(int)
    trrust["trrust_regulation"] = trrust["Regulation"].astype(str)

    # aggregate duplicates (same TF→TG)
    def _agg_pmids(series: pd.Series) -> str:
        vals = []
        for x in series.dropna().astype(str):
            vals.extend([p.strip() for p in x.split(",") if p.strip()])
        return ",".join(sorted(set(vals)))

    pkn = (
        trrust.groupby(["TF", "TG"], as_index=False)
              .agg(
                  trrust_sign=("trrust_sign", "max"),
                  trrust_regulation=("trrust_regulation", lambda x: ";".join(sorted(set(map(str, x))))),
                  trrust_pmids=("PMIDs", _agg_pmids) if "PMIDs" in trrust.columns else ("Regulation", "size"),
                  trrust_support_n=("Regulation", "size"),
              )
    )
    if "trrust_pmids" not in pkn.columns:
        pkn["trrust_pmids"] = ""

    # optional outputs
    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        pkn.to_csv(out_csv, index=False)
        logging.info(f"Wrote TRRUST PKN CSV → {out_csv}")

    if out_gpickle:
        G = nx.from_pandas_edgelist(
            pkn, source="TF", target="TG",
            edge_attr=["trrust_sign", "trrust_regulation", "trrust_pmids", "trrust_support_n"],
            create_using=nx.DiGraph()
        )
        with open(out_gpickle, 'wb') as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
        logging.info(f"Wrote TRRUST PKN GraphML → {out_gpickle}")
