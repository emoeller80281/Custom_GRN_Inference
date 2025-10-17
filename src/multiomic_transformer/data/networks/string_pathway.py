# string_pathway.py

import os
import gc
import logging
import requests
import pandas as pd
import numpy as np
import networkx as nx
import pickle
from typing import Union
from joblib import Parallel, delayed

# -----------------------------
# Download helpers (STRING v12)
# -----------------------------

def _download(url: str, dest_path: str, chunk_size: int = 1 << 20) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    logging.info(f"Downloading {url} → {dest_path}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

def ensure_string_v12_files(string_dir: str, string_org_code: str) -> dict:
    """
    Ensure STRING v12.0 files exist locally; download if missing.

    Parameters
    ----------
    string_dir : str
        Directory to store STRING files.
    string_org_code : str
        NCBI taxonomy code used by STRING (e.g., '10090' for mouse, '9606' for human).

    Returns
    -------
    dict with:
      - protein_info_gz
      - protein_links_detailed_gz
      - protein_info_url
      - protein_links_detailed_url
    """
    base = "https://stringdb-downloads.org/download"

    protein_info_gz = os.path.join(
        string_dir, f"{string_org_code}.protein.info.v12.0.txt.gz"
    )
    protein_info_url = f"{base}/protein.info.v12.0/{string_org_code}.protein.info.v12.0.txt.gz"

    links_det_gz = os.path.join(
        string_dir, f"{string_org_code}.protein.links.detailed.v12.0.txt.gz"
    )
    links_det_url = f"{base}/protein.links.detailed.v12.0/{string_org_code}.protein.links.detailed.v12.0.txt.gz"

    if not os.path.exists(protein_info_gz):
        _download(protein_info_url, protein_info_gz)
    else:
        logging.info(f"Found: {protein_info_gz}")

    if not os.path.exists(links_det_gz):
        _download(links_det_url, links_det_gz)
    else:
        logging.info(f"Found: {links_det_gz}")

    return {
        "protein_info_gz": protein_info_gz,
        "protein_links_detailed_gz": links_det_gz,
        "protein_info_url": protein_info_url,
        "protein_links_detailed_url": links_det_url,
    }


# -----------------------------
# FULL PKN builder (no filtering)
# -----------------------------

def build_string_pkn(
    string_dir: str,
    string_org_code: str = "10090",
    *,
    normalize_case: str = "upper",
    min_combined_score: Union[int, None] = None,
    as_directed: bool = False,
    out_csv: Union[str, None] = None,
    out_gpickle: Union[str, None] = None
):
    """
    Build the FULL STRING v12.0 PKN (for one organism) as a tidy edge list with STRING scores.
    Streams chunks efficiently and optionally builds a small NetworkX graph (<15M edges).
    """
    # Ensure files exist
    paths = ensure_string_v12_files(string_dir, string_org_code)
    protein_info_path = paths["protein_info_gz"]
    links_path = paths["protein_links_detailed_gz"]

    # --- Load protein info ---
    logging.info("Reading STRING protein.info (v12.0)…")
    protein_info_df = pd.read_csv(protein_info_path, sep="\t", compression="gzip")

    id_col = "#string_protein_id" if "#string_protein_id" in protein_info_df.columns else protein_info_df.columns[0]
    if "preferred_name" not in protein_info_df.columns:
        raise ValueError("STRING protein_info file is missing 'preferred_name' column")

    id_to_name = protein_info_df.set_index(id_col)["preferred_name"].to_dict()

    # --- Prepare CSV output ---
    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        if os.path.exists(out_csv):
            os.remove(out_csv)

    # --- Stream chunks ---
    logging.info("Reading STRING protein.links.detailed (v12.0) in chunks…")

    cols_needed = [
        "protein1", "protein2",
        "combined_score", "experimental", "database",
        "coexpression", "textmining", "neighborhood",
        "fusion", "cooccurence"
    ]

    reader = pd.read_csv(
        links_path,
        sep=" ",
        compression="gzip" if links_path.endswith(".gz") else None,
        usecols=lambda c: c in cols_needed,
        chunksize=5_000_000,
        low_memory=False,
    )

    # --- Initialize optional graph ---
    G = nx.DiGraph() if as_directed else nx.Graph()

    for i, chunk in enumerate(reader, start=1):
        logging.info(f"Processing STRING chunk {i} ({len(chunk):,} rows)…")

        # Filter by score
        if min_combined_score is not None and "combined_score" in chunk.columns:
            chunk = chunk[chunk["combined_score"] >= min_combined_score].copy()

        # Map STRING IDs → gene names
        chunk["protein1"] = chunk["protein1"].map(id_to_name)
        chunk["protein2"] = chunk["protein2"].map(id_to_name)
        chunk.dropna(subset=["protein1", "protein2"], inplace=True)

        # Rename columns
        rename_map = {
            "combined_score": "string_combined_score",
            "experimental": "string_experimental_score",
            "database": "string_database_score",
            "coexpression": "string_coexpression_score",
            "textmining": "string_textmining_score",
            "neighborhood": "string_neighborhood_score",
            "fusion": "string_fusion_score",
            "cooccurence": "string_cooccurence_score",
        }
        chunk.rename(columns=rename_map, inplace=True)

        # Normalize
        if "string_combined_score" in chunk.columns:
            chunk["string_combined_score"] = chunk["string_combined_score"] / 1000.0

        # Write batch to CSV
        if out_csv:
            chunk.to_csv(out_csv, mode="a", header=(i == 1), index=False)

        # Optionally add to NetworkX (guard: skip if huge)
        if out_gpickle and len(G) < 15_000_000:
            if as_directed:
                for idx, (u, v) in enumerate(zip(chunk["protein1"], chunk["protein2"])):
                    edge_attrs = {k: chunk.iloc[idx][k] for k in chunk.columns if k.startswith("string_")}
                    G.add_edge(u, v, **edge_attrs)
                    G.add_edge(v, u, **edge_attrs)  # duplicate reverse
            else:
                for idx, (u, v) in enumerate(zip(chunk["protein1"], chunk["protein2"])):
                    edge_attrs = {k: chunk.iloc[idx][k] for k in chunk.columns if k.startswith("string_")}
                    G.add_edge(min(u, v), max(u, v), **edge_attrs)

        del chunk
        gc.collect()

    if out_gpickle and len(G) < 15_000_000:
        with open(out_gpickle, 'wb') as f:
            pickle.dump(G, f, pickle.HIGHEST_PROTOCOL)
        logging.info(f"Wrote STRING PKN graph → {out_gpickle}")
    elif out_gpickle:
        logging.warning("Skipping .gpickle export (graph too large)")

    logging.info("Finished building STRING PKN.")

