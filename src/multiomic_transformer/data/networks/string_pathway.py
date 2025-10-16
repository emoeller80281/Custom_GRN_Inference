# string_pathway.py

import os
import gc
import logging
import requests
import pandas as pd
import numpy as np
import networkx as nx
from typing import Union

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
# Batched directed/undirected PKN writer
# -----------------------------
def write_string_pkn_batched(links_iterable, *, as_directed, out_csv=None, out_graphml=None, normalize_case="upper"):
    """
    Writes the STRING PKN in batches, without ever holding the entire dataset in memory.

    Parameters
    ----------
    links_iterable : Iterable[pd.DataFrame]
        Iterator of pre-processed STRING chunks (with protein1, protein2, string_* columns).
    as_directed : bool
        Whether to duplicate edges both directions.
    out_csv : str, optional
        Path to append CSV output.
    out_graphml : str, optional
        If provided and total edges < 5M, build graph in memory and write GraphML.
    normalize_case : str, optional
        "upper", "lower", or None.
    """
    csv_mode = "w"
    csv_header = True
    G = nx.DiGraph() if as_directed else nx.Graph()

    for i, chunk in enumerate(links_iterable, 1):
        logging.info(f"  Building PKN batch {i} ({len(chunk):,} edges)…")

        # Case normalization (optional)
        if normalize_case == "upper":
            chunk["protein1"] = chunk["protein1"].str.upper()
            chunk["protein2"] = chunk["protein2"].str.upper()
        elif normalize_case == "lower":
            chunk["protein1"] = chunk["protein1"].str.lower()
            chunk["protein2"] = chunk["protein2"].str.lower()

        # ---- Directed ----
        if as_directed:
            fwd = chunk.rename(columns={"protein1": "source_id", "protein2": "target_id"})
            rev = chunk.rename(columns={"protein1": "target_id", "protein2": "source_id"})
            pkn_chunk = pd.concat([fwd, rev], ignore_index=True)
            pkn_chunk.drop_duplicates(subset=["source_id", "target_id"], inplace=True)
        # ---- Undirected ----
        else:
            a = chunk["protein1"]
            b = chunk["protein2"]
            protein_a = np.where(a <= b, a, b)
            protein_b = np.where(a <= b, b, a)
            pkn_chunk = chunk.copy()
            pkn_chunk.insert(0, "protein_a", protein_a)
            pkn_chunk.insert(1, "protein_b", protein_b)
            pkn_chunk.drop(columns=["protein1", "protein2"], inplace=True)
            pkn_chunk.drop_duplicates(subset=["protein_a", "protein_b"], inplace=True)

        # ---- CSV output (append mode) ----
        if out_csv:
            os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
            pkn_chunk.to_csv(out_csv, mode=csv_mode, header=csv_header, index=False)
            csv_mode, csv_header = "a", False  # subsequent chunks append

        # ---- Optional GraphML accumulation ----
        if out_graphml and len(G) < 5_000_000:  # guard: skip if too large
            if as_directed:
                # Directed edges
                for idx, (u, v) in enumerate(zip(pkn_chunk["source_id"], pkn_chunk["target_id"])):
                    edge_attrs = {
                        k: pkn_chunk.iloc[idx][k]
                        for k in pkn_chunk.columns
                        if k.startswith("string_")
                    }
                    G.add_edge(u, v, **edge_attrs)
            else:
                for idx, (u, v) in enumerate(zip(pkn_chunk["protein_a"], pkn_chunk["protein_b"])):
                    edge_attrs = {
                        k: pkn_chunk.iloc[idx][k]
                        for k in pkn_chunk.columns
                        if k.startswith("string_")
                    }
                    G.add_edge(u, v, **edge_attrs)


        del pkn_chunk, chunk
        gc.collect()

    # ---- Write GraphML if requested ----
    if out_graphml and len(G) < 5_000_000:
        nx.write_graphml(G, out_graphml)
        logging.info(f"Wrote STRING PKN GraphML → {out_graphml}")
    elif out_graphml:
        logging.warning("Skipping GraphML export (graph too large).")

    logging.info("Finished writing all STRING PKN batches.")


# -----------------------------
# FULL PKN builder (no filtering)
# -----------------------------

def build_string_pkn(
    string_dir: str,
    string_org_code: str = "10090",
    *,
    normalize_case: str = "upper",       # "upper" | "lower" | None
    min_combined_score: Union[int, None] = None, # keep all if None; STRING scale 0..1000
    as_directed: bool = False,           # False -> undirected canonical pairs; True -> duplicate rows (both directions)
    out_csv: Union[str, None] = None,          # optional CSV path
    out_graphml: Union[str, None] = None       # optional GraphML path
):
    """
    Build the FULL STRING v12.0 PKN (for one organism) as a tidy edge list with STRING scores.

    Returns a DataFrame with columns (undirected mode):
        protein_a, protein_b, string_combined_score, string_experimental_score, ... (present if available)

    If as_directed=True, returns columns:
        source_id, target_id, string_*  (both directions for each edge)

    Notes
    -----
    - Maps STRING protein IDs → preferred_name
    - Drops edges with unmapped endpoints
    - Normalizes combined_score to [0,1] as 'string_combined_score'
    - Keeps any other present STRING evidence columns with 'string_' prefix
    """
    # Ensure files
    paths = ensure_string_v12_files(string_dir, string_org_code)
    protein_info_path = paths["protein_info_gz"]
    links_path = paths["protein_links_detailed_gz"]

    # Read
    logging.info("Reading STRING protein.info (v12.0)…")
    protein_info_df = pd.read_csv(protein_info_path, sep="\t", compression="gzip")

    logging.info("Reading STRING protein.links.detailed (v12.0) in chunks…")

    cols_needed = [
        "protein1", "protein2",
        "combined_score", "experimental", "database",
        "coexpression", "textmining", "neighborhood",
        "fusion", "cooccurence"
    ]

    # Map from STRING IDs → preferred names once
    id_col = "#string_protein_id" if "#string_protein_id" in protein_info_df.columns else protein_info_df.columns[0]
    if "preferred_name" not in protein_info_df.columns:
        raise ValueError("STRING protein_info file is missing 'preferred_name' column")
    id_to_name = protein_info_df.set_index(id_col)["preferred_name"].to_dict()

    # Prepare the output parquet/CSV or a list of processed chunks
    def processed_chunks():

        reader = pd.read_csv(
            links_path,
            sep=" ",
            compression="gzip" if links_path.endswith(".gz") else None,
            usecols=lambda c: c in cols_needed,
            chunksize=5_000_000,
            low_memory=False,
        )

        for i, chunk in enumerate(reader, start=1):

            # Filter early by score
            if min_combined_score is not None and "combined_score" in chunk.columns:
                chunk = chunk[chunk["combined_score"] >= min_combined_score]

            # Map STRING IDs → preferred names
            chunk["protein1"] = chunk["protein1"].map(id_to_name)
            chunk["protein2"] = chunk["protein2"].map(id_to_name)
            chunk = chunk.dropna(subset=["protein1", "protein2"])

            # Rename columns (no large copy)
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

            # Normalize combined_score to [0,1]
            if "string_combined_score" in chunk.columns:
                chunk["string_combined_score"] = chunk["string_combined_score"] / 1000.0

            yield chunk

    write_string_pkn_batched(
        processed_chunks(),
        as_directed=as_directed,
        out_csv=out_csv,
        out_graphml=out_graphml,
        normalize_case=normalize_case,
    )
