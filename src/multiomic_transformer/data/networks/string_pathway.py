# string_pathway.py

import os
import logging
import requests
import pandas as pd
import numpy as np
import networkx as nx

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
    normalize_case: str = "upper",       # "upper" | "lower" | None
    min_combined_score: int | None = None, # keep all if None; STRING scale 0..1000
    as_directed: bool = False,           # False -> undirected canonical pairs; True -> duplicate rows (both directions)
    out_csv: str | None = None,          # optional CSV path
    out_graphml: str | None = None       # optional GraphML path
) -> pd.DataFrame:
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

    logging.info("Reading STRING protein.links.detailed (v12.0)…")
    links_df = pd.read_csv(links_path, sep=" ", compression="gzip")

    # Map to preferred names
    id_col = "#string_protein_id" if "#string_protein_id" in protein_info_df.columns else protein_info_df.columns[0]
    if "preferred_name" not in protein_info_df.columns:
        raise ValueError("STRING protein_info file is missing 'preferred_name' column")

    id_to_name = protein_info_df.set_index(id_col)["preferred_name"].to_dict()
    links_df["protein1"] = links_df["protein1"].map(id_to_name)
    links_df["protein2"] = links_df["protein2"].map(id_to_name)

    # Drop unmapped
    before = len(links_df)
    links_df = links_df.dropna(subset=["protein1", "protein2"])
    logging.info(f"Dropped {before - len(links_df):,} edges with unmapped protein IDs")

    # Case normalization
    def _canon(s: pd.Series) -> pd.Series:
        s = s.astype(str)
        if normalize_case == "upper":
            return s.str.upper()
        if normalize_case == "lower":
            return s.str.lower()
        return s

    links_df["protein1"] = _canon(links_df["protein1"])
    links_df["protein2"] = _canon(links_df["protein2"])

    # Optional score threshold
    if min_combined_score is not None:
        if "combined_score" not in links_df.columns:
            raise ValueError("STRING links file missing 'combined_score'")
        pre = len(links_df)
        links_df = links_df[links_df["combined_score"] >= int(min_combined_score)]
        logging.info(f"Applied combined_score ≥ {min_combined_score}: kept {len(links_df):,}/{pre:,}")

    # Build tidy output with normalized + prefixed columns
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
    present_cols = [c for c in rename_map if c in links_df.columns]
    keep_cols = ["protein1", "protein2"] + present_cols

    df = links_df[keep_cols].rename(columns=rename_map).copy()

    # Normalize combined score to [0,1] if present
    if "string_combined_score" in df.columns:
        df["string_combined_score"] = df["string_combined_score"] / 1000.0

    if as_directed:
        # Duplicate as source/target both ways
        fwd = df.rename(columns={"protein1": "source_id", "protein2": "target_id"})
        rev = df.rename(columns={"protein1": "target_id", "protein2": "source_id"})
        pkn = pd.concat([fwd, rev], ignore_index=True)
        # Optional: drop exact duplicates if any
        pkn = pkn.drop_duplicates(subset=["source_id", "target_id"] + [c for c in pkn.columns if c.startswith("string_")])
    else:
        # Canonicalize undirected pairs (sorted)
        a = df["protein1"]
        b = df["protein2"]
        protein_a = np.where(a <= b, a, b)
        protein_b = np.where(a <= b, b, a)
        pkn = df.copy()
        pkn.insert(0, "protein_a", protein_a)
        pkn.insert(1, "protein_b", protein_b)
        pkn = pkn.drop(columns=["protein1", "protein2"])
        pkn = pkn.drop_duplicates(subset=["protein_a", "protein_b"])

    # Optional outputs
    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        pkn.to_csv(out_csv, index=False)
        logging.info(f"Wrote STRING PKN CSV → {out_csv}")

    if out_graphml:
        if as_directed:
            G = nx.from_pandas_edgelist(
                pkn, source="source_id", target="target_id",
                edge_attr=[c for c in pkn.columns if c.startswith("string_")],
                create_using=nx.DiGraph()
            )
        else:
            G = nx.from_pandas_edgelist(
                pkn, source="protein_a", target="protein_b",
                edge_attr=[c for c in pkn.columns if c.startswith("string_")],
                create_using=nx.Graph()
            )
        nx.write_graphml(G, out_graphml)
        logging.info(f"Wrote STRING PKN GraphML → {out_graphml}")

    return pkn
