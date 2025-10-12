import logging
import pandas as pd
import numpy as np

def add_string_db_scores(
    inferred_net_dd,
    string_dir,
    tf_col_name="source_id",
    tg_col_name="target_id",
    min_combined_score=None,        # STRING combined_score threshold (0-1000). e.g., 700 for high confidence
    normalize_combined=True,        # convert combined_score to [0,1]
    lowercase=False,                # optionally lowercase gene names prior to merge
    uppercase=True,                 # or uppercase (applied after lowercase if both True)
    drop_unmapped=True              # drop STRING rows that fail ID→name mapping
):
    """
    Merge STRING PPI evidence onto your TF–TG edges by gene symbol (preferred_name).

    Notes
    -----
    - Assumes inferred_net_dd columns contain gene symbols that match STRING preferred_name.
    - STRING files expected:
        - protein_info.txt             (tab-delimited; has '#string_protein_id', 'preferred_name')
        - protein_links_detailed.txt   (space-delimited; has 'protein1','protein2','combined_score', etc.)
    """

    # --- 1) Unique TF–TG pairs from your network (Dask- or Pandas-friendly)
    cols_needed = [tf_col_name, tg_col_name]
    if hasattr(inferred_net_dd, "compute"):  # Dask
        pairs_df = inferred_net_dd[cols_needed].drop_duplicates().compute()
    else:
        pairs_df = inferred_net_dd[cols_needed].drop_duplicates()

    # Optional case-normalization to improve joins
    def _canonize(s: pd.Series) -> pd.Series:
        if lowercase:
            s = s.str.lower()
        if uppercase:
            s = s.str.upper()
        return s

    pairs_df = pairs_df.copy()
    pairs_df[tf_col_name] = _canonize(pairs_df[tf_col_name].astype(str))
    pairs_df[tg_col_name] = _canonize(pairs_df[tg_col_name].astype(str))
    logging.info(f"  - Found {len(pairs_df):,} unique TF–TG pairs for STRING matching")

    # --- 2) Load STRING data
    logging.info("  - Reading STRING protein info")
    protein_info_df = pd.read_csv(f"{string_dir}/protein_info.txt", sep="\t")

    logging.info("  - Reading STRING protein links detailed")
    protein_links_df = pd.read_csv(f"{string_dir}/protein_links_detailed.txt", sep=" ")

    # --- 3) Map STRING protein IDs to preferred names and (optionally) drop unmapped
    id_col = "#string_protein_id" if "#string_protein_id" in protein_info_df.columns else protein_info_df.columns[0]
    if "preferred_name" not in protein_info_df.columns:
        raise ValueError("STRING protein_info.txt is missing 'preferred_name' column")

    id_to_name = protein_info_df.set_index(id_col)["preferred_name"].to_dict()
    protein_links_df["protein1"] = protein_links_df["protein1"].map(id_to_name)
    protein_links_df["protein2"] = protein_links_df["protein2"].map(id_to_name)

    if drop_unmapped:
        before = len(protein_links_df)
        protein_links_df = protein_links_df.dropna(subset=["protein1", "protein2"])
        logging.info(f"  - Dropped {before - len(protein_links_df):,} STRING edges with unmapped protein IDs")

    # Canonicalize preferred names to match your network’s case policy
    protein_links_df["protein1"] = _canonize(protein_links_df["protein1"].astype(str))
    protein_links_df["protein2"] = _canonize(protein_links_df["protein2"].astype(str))

    # --- 4) Optional threshold on combined_score (STRING scale 0–1000)
    if min_combined_score is not None:
        if "combined_score" not in protein_links_df.columns:
            raise ValueError("STRING links file missing 'combined_score'")
        pre = len(protein_links_df)
        protein_links_df = protein_links_df[protein_links_df["combined_score"] >= int(min_combined_score)]
        logging.info(f"  - Applied combined_score >= {min_combined_score}: kept {len(protein_links_df):,}/{pre:,}")

    # --- 5) Build both orientations so we catch undirected PPI regardless of TF/TG order
    # left merge on (TF=protein1, TG=protein2)
    left_merge = pairs_df.merge(
        protein_links_df,
        left_on=[tf_col_name, tg_col_name],
        right_on=["protein1", "protein2"],
        how="left",
    )

    # right merge on (TF=protein2, TG=protein1)
    right_merge = pairs_df.merge(
        protein_links_df,
        left_on=[tf_col_name, tg_col_name],
        right_on=["protein2", "protein1"],
        how="left",
        suffixes=("", "_rev"),
    )

    # Combine: prefer direct orientation fields; fill missing from reversed
    def _coalesce(a, b):
        return a.where(~a.isna(), b)

    # Decide which STRING columns to carry over (handle different releases)
    candidate_cols = [
        "experimental", "database", "coexpression", "textmining", "neighborhood",
        "fusion", "cooccurence", "combined_score"
    ]
    present_cols = [c for c in candidate_cols if c in protein_links_df.columns]

    merged = pairs_df.copy()
    for c in present_cols:
        c_rev = f"{c}_rev"
        merged[c] = _coalesce(left_merge.get(c), right_merge.get(c_rev))

    # Keep the protein name columns too (for debugging/QA)
    merged["string_protein1"] = _coalesce(left_merge.get("protein1"), right_merge.get("protein1_rev"))
    merged["string_protein2"] = _coalesce(left_merge.get("protein2"), right_merge.get("protein2_rev"))

    # --- 6) Normalize combined_score to [0,1] if requested
    if normalize_combined and "combined_score" in merged.columns:
        merged["string_combined_score"] = merged["combined_score"] / 1000.0
        merged.drop(columns=["combined_score"], inplace=True, errors="ignore")
    elif "combined_score" in merged.columns:
        merged.rename(columns={"combined_score": "string_combined_score"}, inplace=True)

    # Rename other commonly used score fields with 'string_' prefix
    rename_map = {
        "experimental": "string_experimental_score",
        "database": "string_database_score",
        "coexpression": "string_coexpression_score",
        "textmining": "string_textmining_score",
        "neighborhood": "string_neighborhood_score",
        "fusion": "string_fusion_score",
        "cooccurence": "string_cooccurence_score",
    }
    cols_to_rename = {k: v for k, v in rename_map.items() if k in merged.columns}
    merged.rename(columns=cols_to_rename, inplace=True)

    # --- 7) Merge back onto the original inferred network (preserve all other columns)
    # If original is Dask, convert merged (pairs) to pandas and join as pandas, or compute original first.
    if hasattr(inferred_net_dd, "compute"):
        base_df = inferred_net_dd.compute()
    else:
        base_df = inferred_net_dd

    out = base_df.merge(
        merged,
        on=[tf_col_name, tg_col_name],
        how="left",
        suffixes=("", "_stringdup"),
    )

    # Clean up any accidental suffix columns that duplicated your fields
    for col in list(out.columns):
        if col.endswith("_stringdup"):
            out.drop(columns=[col], inplace=True)

    logging.info("  - STRING scores merged into inferred network")
    return out
