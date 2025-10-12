import os
import io
import logging
import requests
import pandas as pd

def write_trrust_subset(
    inferred_network_df: pd.DataFrame,
    out_path: str,
    trrust_path_or_url: str = "https://www.grnpedia.org/trrust/data/trrust_rawdata.mouse.tsv",
    tf_col: str = "source_id",
    tg_col: str = "target_id",
    normalize_case: str = "upper",     # "upper" | "lower" | None
    restrict_to_pairs: bool = False,   # False: keep all TRRUST edges among network genes; True: only exact TF→TG pairs in network
    sep: str = ",",                    # file delimiter for output
) -> pd.DataFrame:
    """
    Create a TRRUST-only file filtered to your inferred network's TF/TG genes (or exact pairs).

    Output columns:
      - TF, TG, trrust_regulation, trrust_sign, trrust_pmids, trrust_support_n
    """
    # --- checks
    for c in (tf_col, tg_col):
        if c not in inferred_network_df.columns:
            raise ValueError(f"Column '{c}' not found in inferred_network_df")

    # --- load TRRUST
    logging.info("Reading TRRUST mouse...")
    if trrust_path_or_url.startswith(("http://", "https://")):
        resp = requests.get(trrust_path_or_url, timeout=60)
        resp.raise_for_status()
        trrust = pd.read_csv(io.StringIO(resp.text), sep="\t", header=None)
    else:
        trrust = pd.read_csv(trrust_path_or_url, sep="\t", header=None)

    if trrust.shape[1] < 3:
        raise ValueError("Unexpected TRRUST format (fewer than 3 columns).")

    trrust = trrust.iloc[:, :4]
    trrust.columns = ["TF", "TG", "Regulation", "PMIDs"][:trrust.shape[1]]

    # --- normalize case
    def _canon(s: pd.Series) -> pd.Series:
        s = s.astype(str)
        if normalize_case == "upper":
            return s.str.upper()
        if normalize_case == "lower":
            return s.str.lower()
        return s

    trrust["TF"] = _canon(trrust["TF"])
    trrust["TG"] = _canon(trrust["TG"])

    net = inferred_network_df[[tf_col, tg_col]].copy()
    net[tf_col] = _canon(net[tf_col])
    net[tg_col] = _canon(net[tg_col])

    # --- aggregate duplicates in TRRUST (same TF→TG multiple evidence lines)
    def _agg_pmids(series: pd.Series) -> str:
        vals = []
        for x in series.dropna().astype(str):
            vals.extend([p.strip() for p in x.split(",") if p.strip()])
        return ",".join(sorted(set(vals)))

    reg_map = {"Activation": 1, "Repression": -1, "Unknown": 0,
               "ACTIVATION": 1, "REPRESSION": -1, "UNKNOWN": 0}

    trrust["trrust_sign"] = trrust["Regulation"].map(reg_map).fillna(0).astype(int)
    trrust["trrust_regulation"] = trrust["Regulation"].astype(str)

    trrust_agg = (
        trrust.groupby(["TF", "TG"], as_index=False)
              .agg(
                  trrust_sign=("trrust_sign", "max"),  # ±1 dominates 0 if mixed
                  trrust_regulation=("trrust_regulation", lambda x: ";".join(sorted(set(map(str, x))))),
                  trrust_pmids=("PMIDs", _agg_pmids) if "PMIDs" in trrust.columns else ("Regulation", "size"),
                  trrust_support_n=("Regulation", "size"),
              )
    )
    if "trrust_pmids" not in trrust_agg.columns:
        trrust_agg["trrust_pmids"] = ""

    # --- filter: by gene set or by exact pairs
    if restrict_to_pairs:
        # keep only TF→TG pairs that exactly appear in the network
        keep = net.drop_duplicates().merge(
            trrust_agg, left_on=[tf_col, tg_col], right_on=["TF", "TG"], how="inner"
        )[["TF", "TG"]].drop_duplicates()
        trrust_filt = trrust_agg.merge(keep, on=["TF", "TG"], how="inner")
    else:
        # keep TRRUST edges where both genes appear anywhere in the network (TF or TG)
        genes = set(net[tf_col]).union(set(net[tg_col]))
        trrust_filt = trrust_agg[trrust_agg["TF"].isin(genes) & trrust_agg["TG"].isin(genes)].copy()

    # --- write file
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cols = ["TF", "TG", "trrust_regulation", "trrust_sign", "trrust_pmids", "trrust_support_n"]
    trrust_filt.to_csv(out_path, index=False, sep=sep, columns=cols)
    logging.info(f"Wrote TRRUST subset: {trrust_filt.shape[0]:,} edges → {out_path}")

    return trrust_filt
