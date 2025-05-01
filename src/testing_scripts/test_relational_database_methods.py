import dask.dataframe as dd
import pandas as pd

output_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/mESC/filtered_L2_E7.5_rep1"

rna_data_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/mESC/filtered_L2_E7.5_rep1/mESC_filtered_L2_E7.5_rep1_RNA_processed.parquet"
atac_data_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/mESC/filtered_L2_E7.5_rep1/mESC_filtered_L2_E7.5_rep1_ATAC_processed.parquet"

# 1) Read the big Parquet tables
sliding_window_dd = dd.read_parquet(f"{output_dir}/sliding_window_tf_to_peak_score.parquet").head(10000)
homer_dd          = dd.read_parquet(f"{output_dir}/homer_tf_to_peak.parquet").head(10000)

peak_corr_dd = dd.read_parquet(f"{output_dir}/peak_to_gene_correlation.parquet").head(10000)
cicero_dd    = dd.read_parquet(f"{output_dir}/cicero_peak_to_tg_scores.parquet").head(10000)

# 3) Read RNA data, compute mean TF expression
rna_df = dd.read_parquet(rna_data_file).head(1000)
rna_df["mean_TF_expression"] = rna_df.select_dtypes("number").mean(axis=1)
dd_rna_tf = rna_df[["gene_id", "mean_TF_expression"]].rename(columns={"gene_id": "source_id"})

# 5) Add mean TG expression
rna_df = rna_df.drop(columns=["mean_TF_expression"])
rna_df["mean_TG_expression"] = rna_df.select_dtypes("number").mean(axis=1)
dd_rna_tg = rna_df[["gene_id", "mean_TG_expression"]].rename(columns={"gene_id": "target_id"})

# 7) Read ATAC data, compute mean peak accessibility
atac_df = dd.read_parquet(atac_data_file).head(1000)
atac_df["mean_peak_accessibility"] = atac_df.select_dtypes("number").mean(axis=1)
dd_atac = atac_df[["peak_id", "mean_peak_accessibility"]]

dataframe_dict = {
    "sliding_window_dd" : sliding_window_dd,
    "homer_dd" : homer_dd,
    "peak_corr_dd" : peak_corr_dd,
    "cicero_dd" : cicero_dd,
    "dd_rna_tf" : dd_rna_tf,
    "dd_rna_tg" : dd_rna_tg,
    "dd_atac" : dd_atac
}

edge_dict = {
    "source_id" : set(),
    "peak_id" : set(),
    "target_id" : set()
}

tf_to_peak = []
peak_to_tg = []


def extract_edges(df, edge_cols):
    edges = set()
    edge_col_df = df[edge_cols]
    for _, row in edge_col_df.iterrows():
        edge_tuple = tuple(row[col] for col in edge_cols)
        edges.add(edge_tuple)
    return edges

sliding_window_edges = extract_edges(sliding_window_dd, ["source_id", "peak_id"])
homer_edges = extract_edges(homer_dd, ["source_id", "peak_id"])
peak_corr_edges = extract_edges(peak_corr_dd, ["peak_id", "target_id"])
cicero_edges = extract_edges(cicero_dd, ["peak_id", "target_id"])

def build_full_edges(tf_peak_edges, peak_tg_edges):
    full_edges = set()
    # Create a lookup from peak_id to list of target_ids
    peak_to_targets = {}
    for peak_id, target_id in peak_tg_edges:
        peak_to_targets.setdefault(peak_id, set()).add(target_id)

    # Now join: for each (source_id, peak_id), attach matching target_ids
    for source_id, peak_id in tf_peak_edges:
        if peak_id in peak_to_targets:
            for target_id in peak_to_targets[peak_id]:
                full_edges.add((source_id, peak_id, target_id))
    
    return full_edges

# Combine all TF→peak edges
tf_peak_edges = sliding_window_edges | homer_edges

# Combine all peak→TG edges
peak_tg_edges = peak_corr_edges | cicero_edges

# Build full source-peak-target edges
full_edges = build_full_edges(tf_peak_edges, peak_tg_edges)

# Create a Pandas DataFrame of the unique edges
full_edges_df = pd.DataFrame(list(full_edges), columns=["source_id", "peak_id", "target_id"])

# Merge score tables based on the combind edges
combined_df = (
    full_edges_df
    .merge(sliding_window_dd, on=["source_id", "peak_id"], how="left")
    .merge(homer_dd, on=["source_id", "peak_id"], how="left")
    .merge(peak_corr_dd, on=["peak_id", "target_id"], how="left")
    .merge(cicero_dd, on=["peak_id", "target_id"], how="left")
    .merge(dd_rna_tf, on="source_id", how="left")
    .merge(dd_rna_tg, on="target_id", how="left")
    .merge(dd_atac, on="peak_id", how="left")
)

print(combined_df.head())
