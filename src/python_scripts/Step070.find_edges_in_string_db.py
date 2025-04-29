import pandas as pd
import dask.dataframe as dd
import pyarrow.parquet as pq
import argparse
import logging
import os
import sys
import numpy as np

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add STRING scores to inferred network")
    parser.add_argument("--inferred_net_file", type=str, required=True, help="Path to inferred network Parquet file")
    parser.add_argument("--string_dir", type=str, required=True, help="Path to STRING database directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    return parser.parse_args()

def minmax_normalize_column(column: pd.Series) -> pd.Series:
    return (column - column.min()) / (column.max() - column.min())

def main():
    # Check if STRING columns already present
    pf = pq.ParquetFile(INFERRED_NET_FILE)
    cols = set(pf.schema.names)

    needed = {"string_experimental_score", "string_textmining_score", "string_combined_score"}
    file_name = os.path.basename(INFERRED_NET_FILE)

    if needed.issubset(cols):
        logging.info(f"All STRING columns already present in {file_name}; skipping.")
        sys.exit(0)
    else:
        logging.info(f"Adding missing STRING columns to {file_name}")

    # Load inferred network as Dask
    logging.info("Reading inferred network file with Dask")
    inferred_net_dd = dd.read_parquet(INFERRED_NET_FILE)

    # Load STRING metadata files (small)
    logging.info("Reading STRING protein info")
    protein_info_df = pd.read_csv(f"{STRING_DIR}/protein_info.txt", sep="\t", header=0)

    logging.info("Reading STRING protein links detailed")
    protein_links_df = pd.read_csv(f"{STRING_DIR}/protein_links_detailed.txt", sep=" ", header=0)

    # Map STRING protein IDs to human-readable names
    logging.info("Mapping STRING protein IDs to preferred names")
    id_to_name = protein_info_df.set_index("#string_protein_id")["preferred_name"].to_dict()
    protein_links_df["protein1"] = protein_links_df["protein1"].map(id_to_name)
    protein_links_df["protein2"] = protein_links_df["protein2"].map(id_to_name)

    # Select relevant STRING columns
    protein_links_df = protein_links_df.rename(columns={
        "experimental": "string_experimental_score",
        "textmining": "string_textmining_score",
        "combined_score": "string_combined_score"
    })[["protein1", "protein2", "string_experimental_score", "string_textmining_score", "string_combined_score"]]

    # Convert STRING links to Dask
    logging.info("Converting STRING links to Dask")
    protein_links_dd = dd.from_pandas(protein_links_df, npartitions=1)

    # Merge inferred network with STRING scores
    logging.info("Merging inferred network with STRING edges")
    merged_dd = inferred_net_dd.merge(
        protein_links_dd,
        left_on=["source_id", "target_id"],
        right_on=["protein1", "protein2"],
        how="left"
    ).drop(columns=["protein1", "protein2"])

    # Normalize STRING score columns
    logging.info("Normalizing STRING scores")
    cols_to_normalize = ["string_experimental_score", "string_textmining_score", "string_combined_score"]

    for col in cols_to_normalize:
        if col in merged_dd.columns:
            merged_dd[col] = merged_dd[col].map_partitions(minmax_normalize_column)

    # Write out new Parquet file
    out_file = os.path.join(OUTPUT_DIR, os.path.basename(INFERRED_NET_FILE).replace(".parquet", "_w_string.parquet"))
    logging.info(f"Saving inferred network with STRING scores to {out_file}")

    merged_dd.repartition(partition_size="256MB").to_parquet(
        out_file,
        engine="pyarrow",
        compression="snappy",
        write_index=False
    )

    logging.info("Done adding STRING scores!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    INFERRED_NET_FILE = args.inferred_net_file
    STRING_DIR = args.string_dir
    OUTPUT_DIR = args.output_dir

    main()
