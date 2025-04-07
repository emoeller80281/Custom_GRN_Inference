import pandas as pd
import argparse
import logging

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments containing paths for input and output files and CPU count.
    """
    parser = argparse.ArgumentParser(description="Process TF motif binding potential.")
    parser.add_argument(
        "--inferred_net_file",
        type=str,
        required=True,
        help="Path to the inferred network (network must contain at least 'source_id' and 'target_id' columns)"
    )
    parser.add_argument(
        "--string_dir",
        type=str,
        required=True,
        help="Path to the directory containing the STRING database files"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for the sample"
    )

    args: argparse.Namespace = parser.parse_args()
    return args

def main():
    logging.info("----- READING STRING DATABASE FILES -----")
    logging.info("\tProtein Info DataFrame")
    protein_info_df = pd.read_csv(f'{STRING_DIR}/protein_info.txt', sep="\t", header=0)

    logging.info("\tProtein Links DataFrame")
    protein_links_df = pd.read_csv(f'{STRING_DIR}/protein_links_detailed.txt', sep=" ", header=0)

    logging.info("\nReading inferred network file")
    inferred_net_df = pd.read_csv(INFERRED_NET_FILE, header=0)
    logging.info("\tDone!")

    # Find the common name for the proteins in protein_links_df using the ID to name mapping in protein_info_df
    logging.info("Converting STRING protein IDs to protein name")
    protein_links_df["protein1"] = protein_info_df.set_index("#string_protein_id").loc[protein_links_df["protein1"], "preferred_name"].reset_index()["preferred_name"]
    protein_links_df["protein2"] = protein_info_df.set_index("#string_protein_id").loc[protein_links_df["protein2"], "preferred_name"].reset_index()["preferred_name"]

    cols_to_keep = ["protein1", "protein2", "experimental", "textmining", "combined_score"]
    protein_links_df = protein_links_df[cols_to_keep]
    
    # Merge on the canonical edge column
    logging.info("Merging the STRING edges with the inferred network edges")
    inferred_edges_in_string_df = pd.merge(
        inferred_net_df,
        protein_links_df,
        left_on=["source_id", "target_id"],
        right_on=["protein1", "protein2"],
        how="inner"
    ).drop(columns={"protein1", "protein2"})
    logging.info("\tDone!")

    logging.info(f'\tInferred edges in string:')
    logging.info(f'{inferred_edges_in_string_df.head()}')

    logging.info(f"Found {len(inferred_edges_in_string_df)} common edges between the STRING protein-protein interaction database and the inferred network.")

    inferred_edges_in_string_df.to_csv(f'{OUTPUT_DIR}/inferred_net_w_string.csv', header=True, index=False)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Parse command-line arguments
    args: argparse.Namespace = parse_args()
    
    INFERRED_NET_FILE: str = args.inferred_net_file
    STRING_DIR: str = args.string_dir
    OUTPUT_DIR: str = args.output_dir
    
    main()