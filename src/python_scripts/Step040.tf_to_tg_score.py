import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
import logging
import argparse
import gc
import concurrent.futures

def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
    argparse.Namespace: Parsed arguments containing paths for input and output files.
    """
    parser = argparse.ArgumentParser(description="Process TF motif binding potential.")
    parser.add_argument(
        "--rna_data_file",
        type=str,
        required=True,
        help="Path to the scRNAseq data file"
    )
    parser.add_argument(
        "--atac_data_file",
        type=str,
        required=True,
        help="Path to the scATACseq data file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for the sample"
    )
    parser.add_argument(
        "--cell_level_net_dir",
        type=str,
        required=False,
        help="Path to the directory with the cell-level inferred network score pickle files"
    )
    parser.add_argument(
        "--fig_dir",
        type=str,
        required=True,
        help="Path to the figure directory for the sample"
    )
    parser.add_argument(
        "--num_cpu",
        type=str,
        required=False,
        help="Number of processors to run multithreading with"
    )
    parser.add_argument(
        "--num_cells",
        type=str,
        required=False,
        help="Number of cells to generate cell-level grn score dataframes for"
    )
    parser.add_argument(
        "--bulk_or_cell",
        type=str,
        required=True,
        choices=["bulk", "cell"],
        help="Choose 'bulk' to create a single inferred grn, or 'cell' to create cell-level inferred grns"
    )
    
    
    args: argparse.Namespace = parser.parse_args()

    return args

def minmax_normalize_column(column: pd.DataFrame):
    return (column - column.min()) / (column.max() - column.min())

def load_rna_dataset(rna_data_file):
    # Read in the RNAseq data file and extract the gene names to find matching TFs
    # logging.info("Reading and formatting scRNA-seq expression data")
    rna_data = pd.read_csv(rna_data_file, index_col=None, header=0)

    rna_data.rename(columns={rna_data.columns[0]: "gene"}, inplace=True)

    def calculate_rna_cpm(rna_data):  
        # Calculate the normalized log2 counts/million for each gene in each cell
        RNA_dataset = rna_data.astype({col: float for col in rna_data.columns[1:]})
        
        # Find the total number of reads for the cell
        column_sum = np.array(RNA_dataset.iloc[:, 1:].sum(axis=1, numeric_only=True))
        expression_matrix = RNA_dataset.iloc[:, 1:].values
        
        # Scale the counts for each gene by the total number of read counts for the cell * 1e6 to get CPM
        rna_cpm = np.log2(((expression_matrix.T / column_sum).T * 1e6) + 1)
        
        # rna_cpm = minmax_normalize_column(rna_cpm)
        rna_cpm_df = pd.DataFrame(rna_cpm, index=RNA_dataset.index, columns=RNA_dataset.columns[1:])

        RNA_dataset.iloc[:, 1:] = rna_cpm_df

        # Transpose RNA dataset for easier access
        rna_data = RNA_dataset.set_index("gene")  # Rows = cells, Columns = genes

        return rna_data

    return calculate_rna_cpm(rna_data)

def load_atac_dataset(atac_data_file):
    # Read in the RNAseq data file and extract the gene names to find matching TFs
    # logging.info("Reading and formatting scATAC-seq expression data")
    atac_data = pd.read_csv(atac_data_file, index_col=None, header=0)

    atac_data.rename(columns={atac_data.columns[0]: "peak"}, inplace=True)

    atac_data = atac_data.set_index("peak")
    logging.info(atac_data.head())
    
    def log2_cpm_normalize(atac_df):
        """
        Log2 CPM normalize the values in atac_df.
        Assumes:
        - atac_df's first column is a non-numeric peak identifier (e.g., "chr1:100-200"),
        - columns 1..end are numeric count data for samples or cells.
        """
        # Separate the non-numeric first column
        peak_ids = atac_df.iloc[:, 0]
        # Numeric counts
        counts = atac_df.iloc[:, 1:]
        
        # 1. Compute library sizes (sum of each column)
        library_sizes = counts.sum(axis=0)
        
        # 2. Convert counts to CPM
        # Divide each column by its library size, multiply by 1e6
        # Add 1 to avoid log(0) issues in the next step
        cpm = (counts.div(library_sizes, axis=1) * 1e6).add(1)
        
        # 3. Log2 transform
        log2_cpm = np.log2(cpm)
        
        # Reassemble into a single DataFrame
        normalized_df = pd.concat([peak_ids, log2_cpm], axis=1)
        # Optionally rename columns if needed
        # normalized_df.columns = ...
        
        return normalized_df
    
    atac_data = log2_cpm_normalize(atac_data)

    return atac_data

def load_tf_to_peak_scores(tf_to_peak_score_file):
    # logging.info("Reading and formatting TF to peak binding scores")
    tf_to_peak_score = pd.read_csv(tf_to_peak_score_file, sep="\t", header=0, index_col=None).rename(columns={"binding_score":"tf_to_peak_binding_score"})
    tf_to_peak_score = tf_to_peak_score.melt(id_vars="peak", var_name="gene", value_name="tf_to_peak_binding_score")
    tf_to_peak_score["tf_to_peak_binding_score"] = minmax_normalize_column(tf_to_peak_score["tf_to_peak_binding_score"]) # Normalize binding score values
    
    return tf_to_peak_score

def load_peak_to_tg_scores(peak_to_tg_score_file):
    # logging.info("Reading and formatting peak to TG scores")
    peak_to_tg_score = pd.read_csv(peak_to_tg_score_file, sep="\t", header=0, index_col=None)

    # Format the peaks to match the tf_to_peak_score dataframe
    peak_to_tg_score = peak_to_tg_score[["peak", "gene", "score"]]

    peak_to_tg_score = peak_to_tg_score.rename(columns={"score": "peak_to_target_score"})
        
    return peak_to_tg_score

def plot_subscore_histogram(merged_peaks, fig_dir):
    logging.info("Plotting subscore histograms")
    # Initialize a 3x3 charts
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

    # Flatten the axes array (makes it easier to iterate over)
    axes = axes.flatten()

    # Loop through each column and plot a histogram
    selected_cols = ["TF_mean_expression", "TG_mean_expression", "tf_to_peak_binding_score", "peak_to_target_score"]
    for i, column in enumerate(selected_cols):
        
        # Add the histogram
        merged_peaks[column].hist(ax=axes[i], # Define the current ax
                        color='cornflowerblue', # Color of the bins
                        bins=25, # Number of bins
                        grid=False
                    )
        
        # Add title and axis label
        axes[i].set_title(f'{column} distribution') 
        axes[i].set_xlabel(column) 
        axes[i].set_ylabel('Frequency') 
        axes[i].set_xlim((0,1)) # Set the xlim between 0-1 as the data is normalized

    plt.tight_layout()
    plt.savefig(f'{fig_dir}/merged_peaks.png', dpi=500)

def plot_tf_tg_expression_to_score_scatterplot(merged_peaks, fig_dir):
    logging.info("Plotting TF and TG expression to score scatterplot")
    plt.figure(figsize=(8, 10))
    plt.scatter(x=merged_peaks["TF_mean_expression"], y=merged_peaks["TG_mean_expression"], c=merged_peaks["Score"], cmap="coolwarm")
    plt.title("Relationship between TF and TG expression", fontsize=18)
    plt.xlabel("TF Expression", fontsize=16)
    plt.ylabel("TG Expression", fontsize=16)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/tf_vs_tg_expression_score_scatter.png', dpi=500)

def plot_population_score_histogram(merged_peaks, fig_dir):
    merged_peaks = merged_peaks[["Source", "Target", "Score"]]
    # logging.info(merged_peaks.head())
    # logging.info(merged_peaks.shape)

    logging.info("Plotting the final TF-TG score histogram")
    plt.figure(figsize=(8, 10))
    plt.hist(np.log2(merged_peaks["Score"]), bins=25)
    plt.title("TF-TG binding score", fontsize=18)
    plt.xlabel("Score", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/tf_to_tg_binding_score_hist.png', dpi=500)
    plt.close()
    
def process_cell(cell, rna_data_file, atac_data_file, tf_to_peak_score, peak_to_tg_score, output_dir, cell_level_net_dir):
    logging.info(f"Processing cell {cell}")
    # Load the raw RNA and ATAC datasets for the current process
    rna_data = load_rna_dataset(rna_data_file).reset_index()
    atac_data = load_atac_dataset(atac_data_file).reset_index()
    
    # Extract only the column for the current cell from RNA data
    if cell not in rna_data.columns:
        raise ValueError(f"Cell {cell} not found in RNA data.")
    rna_cell_df = rna_data[['gene', cell]].copy()
    rna_cell_df = rna_cell_df.rename(columns={cell: "rna_expression"})
    rna_cell_df["cell"] = cell
    rna_cell_df = rna_cell_df.astype({"rna_expression": "float16"})
    
    # Extract only the column for the current cell from ATAC data
    if cell not in atac_data.columns:
        raise ValueError(f"Cell {cell} not found in ATAC data.")
    atac_cell_df = atac_data[['peak', cell]].copy()
    atac_cell_df = atac_cell_df.rename(columns={cell: "atac_expression"})
    atac_cell_df["cell"] = cell
    atac_cell_df = atac_cell_df.astype({"atac_expression": "float16"})

    # Merge for TF scores and rename columns accordingly
    tf_to_peak_score_and_expr = pd.merge(tf_to_peak_score, rna_cell_df, on=["gene"], how="inner")
    tf_to_peak_score_and_expr = tf_to_peak_score_and_expr.rename(
        columns={"rna_expression": "TF_expression", "gene": "Source"}
    )

    # Merge for target gene scores and rename columns accordingly
    peak_to_tg_score_and_expr = pd.merge(peak_to_tg_score, rna_cell_df, on=["gene"], how="inner")
    peak_to_tg_score_and_expr = peak_to_tg_score_and_expr.rename(
        columns={"rna_expression": "TG_expression", "gene": "Target"}
    )

    # Merge the two score datasets on "cell" and "peak"
    merged_peaks = pd.merge(tf_to_peak_score_and_expr, peak_to_tg_score_and_expr, on=["cell", "peak"], how="inner")
    
    # Merge in the scATAC-seq peak expression for the cell
    inferred_network_raw = pd.merge(merged_peaks, atac_cell_df, on="peak", how="inner")

    # Normalize scores
    inferred_network_raw["tf_to_peak_binding_score"] = minmax_normalize_column(inferred_network_raw["tf_to_peak_binding_score"])
    inferred_network_raw["peak_to_target_score"] = minmax_normalize_column(inferred_network_raw["peak_to_target_score"])

    # print("inferred network raw")
    # print(inferred_network_raw.head())
    # print(inferred_network_raw.columns)  # Fixed: removed parentheses

    # Save the result as a pickle file
    output_path = os.path.join(output_dir, cell_level_net_dir, f"{cell}.pkl")
    inferred_network_raw.to_pickle(output_path)
    
    # Delete the DataFrame and run garbage collection
    del inferred_network_raw
    
    gc.collect()
    print(f"Finished processing cell: {cell}")

def calculate_cell_level_grn_parallel(rna_data_file, atac_data_file, tf_to_peak_score, peak_to_tg_score, output_dir, cell_level_net_dir, num_cpu, num_cells):
    # Load a list of cells you want to process from the raw RNA file header
    rna_data = load_rna_dataset(rna_data_file).reset_index()
    cells = [col for col in rna_data.columns if col != "gene"]
    
    # Process only a subset of cells
    cells_to_process = cells[:num_cells]
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(f'{output_dir}/cell_level_net_dir'):
        os.makedirs(f'{output_dir}/cell_level_net_dir')
    
    # Use ProcessPoolExecutor to process each cell in parallel, skipping cells with output already present.
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpu) as executor:
        futures = [
            executor.submit(process_cell, cell, rna_data_file, atac_data_file, tf_to_peak_score, peak_to_tg_score, output_dir, cell_level_net_dir)
            for cell in cells_to_process if f"{cell}.pkl" not in os.listdir(cell_level_net_dir)
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"A cell processing task generated an exception: {exc}")
    logging.info(f'Finished calculating cell-level dataframes')

def main():
    # Parse arguments
    args: argparse.Namespace = parse_args()

    rna_data_file: str = args.rna_data_file
    atac_data_file: str = args.atac_data_file
    output_dir: str = args.output_dir
    fig_dir: str = args.fig_dir
    bulk_or_cell: str = args.bulk_or_cell
    
    if args.num_cpu:
        num_cpu: int = int(args.num_cpu)
    
    if args.num_cells:
        num_cells: int = int(args.num_cells)
        
    if args.cell_level_net_dir:
        cell_level_net_dir: str = args.cell_level_net_dir
    
    
    # # Alternatively: Pass in specific file paths
    # rna_data_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/mESC/filtered_L2_E7.5_rep1/mESC_filtered_L2_E7.5_rep1_RNA.csv"
    # output_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/mESC/filtered_L2_E7.5_rep1"
    # num_cpu = 4
    # num_cells=50
    
    tf_to_peak_score_file = f'{output_dir}/tf_to_peak_binding_score.tsv'
    peak_to_tg_score_file = f'{output_dir}/peak_to_tg_scores.csv'

    # Load in and format the required files

    tf_to_peak_score = load_tf_to_peak_scores(tf_to_peak_score_file)
    peak_to_tg_score = load_peak_to_tg_scores(peak_to_tg_score_file)


    def calculate_population_grn(
        rna_data_file: pd.DataFrame,
        atac_data_file: pd.DataFrame,
        tf_to_peak_score: pd.DataFrame,
        peak_to_tg_score: pd.DataFrame
        ):
        
        rna_data = load_rna_dataset(rna_data_file)
        atac_data = load_atac_dataset(atac_data_file)
        
        # Calculate the normalized mean gene expression
        rna_data["rna_expression"] = minmax_normalize_column(rna_data.values.mean(axis=1))
        atac_data["atac_expression"] = atac_data.values.mean(axis=1)
        
        rna_data = rna_data.reset_index()
        rna_data = rna_data[["gene", "rna_expression"]].dropna()
        
        atac_data = atac_data.reset_index()
        atac_data = atac_data[["peak", "atac_expression"]]

        logging.info("Combining TF to peak binding scores with TF expression")
        tf_to_peak_score_and_expr = pd.merge(tf_to_peak_score, rna_data, on="gene", how="inner")

        tf_to_peak_score_and_expr = tf_to_peak_score_and_expr.rename(
            columns={
                "rna_expression": "TF_expression",
                "gene": "Source",
                }
            )

        logging.info("Combining peak to TG scores with TG expression")
        peak_to_tg_score_and_expr = pd.merge(peak_to_tg_score, rna_data, on="gene", how="inner")
        
        peak_to_tg_score_and_expr = peak_to_tg_score_and_expr.rename(
            columns={
                "rna_expression": "TG_expression",
                "gene": "Target",
                }
            )

        logging.info("Calculating final TF to TG score")
        merged_peaks = pd.merge(tf_to_peak_score_and_expr, peak_to_tg_score_and_expr, on=["peak"], how="inner")
        
        # print(merged_peaks["peak"][0:10])
        # print(atac_data["peak"][0:10])
        
        inferred_network_raw = pd.merge(merged_peaks, atac_data, on="peak", how="inner")
        print("Inferred network with ATACseq expression")
        print(inferred_network_raw.head())

        inferred_network_raw = inferred_network_raw.drop_duplicates()
        logging.info(f'Inferred network raw with dropped duplicate rows')
        logging.info(inferred_network_raw.head())
        logging.info(f'Columns: {inferred_network_raw.columns}')
        
        logging.info("Writing raw inferred network scores to output directory")
        inferred_network_raw.to_pickle(f'{output_dir}/inferred_network_raw.pkl')
        logging.info("Done!")
        
    if bulk_or_cell == "bulk":
        logging.info("Calculating bulk inferred grn")
        calculate_population_grn(rna_data_file, atac_data_file, tf_to_peak_score, peak_to_tg_score)
    
    elif bulk_or_cell == "cell":
        assert num_cpu != None
        assert num_cells != None
        assert cell_level_net_dir != None
        
        logging.info("Calculating cell-level score dataframes in parallel")
        calculate_cell_level_grn_parallel(rna_data_file, atac_data_file, tf_to_peak_score, peak_to_tg_score, output_dir, cell_level_net_dir, num_cpu, num_cells)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    main()