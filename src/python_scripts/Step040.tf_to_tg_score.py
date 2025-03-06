import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
import logging
import argparse
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
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for the sample"
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
        required=True,
        help="Number of processors to run multithreading with"
    )
    parser.add_argument(
        "--num_cells",
        type=str,
        required=True,
        help="Number of cells to generate cell-level grn score dataframes for"
    )
    
    
    args: argparse.Namespace = parser.parse_args()

    return args

def minmax_normalize_column(column: pd.DataFrame):
    return (column - column.min()) / (column.max() - column.min())

def load_rna_dataset(rna_data_file):
    # Read in the RNAseq data file and extract the gene names to find matching TFs
    logging.info("Reading and formatting expression data")
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
        
        # 
        # rna_cpm = minmax_normalize_column(rna_cpm)
        rna_cpm_df = pd.DataFrame(rna_cpm, index=RNA_dataset.index, columns=RNA_dataset.columns[1:])

        RNA_dataset.iloc[:, 1:] = rna_cpm_df

        # Transpose RNA dataset for easier access
        rna_data = RNA_dataset.set_index("gene")  # Rows = cells, Columns = genes

        return rna_data

    return calculate_rna_cpm(rna_data)

def load_tf_to_peak_scores(tf_to_peak_score_file):
    logging.info("Reading and formatting TF to peak binding scores")
    tf_to_peak_score = pd.read_csv(tf_to_peak_score_file, sep="\t", header=0, index_col=None).rename(columns={"binding_score":"tf_to_peak_binding_score"})
    tf_to_peak_score = tf_to_peak_score.melt(id_vars="peak", var_name="gene", value_name="tf_to_peak_binding_score")
    tf_to_peak_score["tf_to_peak_binding_score"] = minmax_normalize_column(tf_to_peak_score["tf_to_peak_binding_score"]) # Normalize binding score values
    
    return tf_to_peak_score

def load_peak_to_tg_scores(peak_to_tg_score_file):
    logging.info("Reading and formatting peak to TG scores")
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
    
def process_cell(cell, rna_data, tf_to_peak_score, peak_to_tg_score, output_dir):
    # Filter data for the specific cell
    logging.info(f"Processing cell {cell}")
    rna_data_cell = rna_data[rna_data["cell"] == cell]

    # Merge for TF scores and rename columns accordingly
    tf_to_peak_score_and_expr = pd.merge(tf_to_peak_score, rna_data_cell, on=["gene"], how="inner")
    tf_to_peak_score_and_expr = tf_to_peak_score_and_expr.rename(
        columns={"expression": "TF_expression", "gene": "Source"}
    )

    # Merge for target gene scores and rename columns accordingly
    peak_to_tg_score_and_expr = pd.merge(peak_to_tg_score, rna_data_cell, on=["gene"], how="inner")
    peak_to_tg_score_and_expr = peak_to_tg_score_and_expr.rename(
        columns={"expression": "TG_expression", "gene": "Target"}
    )

    # Merge the two score datasets on "cell" and "peak"
    merged_peaks = pd.merge(tf_to_peak_score_and_expr, peak_to_tg_score_and_expr, on=["cell", "peak"], how="inner")

    merged_peaks["prod"] = merged_peaks["tf_to_peak_binding_score"] * merged_peaks["peak_to_target_score"]
    score_df = merged_peaks.groupby(["Source", "Target"], as_index=False)["prod"].sum()
    score_df = score_df.rename(columns={"prod": "tf_to_tg_score"})

    # Merge back with the original merged_peaks to get additional information if needed
    inferred_network_raw = pd.merge(merged_peaks, score_df, how="right", on=["Source", "Target"])
    inferred_network_raw = inferred_network_raw.drop(columns=["tf_to_peak_binding_score", "peak_to_target_score"])

    # Normalize the TF to TG score to range [0, 1]
    inferred_network_raw["tf_to_tg_score"] = minmax_normalize_column(inferred_network_raw["tf_to_tg_score"])

    # Save the result as a pickle file
    output_path = os.path.join(output_dir, "cell_networks_raw", f"{cell}.pkl")
    inferred_network_raw.to_pickle(output_path)
    print(f"Finished processing cell: {cell}")

def calculate_cell_level_grn_parallel(rna_data, tf_to_peak_score, peak_to_tg_score, output_dir, num_cpu, num_cells):
    # Reset index and melt the rna_data into "gene", "cell", "expression" format
    rna_data = rna_data.reset_index()
    rna_data = pd.melt(rna_data, id_vars=["gene"], var_name="cell", value_name="expression")
    rna_data = rna_data.astype({"expression": "float16"})

    # Create the output directory if it doesn't exist
    cell_networks_dir = os.path.join(output_dir, "cell_networks_raw")
    if not os.path.exists(cell_networks_dir):
        os.makedirs(cell_networks_dir)

    # Get the unique cell names
    cells = rna_data["cell"].unique()

    # Use ProcessPoolExecutor to process each cell in parallel, skips cells with dataframes in the output_dir.
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpu) as executor:
        futures = [
            executor.submit(process_cell, cell, rna_data, tf_to_peak_score, peak_to_tg_score, output_dir)
            for cell in cells[0:num_cells] if cell not in os.listdir(cell_networks_dir)
        ]
        # Optionally, you can iterate over the futures to handle exceptions or log progress.
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"A cell processing task generated an exception: {exc}")

def main():
    # Parse arguments
    args: argparse.Namespace = parse_args()

    rna_data_file: str = args.rna_data_file
    output_dir: str = args.output_dir
    fig_dir: str = args.fig_dir
    num_cpu: int = int(args.num_cpu)
    num_cells: int = int(args.num_cells)
    
    # # Alternatively: Pass in specific file paths
    # rna_data_file = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/mESC/filtered_L2_E7.5_rep1/mESC_filtered_L2_E7.5_rep1_RNA.csv"
    # output_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/mESC/filtered_L2_E7.5_rep1"
    # num_cpu = 4
    # num_cells=50
    
    tf_to_peak_score_file = f'{output_dir}/tf_to_peak_binding_score.tsv'
    peak_to_tg_score_file = f'{output_dir}/peak_to_tg_scores.csv'

    # Load in and format the required files
    rna_data = load_rna_dataset(rna_data_file)
    tf_to_peak_score = load_tf_to_peak_scores(tf_to_peak_score_file)
    peak_to_tg_score = load_peak_to_tg_scores(peak_to_tg_score_file)

    logging.info("Calculating cell-level score dataframes in parallel")
    calculate_cell_level_grn_parallel(rna_data, tf_to_peak_score, peak_to_tg_score, output_dir, num_cpu, num_cells)

    def calculate_population_grn(rna_data, tf_to_peak_score, peak_to_tg_score):
        
        # Calculate the normalized mean gene expression
        rna_data["mean_expression"] = minmax_normalize_column(rna_data.values.mean(axis=1))
        # rna_data["std_expression"]    = minmax_normalize_column(rna_data.std(axis=1))
        # rna_data["min_expression"]    = minmax_normalize_column(rna_data.min(axis=1))
        # rna_data["max_expression"]   = minmax_normalize_column(rna_data.max(axis=1))
        # rna_data["median_expression"] = minmax_normalize_column(rna_data.median(axis=1))

        rna_data = rna_data.reset_index()
        # rna_data = rna_data[["gene", "mean_expression", "std_expression", "min_expression", "median_expression"]]
        rna_data = rna_data[["gene", "mean_expression"]]
        # logging.info(rna_data.head())

        logging.info("Combining TF to peak binding scores with TF expression")
        tf_to_peak_score_and_expr = pd.merge(tf_to_peak_score, rna_data, on="gene", how="inner")
        # logging.info(tf_to_peak_score_and_expr.head())

        tf_to_peak_score_and_expr = tf_to_peak_score_and_expr.rename(
            columns={
                "mean_expression": "TF_expression",
                "gene": "Source",
                # "std_expression": "TF_std_expression",
                # "min_expression": "TF_min_expression",
                # "median_expression": "TF_median_expression"
                }
            )
        # logging.info(tf_to_peak_score_and_expr.head())

        logging.info("Combining peak to TG scores with TG expression")
        peak_to_tg_score_and_expr = pd.merge(peak_to_tg_score, rna_data, on="gene", how="inner")
        
        peak_to_tg_score_and_expr = peak_to_tg_score_and_expr.rename(
            columns={
                "mean_expression": "TG_expression",
                "gene": "Target",
                # "std_expression": "TG_std_expression",
                # "min_expression": "TG_min_expression",
                # "median_expression": "TG_median_expression"
                }
            )

        logging.info("Calculating final TF to TG score")
        merged_peaks = pd.merge(tf_to_peak_score_and_expr, peak_to_tg_score_and_expr, on=["peak"], how="inner")
        
        # Calculate the TF to TG mean expression correlation, minmax values between 0-1
        # merged_peaks["pearson_correlation"] = merged_peaks["TF_mean_expression"].corr(merged_peaks["TG_mean_expression"], method="pearson")
        # logging.info(merged_peaks.columns)
        
        # Sums the product of all peak scores between each unique TF to TG pair
        score_df = merged_peaks.groupby(["Source", "Target"])
        score_df = score_df.apply(
            lambda x: (x["tf_to_peak_binding_score"] * x["peak_to_target_score"]).sum()
        ).reset_index(name="tf_to_tg_score")

        inferred_network_raw = pd.merge(merged_peaks, score_df, how="right", on=["Source", "Target"]).drop(columns=["tf_to_peak_binding_score", "peak_to_target_score"])
        # logging.info(inferred_network_raw.columns)
        
        # Normalize the TF to TG score between 0-1
        inferred_network_raw["tf_to_tg_score"] = minmax_normalize_column(inferred_network_raw["tf_to_tg_score"])
        
        # Calculate the final score, which is the TF expression * TF to TG interaction scores through thea peaks * TG expression
        inferred_network_raw["Score"] = inferred_network_raw["TF_expression"] * inferred_network_raw["tf_to_tg_score"] * inferred_network_raw["TG_expression"]

        inferred_network_raw = inferred_network_raw.drop(columns=["peak"]).drop_duplicates()
        logging.info(f'Inferred network raw with dropped duplicate rows')
        logging.info(inferred_network_raw.head())
        
        inferred_network = inferred_network_raw[["Source", "Target", "Score"]].drop_duplicates()
        
        logging.info("Writing inferred network to output directory")
        inferred_network.to_csv(f'{output_dir}/inferred_network.tsv', sep="\t", header=True, index=False)

        # # Convert all float columns to float32
        # float_cols = inferred_network_raw.select_dtypes(include=['float']).columns
        # inferred_network_raw[float_cols] = inferred_network_raw[float_cols].astype(np.float32)

        logging.info("Writing raw inferred network scores to output directory")
        inferred_network_raw.to_pickle(f'{output_dir}/inferred_network_raw.pkl')
        

    # calculate_population_grn(rna_data, tf_to_peak_score, peak_to_tg_score)


    
    
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    main()