import pandas as pd
import os
from tqdm import tqdm
import gc
import math

output_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/K562/K562_human_filtered"
raw_inferred_net_file = f'{output_dir}/full_network_feature_files/inferred_network_raw.csv'

def aggregate_scores_by_method_combo(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates DataFrame score columns into a single regulatory score by adding up
    each combination of the peak accessibility, peak to target gene, and transcription factor
    to peak scoring methods
    
    Sum of the following score combinations:
      - Mean peak accessibility * Cicero peak to TG * sliding window TF to peak
      - Mean peak accessibility * Cicero peak to TG * Homer TF to peak
      - Mean peak accessibility * peak to TG correlation * sliding window TF to peak
      - Mean peak accessibility * peak to TG correlation * Homer TF to peak

    Args:
        df (pd.DataFrame): DataFrame with all merged score columns

    Returns:
        agg_score_df (pd.DataFrame): DataFrame containing the TF, TG, expression, and regulatory score
    """
    print("\nAggregating scoring method combinations")
    
    
    peak_cicero_window = df["mean_peak_accessibility"] * df["cicero_score"] * df["sliding_window_score"]
    peak_cicero_homer = df["mean_peak_accessibility"] * df["cicero_score"] * df["homer_binding_score"]
    peak_corr_window = df["mean_peak_accessibility"] * df["correlation"] * df["sliding_window_score"]
    peak_corr_homer = df["mean_peak_accessibility"] * df["correlation"] * df["homer_binding_score"]
    
    agg_scores = peak_cicero_window + peak_cicero_homer + peak_corr_window + peak_corr_homer
    
    agg_score_df = pd.DataFrame({
        "source_id" : df["source_id"],
        "target_id" : df["target_id"],
        "Mean TF Expression" : df["mean_TF_expression"],
        "Mean TG Expression" : df["mean_TG_expression"],
        "Regulatory Score" : agg_scores
        })
    
    individual_scores = pd.DataFrame({
        "source_id" : df["source_id"],
        "target_id" : df["target_id"],
        "Mean TF Expression" : df["mean_TF_expression"],
        "Mean TG Expression" : df["mean_TG_expression"],
        "Cicero and Sliding Window" : peak_cicero_window,
        "Cicero and Homer" : peak_cicero_homer,
        "Correlation and Sliding Window" : peak_corr_window,
        "Correlation and Homer" : peak_corr_homer
        })
    
    return agg_score_df, individual_scores

def write_csv_in_chunks(df, output_dir, filename):
    print(f'Writing out CSV file to {filename} in 5% chunks')
    output_file = f'{output_dir}/{filename}'
    chunksize = int(math.ceil(0.05 * df.shape[0]))

    # Remove the output file if it already exists
    if os.path.exists(output_file):
        os.remove(output_file)

    # Write the DataFrame in chunks
    for start in tqdm(range(0, len(df), chunksize), unit="chunk"):
        chunk = df.iloc[start:start + chunksize]
        if start == 0:
            # For the first chunk, write with header in write mode
            chunk.to_csv(output_file, mode='w', header=True, index=False)
        else:
            # For subsequent chunks, append without header
            chunk.to_csv(output_file, mode='a', header=False, index=False)

# print("Reading in the raw inferred network")
# raw_inferred_df = pd.read_csv(raw_inferred_net_file, header=0)

# For testing, randomly downsample to 10% of the rows
# print("Creating and saving a 10% downsampling of the dataset for testing")
# sample_raw_inferred_df = raw_inferred_df.sample(frac=0.1)
# write_csv_in_chunks(sample_raw_inferred_df, output_dir, 'sampled_network_feature_files/sample_raw_inferred_df.csv')

# ===== AGGREGATE FEATURE SCORES BY COMBINING PERMUTATIONS OF FEATURE COMBINATIONS =====
# # Aggregating scores for the 10% downsampled DataFrame
# sample_agg_score_df, sample_each_combo_df = aggregate_scores_by_method_combo(sample_raw_inferred_df)
# write_csv_in_chunks(sample_agg_score_df, output_dir, 'sampled_network_feature_files/sample_inferred_network_agg_method_combo.csv')
# write_csv_in_chunks(sample_each_combo_df, output_dir, 'sampled_network_feature_files/sample_inferred_network_each_method_combo.csv')

# # Aggregating scores for the whole raw inferred DataFrame
# full_agg_score_df, full_each_combo_df = aggregate_scores_by_method_combo(raw_inferred_df)

# # Write out the CSV files in chunks
# write_csv_in_chunks(full_agg_score_df, output_dir, 'full_network_feature_files/full_inferred_network_agg_method_combo.csv')
# write_csv_in_chunks(full_each_combo_df, output_dir, 'full_network_feature_files/full_inferred_network_each_method_combo.csv')
# gc.collect()

# Subset to only have the STRING edges
print("Reading in the inferred network with STRING edge scores")
inferred_net_w_string_df = pd.read_csv(f'{output_dir}/full_network_feature_files/inferred_network_w_string.csv', header=0)
string_only_df = inferred_net_w_string_df[["source_id", "target_id", "string_experimental_score", "string_textmining_score", "string_combined_score"]].dropna(subset=["string_combined_score"])
write_csv_in_chunks(string_only_df, output_dir, 'full_network_feature_files/string_score_only.csv')
