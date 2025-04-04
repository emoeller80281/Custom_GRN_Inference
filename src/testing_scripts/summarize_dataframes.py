import pandas as pd
BASE_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/K562/K562_human_filtered/"

def write_summary(df, file_name):
    with open(f'{BASE_DIR}/{file_name}.txt', 'w') as f:
        df_string = df.iloc[:5, :].to_string(header=True, index=False)
        f.write(df_string)

        f.write('\n\n----- SUMMARY -----\n')
        if "source_id" in df.columns:
            f.write(f'Transcription Factors: {df["source_id"].nunique():,}\n')
        if "target_id" in df.columns:
            f.write(f'Target Genes: {df["target_id"].nunique():,}\n')
        if "peak_id" in df.columns:
            f.write(f'Peaks: {df["peak_id"].nunique():,}\n')
        f.write(f'Edges: {df.shape[0]:,}\n')
        f.write(f'DataFrame Columns: {df.shape[1]:,}')

def summarize_dataframe(file, sep, outfile_name):
    df = pd.read_csv(f'{BASE_DIR}/{file}', sep=sep, header=0)
    print(df.head())

    print('\n----- SUMMARY -----')
    if "source_id" in df.columns:
        print(f'Transcription Factors: {df["source_id"].nunique():,}')
    if "target_id" in df.columns:
        print(f'Target Genes: {df["target_id"].nunique():,}')
    if "peak_id" in df.columns:
        print(f'Peaks: {df["peak_id"].nunique():,}')
    print(f'Edges: {df.shape[0]:,}')
    print(f'DataFrame Columns: {df.shape[1]:,}\n')
    
    write_summary(df, outfile_name)

# Cicero
print(f'\n ===== Cicero =====')
summarize_dataframe("cicero_peak_to_tg_scores.csv", "\t", "cicero_output_summary")

# Peak to Gene Correlation
print(f'\n ===== Peak to Gene Correlation =====')
summarize_dataframe("peak_to_gene_correlation.csv", "\t", "peak_to_gene_correlation_summary")

# Sliding window TF to peak
print(f'\n ===== Sliding Window TF to Peak Score =====')
summarize_dataframe("sliding_window_tf_to_peak_score.tsv", "\t", "sliding_window_tf_to_peak_score_summary")

# Homer TF to peak
print(f'\n ===== Homer TF to Peak Score =====')
summarize_dataframe("homer_tf_to_peak.tsv", "\t", "homer_output_summary")

# Inferred Network (Full)
print(f'\n ===== Full Inferred Network =====')
summarize_dataframe("inferred_network_raw.csv", ",", "inferred_network_raw_summary")

# Inferred Network Aggregated Features (10% subset)
print(f'\n ===== Inferred Network (Aggregated Features, 10% Subset) =====')
summarize_dataframe("sample_inferred_network_raw_agg_features.csv", ",", "sample_inferred_network_agg_features_summary")

# Inferred Network All Features (10% subset)
print(f'\n ===== Inferred Network (All Features, 10% Subset) =====')
summarize_dataframe("sample_inferred_network_raw_all_features.csv", ",", "sample_inferred_network_all_features_summary")
