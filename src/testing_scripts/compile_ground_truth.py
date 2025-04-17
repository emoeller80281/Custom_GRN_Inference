import pandas as pd

reference_net_dict = {
    # "RN002_mouse_ppi_string_nonspecific_path": "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN002_PPI_STRING_Mouse_Nonspecific.tsv",
    # "RN004_mouse_textmining_nonspecific_path": "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN004_TextMining_TRRUST_Mouse_Nonspecific.tsv",
    # "RN006_mouse_chipseq_nonspecific_path": "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN006_ChIPSeq_BEELINE_Mouse_Nonspecific.tsv",
    # "RN107_mouse_literature_CAD_path": "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN107_Literature_PMID20862356_Mouse_CAD.tsv",
    # "RN108_mouse_literature_IPSC_path": "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN108_Literature_PMID31067459_Mouse_IPSC.tsv",
    # "RN110_mouse_chipseq_DC_path": "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN110_ChIPSeq_BEELINE_Mouse_DC.tsv",
    "RN111_mouse_chipseq_ESC_path": "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/LINGER/LINGER_MESC_SC_DATA/RN111.tsv",
    "RN112_mouse_logof_ESC_path": "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN112_LOGOF_BEELINE_Mouse_ESC.tsv",
    # "RN113_mouse_chipseq_HSC_path": "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN113_ChIPSeq_BEELINE_Mouse_HSC.tsv",
    "RN114_mouse_chipx_escape_ESC_path": "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN114_ChIPX_ESCAPE_Mouse_ESC.tsv",
    "RN115_mouse_logof_escape_ESC_path": "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SC_MO_TRN_DB.MIRA/REPOSITORY/CURRENT/REFERENCE_NETWORKS/RN115_LOGOF_ESCAPE_Mouse_ESC.tsv",
}




# Initialize an empty list to store DataFrames
dfs = []

# Iterate over the paths in the reference_net_dict
for key, path in reference_net_dict.items():
    if key in ["RN111_mouse_chipseq_ESC_path", "RN112_mouse_logof_ESC_path", "RN006_mouse_chipseq_nonspecific_path"]:
        # Read the file
        with open(path, 'r') as f:
            lines = f.readlines()

        # Split each line based on tabs and further process the second column
        data = []
        for line in lines:
            # Split the line by tabs to isolate the first two fields
            parts = line.split('\t')

            source = parts[0].strip()  # First column
            target_relationship = parts[1].strip()  # Second column
            
            
            # Split the second column further by two spaces
            if ' ' in target_relationship:
                target, relationship = target_relationship.split('  ', 1)
            else:
                target = target_relationship
                relationship = None  # If there's no relationship column
                
            # Append the processed data
            data.append([source, target.strip()])

        # Create a DataFrame
        df = pd.DataFrame(data, columns=["Source", "Target"])
        print(f'Writing {key}')
        df.to_csv(f"/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/reference_networks/{key}.tsv", index=False, sep='\t', header=0)

    else:
        separator = '\t'
        # Load the file and select the first two columns
        df = pd.read_csv(path, sep=separator, usecols=[0, 1], header=0)
    
    # Rename columns
    df.columns = ["Source", "Target"]
    # print()
    # print(key)
    # print(df.head())
    # Append to the list
    dfs.append(df)

# Concatenate all DataFrames into one
merged_df = pd.concat(dfs, ignore_index=True)

merged_df["Source"] = merged_df["Source"].str.upper()
merged_df["Target"] = merged_df["Target"].str.upper()

# Display the first few rows
print(merged_df.head())
print(merged_df.shape)

# Optionally, save the merged DataFrame to a file
merged_df.to_csv("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/reference_networks/merged_reference_networks.tsv", index=False, sep='\t')