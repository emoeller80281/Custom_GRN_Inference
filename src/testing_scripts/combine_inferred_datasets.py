import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

output_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output"
print(output_dir)

inferred_net_paths = {}
feature_set_filename = "inferred_network_w_string.csv"

for cell_type in os.listdir(output_dir):
    cell_type_path = os.path.join(output_dir, cell_type)
    print(f'\n----- {cell_type.upper()} -----')
    
    
    for sample in os.listdir(cell_type_path):
        sample_path = os.path.join(cell_type_path, sample)
        print(f'{sample}')
        
        for folder in os.listdir(sample_path):
            if folder == "inferred_grns":
                folder_path = os.path.join(sample_path, folder)
                
                for inferred_net in os.listdir(folder_path):
                    inferred_net_path = os.path.join(folder_path, inferred_net)
                    if os.path.isfile(inferred_net_path):
                        print(f'    |____{inferred_net}')
                        if inferred_net == feature_set_filename:
                            if not cell_type in inferred_net_paths:
                                inferred_net_paths[cell_type] = {}
                                
                            if not sample in inferred_net_paths[cell_type]:
                                inferred_net_paths[cell_type][sample] = inferred_net_path


# print(f'\n===== Found {feature_set_filename} in: =====')             
# dataframes = {}                   
# ncols = len(inferred_net_paths.keys())
# nrows = max([len(sample_dict.keys()) for sample_dict in inferred_net_paths.values()])
# num_samples = sum([len(sample_dict.keys()) for sample_dict in inferred_net_paths.values()])

# print(f'Cell Types: {ncols}')
# print(f'Max num samples: {nrows}')
# print(f'Num samples: {num_samples}')

# plt.figure(figsize=(ncols*7,nrows*5))
# plt.suptitle("Number of rows with N non-NaN features")
# for i, (cell_type, sample_path_dict) in enumerate(inferred_net_paths.items()):
#     print(cell_type)
#     col = i + 1
#     print(f'Col: {col}')
    

#     for x, (sample_name, sample_grn_path) in enumerate(sample_path_dict.items()):
#         row = x + 1
#         print(f'Row: {row}')
#         print(f'  |__{sample_name}')
        
#         df = pd.read_csv(sample_grn_path, header=0, index_col=None, nrows=100000)
#         feature_counts = df.count(numeric_only=True, axis=1).value_counts().sort_index(ascending=False)
        
#         n_score_cols = len(df.select_dtypes(include=np.number).columns)
#         feature_threshold = n_score_cols - 2
#         df = df[df.count(numeric_only=True, axis=1) >= feature_threshold]
#         print(f'\tNumber of rows with >= {feature_threshold}/{n_score_cols} feature columns {len(df)}')
#         print(feature_counts.values)
#         # print(sorted(feature_counts.index.to_list()))      
# #         print(feature_counts.indices)
        
#         index = (row - 1) * ncols + col
#         plt.subplot(nrows, ncols, index)
#         plt.bar(feature_counts.index, feature_counts.values)
#         plt.title(f'{sample_name}')
#         plt.xlabel("Non-NaN Features")
#         plt.ylabel("Edges")
#         plt.xticks()
#         plt.tight_layout
# plt.show()

# How many cell types?
cell_types = list(inferred_net_paths.keys())
n_cells = len(cell_types)

# One row of subplots, one per cell_type
fig, axes = plt.subplots(
    nrows=1,
    ncols=n_cells,
    figsize=(6*n_cells, 5),
    squeeze=False  # ensures axes is 2D even if n_cells=1
)

for ax, cell_type in zip(axes[0], cell_types):
    sample_dict = inferred_net_paths[cell_type]
    
    # build a DataFrame of feature_counts for this cell_type
    all_counts = {}
    for sample_name, sample_path in sample_dict.items():
        df = pd.read_csv(sample_path)
        counts = (
            df
            .count(numeric_only=True, axis=1)
            .value_counts()
            .sort_index(ascending=False)
        )
        all_counts[sample_name] = counts
    
    feature_df = pd.DataFrame(all_counts).fillna(0)
    feature_df.index.name = "# non‑NaN features"
    feature_df.sort_index(ascending=True, inplace=True)
    
    # stacked bar chart
    feature_df.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        width=0.8,
    )
    # place legend below
    n_samples = len(sample_dict)
    ax.legend(
        feature_df.columns,
        title="Sample",
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=n_samples,
        fontsize='medium'
    )
    
    ax.set_title(f"{cell_type}", fontsize=14)
    ax.set_xlabel(feature_df.index.name, fontsize=14)
    ax.set_ylabel("Number of edges", fontsize=14)
    ax.tick_params(axis='x', rotation=0)
    ax.set_ylim((0, 200000))

fig.tight_layout()
plt.subplots_adjust(bottom=0.2)  # make room at bottom for legends
plt.suptitle("Number of non‑NaN feature columns", y=1.02, fontsize=18)
plt.show()

        
        
        
# def plot_column_histograms(df, fig_dir, df_name="inferred_net"):
#     # Create a figure and axes with a suitable size
#     plt.figure(figsize=(15, 8))
    
#     # Select only the numerical columns (those with numeric dtype)
#     cols = df.select_dtypes(include=[np.number]).columns

#     # Loop through each feature and create a subplot
#     for i, col in enumerate(cols, 1):
#         plt.subplot(3, 4, i)  # 3 rows, 4 columns, index = i
#         plt.hist(df[col], bins=50, alpha=0.7, edgecolor='black')
#         plt.title(f"{col} distribution")
#         plt.xlabel(col)
#         plt.ylabel("Frequency")

#     plt.tight_layout()
#     plt.savefig(f'{fig_dir}/{df_name}_column_histograms.png', dpi=300)
#     plt.close()
        
#         if cell_type not in dataframes.keys():
#             dataframes[cell_type] = []
            
#         dataframes[cell_type].append(df)
        
#     dataframes[cell_type] = pd.concat(dataframes[cell_type])
#     print()

# print(f'\n===== Combined DataFrames')
# for cell_type, combined_dataframe in dataframes.items():
#     print(cell_type)
#     print(combined_dataframe.head())
#     print()