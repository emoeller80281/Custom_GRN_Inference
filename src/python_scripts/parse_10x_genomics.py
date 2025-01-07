import pandas as pd
import scipy
import anndata
from scipy.sparse import  csc_matrix
import scanpy as sc
import numpy as np

# From Duren Lab's LINGER.preprocessing
def get_adata(matrix: csc_matrix, features: pd.DataFrame, barcodes: pd.DataFrame, label: pd.DataFrame):
    """
    Processes input RNA and ATAC-seq data to generate AnnData objects for RNA and ATAC data, 
    filters by quality, aligns by barcodes, and adds cell-type labels.

    Parameters:
        matrix (csc_matrix):
            A sparse matrix (CSC format) containing gene expression or ATAC-seq data where rows are features 
            (genes/peaks) and columns are cell barcodes.
        features (pd.DataFrame):
            A DataFrame containing information about the features. 
            Column 1 holds the feature names (e.g., gene IDs or peak names), and column 2 categorizes features 
            as "Gene Expression" or "Peaks".
        barcodes (pd.DataFrame):
            A DataFrame with one column of cell barcodes corresponding to the columns of the matrix.
        label (pd.DataFrame):
            A DataFrame containing cell-type annotations with the columns 'barcode_use' (for cell barcodes) 
            and 'label' (for cell types).

    Returns:
        tuple[AnnData, AnnData]:
            A tuple containing the filtered and processed AnnData objects for RNA and ATAC data.
            1. `adata_RNA`: The processed RNA-seq data.
            2. `adata_ATAC`: The processed ATAC-seq data.
    """



    # Ensure matrix data is in float32 format for memory efficiency and consistency
    matrix.data = matrix.data.astype(np.float32)

    # Create an AnnData object with the transposed matrix (cells as rows, features as columns)
    adata = anndata.AnnData(X=csc_matrix(matrix).T)
    print(adata.shape)
    
    # Assign feature names (e.g., gene IDs or peak names) to the variable (features) metadata in AnnData
    adata.var['gene_ids'] = features[1].values
    
    # Assign cell barcodes to the observation (cells) metadata in AnnData
    adata.obs['barcode'] = barcodes[0].values

    # Check if barcodes contain sample identifiers (suffix separated by '-'). If so, extract the sample number
    if len(barcodes[0].values[0].split("-")) == 2:
        adata.obs['sample'] = [int(string.split("-")[1]) for string in barcodes[0].values]
    else:
        # If no sample suffix, assign all cells to sample 1
        adata.obs['sample'] = 1

    # Subset features based on their type (Gene Expression or Peaks)
    # Select rows corresponding to "Gene Expression"
    rows_to_select: pd.Index = features[features[2] == 'Gene Expression'].index
    adata_RNA = adata[:, rows_to_select]

    # Select rows corresponding to "Peaks"
    rows_to_select = features[features[2] == 'Peaks'].index
    adata_ATAC = adata[:, rows_to_select]

    ### If cell-type label (annotation) is provided, filter and annotate AnnData objects based on the label

    # Filter RNA and ATAC data to keep only the barcodes present in the label
    idx: pd.Series = adata_RNA.obs['barcode'].isin(label['barcode_use'].values)
    adata_RNA = adata_RNA[idx]
    adata_ATAC = adata_ATAC[idx]

    # Set the index of the label DataFrame to the barcodes
    label.index = label['barcode_use']

    # Annotate cell types (labels) in the RNA data
    adata_RNA.obs['label'] = label.loc[adata_RNA.obs['barcode']]['label'].values

    # Annotate cell types (labels) in the ATAC data
    adata_ATAC.obs['label'] = label.loc[adata_ATAC.obs['barcode']]['label'].values

    ### Quality control filtering on the RNA data
    # Identify mitochondrial genes (which start with "MT-")
    adata_RNA.var["mt"] = adata_RNA.var_names.str.startswith("MT-")

    # Calculate QC metrics, including the percentage of mitochondrial gene counts
    sc.pp.calculate_qc_metrics(adata_RNA, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

    # Filter out cells with more than 5% of counts from mitochondrial genes
    adata_RNA = adata_RNA[adata_RNA.obs.pct_counts_mt < 5, :].copy()

    # Ensure that gene IDs are unique in the RNA data
    adata_RNA.var.index = adata_RNA.var['gene_ids'].values
    adata_RNA.var_names_make_unique()
    adata_RNA.var['gene_ids'] = adata_RNA.var.index

    ### Aligning RNA and ATAC data by barcodes (cells)
    # Identify barcodes present in both RNA and ATAC data
    selected_barcode: list = list(set(adata_RNA.obs['barcode'].values) & set(adata_ATAC.obs['barcode'].values))

    # Filter RNA data to keep only the barcodes present in both RNA and ATAC datasets
    barcode_idx: pd.DataFrame = pd.DataFrame(range(adata_RNA.shape[0]), index=adata_RNA.obs['barcode'].values)
    adata_RNA = adata_RNA[barcode_idx.loc[selected_barcode][0]]

    # Filter ATAC data to keep only the barcodes present in both RNA and ATAC datasets
    barcode_idx = pd.DataFrame(range(adata_ATAC.shape[0]), index=adata_ATAC.obs['barcode'].values)
    adata_ATAC = adata_ATAC[barcode_idx.loc[selected_barcode][0]]

    # Return the filtered and annotated RNA and ATAC AnnData objects
    return adata_RNA, adata_ATAC

print('\tReading in cell labels...')
# Read in the data files
matrix=scipy.io.mmread("/home/emoeller/github/Custom_GRN_Inference/input/filtered_feature_bc_matrix/matrix.mtx")
features=pd.read_csv("/home/emoeller/github/Custom_GRN_Inference/input/filtered_feature_bc_matrix/features.tsv",sep='\t',header=None)
barcodes=pd.read_csv("/home/emoeller/github/Custom_GRN_Inference/input/filtered_feature_bc_matrix/barcodes.tsv",sep='\t',header=None)
label=pd.read_csv("/home/emoeller/github/Custom_GRN_Inference/input/filtered_feature_bc_matrix/PBMC_label.txt",sep='\t',header=0)
# ---------------------------------------------------

print('\nExtracting the adata RNA and ATAC seq data...')
# Create AnnData objects for the scRNA-seq and scATAC-seq datasets
adata_RNA, adata_ATAC = get_adata(matrix, features, barcodes, label)  # adata_RNA and adata_ATAC are scRNA and scATAC

print(f'\tscRNAseq Dataset: {adata_RNA.shape[0]} genes, {adata_RNA.shape[1]} cells')
print(f'\tscATACseq Dataset: {adata_ATAC.shape[0]} peaks, {adata_ATAC.shape[1]} cells')

# Filter RNA data for 'classical monocytes'
rna_filtered = adata_RNA[adata_RNA.obs['label'] == 'classical monocytes']

# Extract the expression matrix (X) and convert it to a gene x cell DataFrame
RNA_expression_matrix = pd.DataFrame(
    data=rna_filtered.X.T.toarray(),  # Convert sparse matrix to dense
    index=rna_filtered.var['gene_ids'],    # Gene IDs as rows
    columns=rna_filtered.obs['barcode']  # Cell barcodes as columns
)
print(RNA_expression_matrix.head())

# Export the filtered RNA expression matrix to a CSV file
print(f'\tExporting filtered RNA data for classical monocytes')
RNA_output_file = "/home/emoeller/github/Custom_GRN_Inference/input/PBMC_RNA.csv"
RNA_expression_matrix.to_csv(RNA_output_file)

# Filter ATAC data for 'classical monocytes'
atac_filtered = adata_ATAC[adata_ATAC.obs['label'] == 'classical monocytes']

# Extract the expression matrix (X) and convert it to a gene x cell DataFrame
ATAC_expression_matrix = pd.DataFrame(
    data=atac_filtered.X.T.toarray(),  # Convert sparse matrix to dense
    index=atac_filtered.var['gene_ids'],    # Gene IDs as rows
    columns=atac_filtered.obs['barcode']   # Cell barcodes as columns
)
print(ATAC_expression_matrix.head())

# Export the filtered ATAC expression matrix to a CSV file
print(f'\tExporting filtered ATAC data for classical monocytes')
ATAC_output_file = "/home/emoeller/github/Custom_GRN_Inference/input/PBMC_ATAC.csv"
ATAC_expression_matrix.to_csv(ATAC_output_file)