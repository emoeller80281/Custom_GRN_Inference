# Single-cell packages
import argparse
import matplotlib.pyplot as plt
import muon as mu
import numpy as np
import anndata as ad
import pysam
import scipy.sparse as sp
from anndata import AnnData
import networkx as nx

from pathlib import Path

# General helpful packages for data analysis and visualization
import pandas as pd
import scanpy as sc
import seaborn as sns
from muon import atac as ac  # the module containing function for scATAC data processing
import matplotlib.pyplot as plt

# Setting figure parameters
sc.settings.verbosity = 0

def parse_args():
    parser = argparse.ArgumentParser(description="Run Muon preprocessing with configurable file paths.")
    parser.add_argument("--project-dir", type=str, required=True)
    parser.add_argument("--tss-path", type=str, required=True)
    parser.add_argument("--raw-data-dir", type=str, required=True)
    parser.add_argument("--processed-data-dir", type=str, required=True)
    parser.add_argument("--sample-name", type=str, required=True)
    parser.add_argument("--rna-count-file", type=str, default=None)
    parser.add_argument("--atac-count-file", type=str, default=None)
    parser.add_argument("--raw-h5-file", type=str, default=None)
    parser.add_argument("--tf-list-file", type=str, required=True)
    parser.add_argument("--frag-path", type=str, required=True)
    return parser.parse_args()

def normalize_peak_format(peak_id: str) -> str:
    """
    Normalize peak format from chrN-start-end or chrN:start:end to chrN:start-end.
    Handles both formats as input and always outputs chrN:start-end.
    """
    if not isinstance(peak_id, str):
        return peak_id
    
    # Try to parse chr-start-end format (with dashes)
    parts = peak_id.split('-')
    if len(parts) >= 3:
        # Assume format is chr-start-end where chr might have dashes
        # Work backwards: the last two parts are start and end
        try:
            end = int(parts[-1])
            start = int(parts[-2])
            chrom = '-'.join(parts[:-2])  # Everything before the last two parts
            return f"{chrom}:{start}-{end}"
        except (ValueError, IndexError):
            pass
    
    # Already in chr:start-end format or some other format, return as-is
    return peak_id

def construct_from_gene_by_cell_matrices(rna_count_file, atac_count_file):
    assert rna_count_file.exists(), "rna count file does not exist"
    assert atac_count_file.exists(), "atac count file does not exist"
    
    rna_count_matrix = pd.read_csv(rna_count_file, header=0, index_col=0)
    atac_count_matrix = pd.read_csv(atac_count_file, header=0, index_col=0)
    
    rna_matrix = rna_count_matrix.T.values
    rna_metadata_df = pd.DataFrame(index=rna_count_matrix.columns)
    rna_features_df = pd.DataFrame(index=rna_count_matrix.index)
    
    atac_count_matrix.index = atac_count_matrix.index.map(normalize_peak_format)

    atac_matrix = atac_count_matrix.T.values
    atac_metadata_df = pd.DataFrame(index=atac_count_matrix.columns)
    atac_features_df = pd.DataFrame(index=atac_count_matrix.index)
    
    adata_rna = ad.AnnData(X=rna_matrix, obs=rna_metadata_df, var=rna_features_df)
    adata_rna.var["feature_types"] = "Gene Expression"
    adata_rna.var["gene_ids"] = adata_rna.var_names

    adata_atac = ad.AnnData(X=atac_matrix, obs=atac_metadata_df, var=atac_features_df)
    adata_atac.var["feature_types"] = "Peaks"
    adata_atac.var["gene_ids"] = adata_atac.var_names

    mdata = mu.MuData({'rna': adata_rna, 'atac': adata_atac})

    return mdata

def filter_to_human(mdata):
    """
    Filter a barnyard MuData object to hg38 ATAC peaks only,
    then strip the 'hg38.' prefix from peak IDs.
    """
    if "hg38" in mdata["atac"].var_names[0]:
        # annotate species from interval
        mdata["atac"].var["species"] = (
            mdata["atac"].var["interval"].str.split(".", n=1).str[0]
        )

        # keep only hg38 peaks
        hg38_mask = mdata["atac"].var["species"] == "hg38"
        mdata.mod["atac"] = mdata["atac"][:, hg38_mask].copy()

        # strip prefix from identifiers
        mdata.mod["atac"].var_names = (
            mdata.mod["atac"].var_names.str.replace(r"^hg38\.", "", regex=True)
        )
        mdata.mod["atac"].var["gene_ids"] = (
            mdata.mod["atac"].var["gene_ids"].str.replace(r"^hg38\.", "", regex=True)
        )
        mdata.mod["atac"].var["interval"] = (
            mdata.mod["atac"].var["interval"].str.replace(r"^hg38\.", "", regex=True)
        )
        mdata = mu.MuData(mdata.mod)

    return mdata

def flag_tfs_to_keep(rna, tf_list_file):
    """
    Flags TF genes to keep based on a provided list of TFs. If the list is not provided or the file does not exist, flags all genes as False.
    """
    rna.var['keep_tf'] = False
    
    if tf_list_file is not None and tf_list_file.exists():
        tf_list_df = pd.read_csv(tf_list_file)

        tf_genes_to_keep = set(
            tf_list_df["source_id"].astype(str).str.strip().str.upper()
        )

        rna.var["keep_tf"] = (
            rna.var_names.astype(str).str.strip().str.upper().isin(tf_genes_to_keep)
        )

def process_rna(mdata):
    rna = mdata.mod['rna']
    
class MudataProcessor:
    def __init__(
        self, 
        mdata, 
        processed_data_dir,
        sample_name,
        tf_list_file=None,
        ):
        
        self.mdata = mdata
        self.rna = mdata.mod['rna']
        self.atac = mdata.mod['atac']
        self.tf_list_file = tf_list_file
        self.processed_data_dir = processed_data_dir
        self.sample_name = sample_name

    def flag_tfs_to_keep(self):
        self.rna.var['keep_tf'] = False
        
        if self.tf_list_file is not None and self.tf_list_file.exists():
            tf_list_df = pd.read_csv(self.tf_list_file)

            tf_genes_to_keep = set(
                tf_list_df["source_id"].astype(str).str.strip().str.upper()
            )

            self.rna.var["keep_tf"] = (
                self.rna.var_names.astype(str).str.strip().str.upper().isin(tf_genes_to_keep)
            )
    
    def rna_qc_filter(
        self,
        min_cells_per_gene: int = 20,
        min_genes_per_cell: int = 500,
        max_genes_per_cell: int = 2500,
        min_total_counts_per_cell: int = 1000,
        max_total_counts_per_cell: int = 5000,
        max_pct_counts_mt: int = 20,
        norm_target_sum: float = 1e4,
        min_rna_disp: float = 0.5,
        filter_hvgs: bool = True,
        tf_list_file: str|None = None,
        fig_dir: Path|None = None,
        
    ):
        """
        Filter RNA data based on quality control criteria.

        Parameters
        ----------
        min_cells_per_gene : int, optional
            Minimum number of cells a gene must be present in to be kept.
            Defaults to 20.
        min_genes_per_cell : int, optional
            Minimum number of genes a cell must have to be kept.
            Defaults to 500.
        max_genes_per_cell : int, optional
            Maximum number of genes a cell can have to be kept.
            Defaults to 2500.
        min_total_counts_per_cell : int, optional
            Minimum total number of counts a cell must have to be kept.
            Defaults to 1000.
        max_total_counts_per_cell : int, optional
            Maximum total number of counts a cell can have to be kept.
            Defaults to 5000.
        max_pct_counts_mt : int, optional
            Maximum percentage of counts a cell can have in mitochondrial genes to be kept.
            Defaults to 20.

        Returns
        -------
        Nothing. The function modifies the RNA data in-place.
        """
        
        self.rna.var['mt'] = self.rna.var_names.str.upper().str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
        sc.pp.calculate_qc_metrics(self.rna, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        
        if fig_dir is not None and fig_dir.exists():
            sc.pl.violin(self.rna, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True)
            plt.savefig(fig_dir / "qc_violin_plots_pre_filtering.png")
            plt.close()

        # Filter
        if tf_list_file is not None and tf_list_file.exists():
            self.flag_tfs_to_keep()
        else:
            mu.pp.filter_var(self.rna, 'n_cells_by_counts', lambda x: (x >= min_cells_per_gene) | self.rna.var['keep_tf'].to_numpy(dtype=bool))

            
        mu.pp.filter_obs(self.rna, 'n_genes_by_counts', lambda x: (x >= min_genes_per_cell) & (x <= max_genes_per_cell))
        mu.pp.filter_obs(self.rna, 'total_counts', lambda x: (x >= min_total_counts_per_cell) & (x <= max_total_counts_per_cell))
        mu.pp.filter_obs(self.rna, 'pct_counts_mt', lambda x: x <= max_pct_counts_mt)
        
        if fig_dir is not None and fig_dir.exists():
            sc.pl.violin(self.rna, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True)
            plt.savefig(fig_dir / "qc_violin_plots_post_filtering.png")
            plt.close()
    
        # Normalize and log-transform
        sc.pp.normalize_total(self.rna, target_sum=norm_target_sum)
        sc.pp.log1p(self.rna)
        
        # Select highly variable genes
        sc.pp.highly_variable_genes(self.rna, min_mean=0.02, max_mean=4, min_disp=min_rna_disp)
        
        if fig_dir is not None and fig_dir.exists():
            sc.pl.highly_variable_genes(self.rna)
            plt.savefig(fig_dir / "highly_variable_genes.png")
            plt.close()
            
        if filter_hvgs:
            keep_genes = self.rna.var['highly_variable'] | self.rna.var['keep_tf']
            self.rna = self.rna[:, keep_genes]
            
        # Scaling
        self.rna.raw = self.rna
        sc.pp.scale(self.rna, max_value=10)
    
    def rna_pca_and_neighbors(self, rna, n_pcs=20, n_neighbors=10, fig_dir=None):
        """
        Perform principal component analysis (PCA) and k-nearest neighbors (kNN) on the RNA data.

        Parameters
        ----------
        rna : ad.AnnData
            The RNA data.
        n_pcs : int, optional
            The number of principal components to keep.
            Defaults to 20.
        n_neighbors : int, optional
            The number of k-nearest neighbors to keep.
            Defaults to 10.
        fig_dir : Path|None, optional
            The directory to save the figures in.
            Defaults to None.

        Returns
        -------
        Nothing. The function modifies the RNA data in-place.
        """
        sc.tl.pca(rna, svd_solver='arpack')

        if fig_dir is not None and fig_dir.exists():
            if "highly_variable" in rna.var.columns:
                first_three_hvg_genes = rna.var[rna.var.highly_variable].index[:3].to_list()
                sc.pl.pca(rna, color=first_three_hvg_genes)
                plt.savefig(fig_dir / "pca_hvgs.png")
                plt.close()
                
            sc.pl.pca_variance_ratio(rna, log=True)
            plt.savefig(fig_dir / "pca_variance_ratio.png")
            plt.close()
            
        sc.pp.neighbors(rna, n_neighbors=n_neighbors, n_pcs=n_pcs)
        
        sc.tl.umap(rna, spread=1., min_dist=.5, random_state=11)
        sc.tl.leiden(rna, flavor="igraph", n_iterations=2)
        
        if fig_dir is not None and fig_dir.exists():
            sc.pl.umap(rna, color=["leiden"])
            plt.savefig(fig_dir / "umap_leiden.png")
            plt.close()
            
    def atac_qc_filter(
        self, 
        min_cells_per_peak=20, 
        min_peaks_per_cell=500, 
        max_peaks_per_cell=2500, 
        min_total_counts_per_cell=1000, 
        max_total_counts_per_cell=5000
        ):
        ac.pp.calculate_qc_metrics(self.atac, inplace=True)
        
        ac.pp.filter_var(self.atac, 'n_cells_by_counts', lambda x: x >= min_cells_per_peak)
        ac.pp.filter_obs(self.atac, 'n_peaks_by_counts', lambda x: (x >= min_peaks_per_cell) & (x <= max_peaks_per_cell))
        ac.pp.filter_obs(self.atac, 'total_counts', lambda x: (x >= min_total_counts_per_cell) & (x <= max_total_counts_per_cell))
            
    def save_mdata(self):
        mu.write(self.sample_processed_data_dir / f"{self.sample_name}.h5mu", self.mdata)

    def save_rna(self):
        
        mu.write(self.sample_processed_data_dir / f"{self.sample_name}.h5mu/rna", self.rna)
        
    def save_atac(self):
        mu.write(self.sample_processed_data_dir / f"{self.sample_name}.h5mu/atac", self.atac)
        
        
    

if __name__ == "__main__":
    args = parse_args()

    PROJECT_DIR = args.project_dir
    RAW_DATA_DIR = args.raw_data_dir
    PROCESSED_DATA_DIR = args.processed_data_dir
    SAMPLE_NAME = args.sample_name

    tss_path = args.tss_path
    rna_count_file = args.rna_count_file
    atac_count_file = args.atac_count_file
    raw_h5_file = args.raw_h5_file
    tf_list_file = args.tf_list_file
    frag_path = args.frag_path

    SAMPLE_DATA_DIR = RAW_DATA_DIR / SAMPLE_NAME
    SAMPLE_PROCESSED_DATA_DIR = PROCESSED_DATA_DIR / SAMPLE_NAME

    if not SAMPLE_PROCESSED_DATA_DIR.exists():
        SAMPLE_PROCESSED_DATA_DIR.mkdir(parents=True)

    # Print all files in the sample data directory.
    print(f"Loading data for sample {SAMPLE_NAME} from {RAW_DATA_DIR}...")
    # print all files in the data directory
    for file in SAMPLE_DATA_DIR.glob("*"):
        print(f"  - {file.name}")
        file_end = file.name.split("_")[-1]
        print(file_end)
        if file_end.endswith("barcodes.tsv.gz"):
            file.rename(SAMPLE_DATA_DIR / f"barcodes.tsv.gz")
        elif file_end.endswith("features.tsv.gz"):
            file.rename(SAMPLE_DATA_DIR / f"features.tsv.gz")
        elif file_end.endswith("matrix.mtx.gz"):
            file.rename(SAMPLE_DATA_DIR / f"matrix.mtx.gz")
        elif file_end.endswith("fragments.tsv.gz"):
            file.rename(SAMPLE_DATA_DIR / f"fragments.tsv.gz")
        elif file_end.endswith("fragments.tsv.gz.tbi.gz"):
            file.rename(SAMPLE_DATA_DIR / f"fragments.tsv.gz.tbi.gz")
        
    if rna_count_file is not None and atac_count_file is not None:
        rna_count_filepath = SAMPLE_DATA_DIR / rna_count_file
        atac_count_filepath = SAMPLE_DATA_DIR / atac_count_file
        
        mdata = construct_from_gene_by_cell_matrices(rna_count_filepath, atac_count_filepath)
        print("Constructed MuData object from count files.")
    elif raw_h5_file is not None:
        mdata = mu.read_10x_h5(raw_h5_file)
    else:
        mdata = mu.read_10x_mtx(SAMPLE_DATA_DIR)
            
    mdata.var_names_make_unique()

    mdata.write(SAMPLE_PROCESSED_DATA_DIR / f"{SAMPLE_NAME}.h5mu")