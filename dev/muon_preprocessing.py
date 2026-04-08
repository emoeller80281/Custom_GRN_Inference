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
from matplotlib.axes import Axes

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

def load_raw_data(
    sample_name: str, 
    sample_data_dir: Path, 
    rna_count_file: Path | None = None, 
    atac_count_file: Path | None = None, 
    raw_h5_file: Path | None = None
    ):

    # Print all files in the sample data directory.
    print(f"Loading data for sample {sample_name} from {sample_data_dir}...")
    
    found_barcode = False
    found_features = False
    found_matrix = False
    
    frag_path = None
    
    # print all files in the data directory
    for file in sample_data_dir.glob("*"):
        file_name = file.name
        print(f"  - {file_name}")
        if file_name.endswith("barcodes.tsv.gz"):
            file.rename(sample_data_dir / f"barcodes.tsv.gz")
            found_barcode = True
        if file_name.endswith("features.tsv.gz"):
            file.rename(sample_data_dir / f"features.tsv.gz")
            found_features = True
        if file_name.endswith("matrix.mtx.gz"):
            file.rename(sample_data_dir / f"matrix.mtx.gz")
            found_matrix = True
        if file_name.endswith("fragments.tsv.gz"):
            file.rename(sample_data_dir / f"fragments.tsv.gz")
            frag_path = file
        if file_name.endswith("fragments.tsv.gz.tbi.gz"):
            file.rename(sample_data_dir / f"fragments.tsv.gz.tbi.gz")

    if not (found_barcode and found_features and found_matrix):
        for file in sample_data_dir.glob("*"):
            file_name = file.name
            if file_name.suffix == ".h5" or file_name.suffixes == [".h5", ".gz"]:
                print(f"Found h5 file: {file_name}")
                raw_h5_file = file
    
    # If raw count files are passed in, use them above any other file formats
    if rna_count_file is not None and atac_count_file is not None:
        rna_count_filepath = sample_data_dir / rna_count_file
        atac_count_filepath = sample_data_dir / atac_count_file
        
        mdata = construct_from_gene_by_cell_matrices(rna_count_filepath, atac_count_filepath)
        mdata.var_names_make_unique()
        return mdata, frag_path
        
    # If no count files are passed in, use the h5 file if it exists
    elif raw_h5_file is not None:
        mdata = mu.read_10x_h5(raw_h5_file)
        mdata.var_names_make_unique()
        return mdata, frag_path
        
    # If no h5 file is found, look for the 10x mtx files. If they exist, load them using muon.
    elif found_barcode and found_features and found_matrix:
        mdata = mu.read_10x_mtx(sample_data_dir)
        mdata.var_names_make_unique()
        return mdata, frag_path
    
    else:
        raise FileNotFoundError(f"Could not find the necessary files to load the data for sample {sample_name}. Please ensure that the sample directory contains either a raw h5 file or the 10x mtx files (barcodes.tsv.gz, features.tsv.gz, matrix.mtx.gz).")

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
    
class MudataProcessor:
    def __init__(
        self, 
        mdata, 
        processed_data_dir,
        sample_name,
        tss_path,
        tf_list_file=None,
        ):
        
        self.raw_mdata = mdata
        self.mdata = self.raw_mdata.copy()
        self.tss_path = tss_path
        self.rna = self.mdata.mod['rna']
        self.atac = self.mdata.mod['atac']
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
            
    def show_pre_filtering_qc_rna(self):
        self.rna.var['mt'] = self.rna.var_names.str.upper().str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
        sc.pp.calculate_qc_metrics(self.rna, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        
        sc.pl.violin(self.rna, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True)
        
    def show_pre_filtering_qc_atac(self):
        sc.pp.calculate_qc_metrics(self.atac, percent_top=None, log1p=False, inplace=True)
        
        sc.pl.violin(self.atac, ['n_genes_by_counts', 'total_counts'], jitter=0.4, multi_panel=True)
        
    
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
        tf_list_file: Path|None = None,
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
        
        if fig_dir is not None:
            fig_dir.mkdir(parents=True, exist_ok=True)
            sc.settings.figdir = fig_dir
        
        self.rna.var['mt'] = self.rna.var_names.str.upper().str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
        sc.pp.calculate_qc_metrics(self.rna, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        
        if fig_dir is not None and fig_dir.exists():
            sc.pl.violin(self.rna, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, save="_pre_qc_filtering_rna.png")

        # Filter
        if tf_list_file is not None and tf_list_file.exists():
            self.flag_tfs_to_keep()
            mu.pp.filter_var(self.rna, 'n_cells_by_counts', lambda x: (x >= min_cells_per_gene) | self.rna.var['keep_tf'].to_numpy(dtype=bool))
        else:
            mu.pp.filter_var(self.rna, 'n_cells_by_counts', lambda x: (x >= min_cells_per_gene))

            
        mu.pp.filter_obs(self.rna, 'n_genes_by_counts', lambda x: (x >= min_genes_per_cell) & (x <= max_genes_per_cell))
        mu.pp.filter_obs(self.rna, 'total_counts', lambda x: (x >= min_total_counts_per_cell) & (x <= max_total_counts_per_cell))
        mu.pp.filter_obs(self.rna, 'pct_counts_mt', lambda x: x <= max_pct_counts_mt)
        
        if fig_dir is not None and fig_dir.exists():
            sc.pl.violin(self.rna, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, save="_post_qc_filtering_rna.png")
            plt.close()
            
        # Save raw counts
        self.rna.layers["counts"] = self.rna.X.copy()
    
        # Normalize and log-transform
        sc.pp.normalize_total(self.rna, target_sum=norm_target_sum)
        sc.pp.log1p(self.rna)
        
        # Select highly variable genes
        sc.pp.highly_variable_genes(self.rna, min_mean=0.02, max_mean=4, min_disp=min_rna_disp)
        
        if fig_dir is not None and fig_dir.exists():
            sc.pl.highly_variable_genes(self.rna, save="_rna.png")
            
        if filter_hvgs:
            if tf_list_file is not None and tf_list_file.exists():
                keep_genes = self.rna.var['highly_variable'] | self.rna.var['keep_tf']
            else:
                keep_genes = self.rna.var['highly_variable']
            self.rna = self.rna[:, keep_genes]
            
        self.rna.layers["counts"] = self.rna.X.copy()
        
        # Scaling
        self.rna.raw = self.rna
        sc.pp.scale(self.rna, max_value=10)
        
        mu.write(str(self.processed_data_dir / f"{self.sample_name}.h5mu/rna"), self.rna)
    
    def rna_pca_and_neighbors(self, rna, n_pcs=20, n_neighbors=10, fig_dir: Path|None = None):
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
        if fig_dir is not None:
            fig_dir.mkdir(parents=True, exist_ok=True)
            sc.settings.figdir = fig_dir
        
        sc.tl.pca(rna, svd_solver='arpack')

        if fig_dir is not None and fig_dir.exists():
            if "highly_variable" in rna.var.columns:
                first_three_hvg_genes = rna.var[rna.var.highly_variable].index[:3].to_list()
                sc.pl.pca(rna, color=first_three_hvg_genes, save="_pca_hvgs.png")
                
            sc.pl.pca_variance_ratio(rna, log=True, save="_pca_variance_ratio_rna.png")
            
        sc.pp.neighbors(rna, n_neighbors=n_neighbors, n_pcs=n_pcs)
        
        sc.tl.umap(rna, spread=1., min_dist=.5, random_state=11)
        sc.tl.leiden(rna, flavor="igraph", n_iterations=2)
        
        if fig_dir is not None and fig_dir.exists():
            sc.pl.umap(rna, color=["leiden"], save="_umap_leiden_rna.png")
    
    def construct_peak_annotation(
        self, 
        save_dir: Path,
        promoter_upstream: int = 1000,
        promoter_downstream: int = 100,
        distal_max: int = 200_000
        ):
        """Construct a peak annotation table in 10x-style format, assigning each peak to a gene and distance based on TSS proximity."""

        peaks = pd.DataFrame({"peak": self.atac.var_names.astype(str)})
        
        coords = peaks["peak"].str.extract(
            r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$"
        )

        if coords.isna().any().any():
            bad = peaks.loc[coords.isna().any(axis=1), "peak"].head(10).tolist()
            raise ValueError(f"Could not parse some peak names. Examples: {bad}")

        peaks = pd.concat([peaks, coords], axis=1)
        peaks["start"] = peaks["start"].astype(int)
        peaks["end"] = peaks["end"].astype(int)

        tss = pd.read_csv(
            self.tss_path,
            sep="\t",
            header=None,
            names=["chrom", "tss_start", "tss_end", "gene"]
        )

        tss["tss"] = tss["tss_start"].astype(int)

        # Only keep protein coding genes
        if "gene_biotype" in tss.columns:
            tss = tss[tss["gene_biotype"] == "protein_coding"].copy()

        # Cross-join peaks and genes by chromosome
        cand = peaks.merge(
            tss[["chrom", "gene", "tss"]],
            on="chrom",
            how="inner"
        )

        # 10x signed distance:
        # positive if peak start is downstream of TSS
        # negative if peak end is upstream of TSS
        # zero if TSS overlaps peak
        cand["distance"] = np.where(
            cand["start"] > cand["tss"],
            cand["start"] - cand["tss"],
            np.where(
                cand["end"] < cand["tss"],
                cand["end"] - cand["tss"],
                0
            )
        )

        cand["abs_distance"] = cand["distance"].abs()

        # PROMOTER peaks:
        # overlap promoter region [TSS-1000, TSS+100]
        cand["is_promoter"] = (
            (cand["end"] >= (cand["tss"] - promoter_upstream)) &
            (cand["start"] <= (cand["tss"] + promoter_downstream))
        )

        promoter = cand.loc[cand["is_promoter"], ["peak", "chrom", "start", "end", "gene", "distance"]].copy()
        promoter["peak_type"] = "promoter"

        # DISTAL peaks:
        # within 200 kb of the CLOSEST TSS,
        # but not promoter for that same gene
        
        # find closest TSS gene per peak
        closest_idx = cand.groupby("peak")["abs_distance"].idxmin()
        closest = cand.loc[closest_idx, ["peak", "chrom", "start", "end", "gene", "distance", "abs_distance"]].copy()

        closest = closest.loc[closest["abs_distance"] <= distal_max].copy()

        # remove cases where that peak is already promoter for that same gene
        promoter_pairs = set(zip(promoter["peak"], promoter["gene"]))
        closest["is_promoter_same_gene"] = [
            (p, g) in promoter_pairs for p, g in zip(closest["peak"], closest["gene"])
        ]

        distal = closest.loc[~closest["is_promoter_same_gene"], ["peak", "chrom", "start", "end", "gene", "distance"]].copy()
        distal["peak_type"] = "distal"

        # INTERGENIC peaks:
        # peaks with no promoter or distal assignment
        assigned_peaks = set(promoter["peak"]) | set(distal["peak"])

        intergenic = peaks.loc[~peaks["peak"].isin(assigned_peaks), ["peak", "chrom", "start", "end"]].copy()
        intergenic["gene"] = ""
        intergenic["distance"] = np.nan
        intergenic["peak_type"] = "intergenic"

        # Final table in 10x format
        peak_annotation_10x = pd.concat(
            [
                promoter[["chrom", "start", "end", "gene", "distance", "peak_type"]],
                distal[["chrom", "start", "end", "gene", "distance", "peak_type"]],
                intergenic[["chrom", "start", "end", "gene", "distance", "peak_type"]],
            ],
            axis=0,
            ignore_index=True
        ).sort_values(["chrom", "start", "end", "gene", "peak_type"])
        
        peak_annotation_10x = peak_annotation_10x.dropna()

        # save as 10x-style TSV
        out_path = save_dir / "atac_peak_annotation.tsv"
        if not out_path.parent.exists():
            out_path.parent.mkdir(parents=True)
        peak_annotation_10x.to_csv(out_path, sep="\t", index=False)

    
    def atac_qc_filter(
        self, 
        min_cells_per_peak=20, 
        min_peaks_per_cell=500, 
        max_peaks_per_cell=2500, 
        min_total_counts_per_cell=1000, 
        max_total_counts_per_cell=5000,
        min_atac_disp=0.5,
        promoter_upstream=1000,
        promoter_downstream=100,
        distal_max=200_000,
        filter_hvgs: bool = True,
        fig_dir: Path|None = None
        ):
        if fig_dir is not None:
            sc.settings.figdir = fig_dir
            if not fig_dir.exists():
                fig_dir.mkdir(parents=True, exist_ok=True)
        
        sc.pp.calculate_qc_metrics(self.atac, percent_top=None, log1p=False, inplace=True)
        
        if fig_dir is not None and fig_dir.exists():
            sc.pl.violin(self.atac, ['n_genes_by_counts', 'total_counts'], jitter=0.4, multi_panel=True, save="_pre_qc_filtering_atac.png")                    
        
        mu.pp.filter_var(self.atac, 'n_cells_by_counts', lambda x: x >= min_cells_per_peak)
        mu.pp.filter_obs(self.atac, 'n_genes_by_counts', lambda x: (x >= min_peaks_per_cell) & (x <= max_peaks_per_cell))
        mu.pp.filter_obs(self.atac, 'total_counts', lambda x: (x >= min_total_counts_per_cell) & (x <= max_total_counts_per_cell))
        
        if fig_dir is not None and fig_dir.exists():
            sc.pl.violin(self.atac, ['n_genes_by_counts', 'total_counts'], jitter=0.4, multi_panel=True, save="_post_qc_filtering_atac.png")         
            mu.pl.histogram(self.atac, ['n_genes_by_counts', 'total_counts'], save="_qc_histograms_atac.png")
        

        # Save original counts
        self.atac.layers["counts"] = self.atac.X.copy()
        
        ac.pp.tfidf(self.atac, scale_factor=1e4)
        
        sc.pp.normalize_per_cell(self.atac, counts_per_cell_after=1e4)
        sc.pp.log1p(self.atac)
        
        sc.pp.highly_variable_genes(self.atac, min_mean=0.05, max_mean=1.5, min_disp=min_atac_disp)
        
        if fig_dir is not None and fig_dir.exists():
            sc.pl.highly_variable_genes(self.atac, save="_peaks.png")
            
        if filter_hvgs:
            keep_peaks = self.atac.var['highly_variable']
            self.atac = self.atac[:, keep_peaks]
            
        # Scaling
        self.atac.raw = self.atac
        
        # LSI
        ac.tl.lsi(self.atac)
        
        self.atac.obsm['X_lsi'] = self.atac.obsm['X_lsi'][:,1:]
        self.atac.varm["LSI"] = self.atac.varm["LSI"][:,1:]
        self.atac.uns["lsi"]["stdev"] = self.atac.uns["lsi"]["stdev"][1:]
        
        # Neighbors
        sc.pp.neighbors(self.atac, use_rep="X_lsi", n_neighbors=10, n_pcs=30)
        
        # PCA
        sc.pp.scale(self.atac)
        sc.tl.pca(self.atac)
        
        if fig_dir is not None and fig_dir.exists():
            sc.pl.pca(self.atac, color=["n_genes_by_counts", "n_counts"], save="_pca_qc_atac.png")
        
        # Annotate peaks as promoter/distal/intergenic based on TSS proximity
        # assign gene names and distances for promoter/distal peaks
        self.construct_peak_annotation(
            self.processed_data_dir / self.sample_name, 
            promoter_upstream=promoter_upstream, 
            promoter_downstream=promoter_downstream, 
            distal_max=distal_max
            )
        
        ac.tl.add_peak_annotation(self.atac, annotation=str(self.processed_data_dir / self.sample_name / "atac_peak_annotation.tsv"))
        
        # Neighbors
        sc.pp.neighbors(self.atac, n_neighbors=10, n_pcs=30)
        
        # Clustering
        sc.tl.umap(self.atac, spread=1., min_dist=.5, random_state=11)
        sc.tl.leiden(self.atac, flavor="igraph", n_iterations=2)
        
        if fig_dir is not None and fig_dir.exists():
            sc.pl.umap(self.atac, color=["leiden", "n_genes_by_counts"], legend_loc="on data", save="_umap_leiden_atac.png")
            
        mu.write(str(self.processed_data_dir / f"{self.sample_name}.h5mu/atac"), self.atac)
                    
    def nucleosome_signal(
        self,
        frag_path: Path,
        fig_dir: Path|None = None
    ):
        index_file = str(frag_path) + ".tbi"

        if Path(index_file).exists():
            print("Found index:", index_file)

        else:
            print("Index file not found. Creating index file...")
            pysam.tabix_index(
                str(frag_path),
                preset="bed",
                force=True
            )
            index_file = str(frag_path) + ".tbi"

        tbx = pysam.TabixFile(str(frag_path))
        print(tbx.contigs[:20])

        # register the fragment file with the ATAC AnnData
        ac.tl.locate_fragments(self.atac, fragments=str(frag_path))

        self.atac.obs["NS"] = 1

        def find_nonempty_region(tbx, chrom="chr1", window=1_000_000, max_end=200_000_000):
            for start in range(0, max_end, window):
                rows = list(tbx.fetch(chrom, start, start + window))
                if rows:
                    return f"{chrom}:{start}-{start+window}", rows[0]
            return None, None

        region, example = find_nonempty_region(tbx, chrom="chr1")

        if fig_dir is not None and fig_dir.exists():
            sc.settings.figdir = fig_dir
            ac.pl.fragment_histogram(self.atac, region=region, save="_fragment_histogram.png")
            
        ac.tl.nucleosome_signal(self.atac, n=1e6)
        
        if fig_dir is not None and fig_dir.exists():
            mu.pl.histogram(self.atac, "nucleosome_signal", kde=False, save="_nucleosome_signal.png")
            
    def tss_enrichment(
        self, 
        frag_path: Path, 
        n_tss: int = 500, 
        extend_upstream: int = 1000, 
        extend_downstream: int = 1000,
        fig_dir: Path|None = None
        ):
        if frag_path is not None and frag_path.exists():
            
            tss_df = pd.read_csv(
                self.tss_path,
                sep="\t",
                header=None,
                names=["tss_chrom", "tss_start", "tss_end", "tss_gene"]
            )

            var = self.rna.var.copy()
            var["original_var_name"] = var.index

            # Remove any old columns that could collide
            cols_to_drop = [
                "tss_chrom", "tss_start", "tss_end", "tss_gene",
                "Chromosome", "Start", "End", "interval"
            ]
            var = var.drop(columns=[c for c in cols_to_drop if c in var.columns], errors="ignore")

            # Merge gene annotations onto var
            var = (
                var.merge(
                    tss_df,
                    left_index=True,
                    right_on="tss_gene",
                    how="left"
                )
                .set_index("original_var_name")
            )

            var["interval"] = pd.NA
            mask = var["tss_chrom"].notna() & var["tss_start"].notna() & var["tss_end"].notna()

            var.loc[mask, "interval"] = (
                var.loc[mask, "tss_chrom"].astype(str) + ":" +
                var.loc[mask, "tss_start"].astype(int).astype(str) + "-" +
                var.loc[mask, "tss_end"].astype(int).astype(str)
            )

            var["Chromosome"] = var["tss_chrom"]
            var["Start"] = var["tss_start"]
            var["End"] = var["tss_end"]

            self.rna.var = var

            rna_tss = self.rna[:, self.rna.var["interval"].notna()].copy()
            
            genes = ac.tl.get_gene_annotation_from_rna(rna_tss)
            
            tss = ac.tl.tss_enrichment(
                self.mdata, 
                features=genes, 
                n_tss=n_tss, 
                extend_upstream=extend_upstream, 
                extend_downstream=extend_downstream, 
                random_state=11
                )
            
            if fig_dir is not None and fig_dir.exists():
                self.tss_enrichment_plot(tss, save=str(fig_dir / "tss_enrichment.png"))

    def tss_enrichment_plot(
        self,
        data: AnnData,
        color: str | None = None,
        title: str = "TSS Enrichment",
        ax: Axes | None = None,
        save: str = None
    ):
        """
        Plot relative enrichment scores around a TSS.

        Parameters
        ----------
        data
            AnnData object with cell x TSS_position matrix as generated by `muon.atac.tl.tss_enrichment`.
        color
            Column name of .obs slot of the AnnData object which to group TSS signals by.
        title
            Plot title.
        ax
            A matplotlib axes object.
        """
        ax = ax or plt.gca()

        if color is not None:
            if isinstance(color, str):
                color = [color]

            groups = data.obs.groupby(color)

            for name, group in groups:
                ad = data[group.index]
                ac.pl._tss_enrichment_single(ad, ax, label=name)
        else:
            ac.pl._tss_enrichment_single(data, ax)

        # TODO Not sure how to best deal with plot returning/showing
        ax.set_title(title)
        ax.set_xlabel("Distance from TSS, bp")
        ax.set_ylabel("Average TSS enrichment score")
        if color:
            ax.legend(loc="upper right", title=", ".join(color))
        if save:
            plt.savefig(save, dpi=150)
        
        plt.show()
        return None

    def save_mdata(self):
        mu.write(self.processed_data_dir / self.sample_name / f"{self.sample_name}.h5mu", self.mdata)

    def save_rna(self):
        
        mu.write(self.processed_data_dir / self.sample_name / f"{self.sample_name}.h5mu/rna", self.rna)
        
    def save_atac(self):
        mu.write(self.processed_data_dir / self.sample_name / f"{self.sample_name}.h5mu/atac", self.atac)
            
def integrate_rna_atac(
    mdata: ad.AnnData, 
    sample_processed_data_dir: Path, 
    sample_name: str,
    fig_dir: Path|None = None
    ):
    if fig_dir is not None:
        fig_dir.mkdir(parents=True, exist_ok=True)
        sc.settings.figdir = fig_dir
    
    # Restrict to cells passing QC in both modalities
    mu.pp.intersect_obs(mdata)
    
    # Perform MOFA+ integration
    mu.tl.mofa(mdata, outfile=sample_processed_data_dir / f"{sample_name}_rna_atac.h5mu")
    
    sc.pp.neighbors(mdata, use_rep="X_mofa")
    sc.tl.umap(mdata)
    sc.tl.umap(mdata, min_dist=.2, spread=1., random_state=10)
    sc.tl.leiden(mdata, flavor="igraph", n_iterations=2)

    if fig_dir is not None and fig_dir.exists():
        # Plot the UMAP colored by MOFA clusters
        sc.pl.umap(mdata, color=["leiden"], save="mofa_umap_leiden.png")
        
        # Plot the first 4 MOFA factors in pairwise scatter plots
        df = pd.DataFrame(mdata.obsm["X_mofa"])
        df.columns = [f"Factor {i+1}" for i in range(df.shape[1])]

        plot_scatter = lambda i, ax: sns.scatterplot(data=df, x=f"Factor {i+1}", y=f"Factor {i+2}", color="black", linewidth=0, s=3, ax=ax)

        fig, axes = plt.subplots(2, 2)
        for i in range(4):
            plot_scatter(i, axes[i%2][i//2])
            
        plt.tight_layout()
        plt.savefig(fig_dir / "mofa_factor_scatter.png", dpi=150)
        plt.close()
        
    # Ranking genes and peaks
    mdata["rna"].obs["leiden_joint"] = mdata.obs["leiden"]
    mdata["atac"].obs["leiden_joint"] = mdata.obs["leiden"]
    
    sc.tl.rank_genes_groups(mdata['rna'], 'leiden_joint', method='t-test_overestim_var')
    ac.tl.rank_peaks_groups(mdata['atac'], 'leiden_joint', method='t-test_overestim_var')
    
def save_processed_data(mdata: ad.AnnData, sample_processed_data_dir: Path):
    def _adata_to_feature_by_cell_df(adata: ad.AnnData) -> pd.DataFrame:
        """
        Convert an AnnData object from cell x feature to feature x cell DataFrame.
        Preference order:
        1. adata.layers["log1p"]
        2. adata.layers["counts"]
        3. adata.X
        """
        if "log1p" in adata.layers:
            X = adata.layers["log1p"]   
        elif "counts" in adata.layers:
            X = adata.layers["counts"]
        else:
            X = adata.X

        if sp.issparse(X):
            arr = X.T.toarray()
        else:
            arr = np.asarray(X, dtype=np.float32).T

        return pd.DataFrame(
            arr,
            index=adata.var_names.astype(str),
            columns=adata.obs_names.astype(str),
        )

    def standardize_name(name: str) -> str:
        """Convert gene/motif name to upper style."""
        if not isinstance(name, str):
            return name
        return name.upper()

    processed_rna_file = sample_processed_data_dir / "scRNA_seq_processed.parquet"
    processed_atac_file = sample_processed_data_dir / "scATAC_seq_processed.parquet"

    adata_rna_file = sample_processed_data_dir / "adata_RNA.h5ad"
    adata_atac_file = sample_processed_data_dir / "adata_ATAC.h5ad"
    mdata_file = sample_processed_data_dir / "multiome_processed.h5mu"

    # Pull modalities from MuData
    adata_rna = mdata["rna"]
    adata_atac = mdata["atac"]

    # Convert to feature x cell DataFrames
    processed_rna_df = _adata_to_feature_by_cell_df(adata_rna).astype("float32")
    processed_atac_df = _adata_to_feature_by_cell_df(adata_atac).astype("float32")

    # Standardize RNA gene names
    processed_rna_df.index = processed_rna_df.index.astype(str).map(standardize_name)

    # Save parquet outputs
    processed_rna_df.to_parquet(processed_rna_file, engine="pyarrow", compression="snappy")
    processed_atac_df.to_parquet(processed_atac_file, engine="pyarrow", compression="snappy")

    # Save modality-level AnnData files
    adata_rna.write_h5ad(adata_rna_file)
    adata_atac.write_h5ad(adata_atac_file)

    # Save the full MuData object 
    mdata.write(mdata_file) 
    
def create_metacells(
    mdata: ad.AnnData, 
    sample_processed_data_dir: Path, 
    hops: int = 2,
    ):
    """
    Create metacell-level profiles from RNA and ATAC data matrices.

    Parameters
    ----------
    mdata : ad.AnnData
        The MuData object containing RNA and ATAC data matrices.
    sample_processed_data_dir : Path
        The directory where the pseudobulk DataFrames will be saved.
    hops : int, default=2
        The number of hops between neighbors to consider when diffusing information.

    Returns
    -------
    None

    Notes
    -----
    This function creates metacell-level profiles by applying a diffusion operator to the RNA and ATAC data matrices.
    The diffusion operator is constructed by first extracting the neighbor graph from the MuData object and converting it to a row-normalized sparse matrix.
    The operator is then applied to the RNA and ATAC data matrices to obtain the metacell-level profiles.
    The resulting profiles are saved as parquet files in the specified directory.
    """
    # Extract the neighbor graph and convert to a row-normalized sparse matrix
    W = mdata.obsp["connectivities"].tocsr().astype(np.float32)
    
    # Add self-connections
    W = W + sp.diags(np.full(W.shape[0], 1, dtype=np.float32), format="csr")
    
    def row_norm(mat: sp.csr_matrix) -> sp.csr_matrix:
        row_sum = np.asarray(mat.sum(axis=1)).ravel()
        row_sum[row_sum == 0] = 1.0
        inv = sp.diags(1.0 / row_sum, dtype=np.float32)
        return inv @ mat

    W = row_norm(W)
    
    # Diffusion based on the number of hops between neighbors. 
    # Pools information from neighbors up to HOPS distance away, with more weight on closer neighbors.
    W_h = W
    for _ in range(1, int(hops)):
        W_h = W_h @ W 
        W_h = row_norm(W_h)
    W = W_h

    # Final row normalization to make sure rows sum to 1
    W = row_norm(W)
    
    # Apply the diffusion operator to the RNA and ATAC data matrices to get metacell-level profiles.
    X_rna = sp.csr_matrix(np.asarray(mdata["rna"].X, dtype=np.float32, order="C"))
    X_atac = sp.csr_matrix(np.asarray(mdata["atac"].X, dtype=np.float32, order="C"))

    X_rna_soft = W @ X_rna      # cells × genes
    X_atac_soft = W @ X_atac    # cells × peaks
    
    # Create and save the pseudobulk DataFrames
    def _standardize_symbols_index(
        df: pd.DataFrame,
        *,
        strip_version_suffix: bool = True,
        uppercase: bool = True,
        deduplicate: str = "sum",
    ) -> pd.DataFrame:
        x = df.copy()
        idx = x.index.astype(str).str.strip()
        if strip_version_suffix:
            idx = idx.str.replace(r"\.\d+$", "", regex=True)
        if uppercase:
            idx = idx.str.upper()
        x.index = idx
        if deduplicate:
            if deduplicate == "sum":
                x = x.groupby(level=0).sum()
            elif deduplicate == "mean":
                x = x.groupby(level=0).mean()
            elif deduplicate == "first":
                x = x[~x.index.duplicated(keep="first")]
            elif deduplicate in {"max", "min", "median"}:
                x = getattr(x.groupby(level=0), deduplicate)()
            else:
                raise ValueError(f"Unknown deduplicate policy: {deduplicate}")
        return x

    pseudo_bulk_rna_df = pd.DataFrame(
        X_rna_soft.T.toarray(),
        index=mdata["rna"].var_names,
        columns=mdata["rna"].obs_names,
    ).fillna(0)

    pseudo_bulk_atac_df = pd.DataFrame(
        X_atac_soft.T.toarray(),
        index=mdata["atac"].var_names,
        columns=mdata["atac"].obs_names,
    ).fillna(0)

    pseudo_bulk_rna_df = _standardize_symbols_index(pseudo_bulk_rna_df)
    pseudobulk_rna_file = sample_processed_data_dir / "TG_pseudobulk.parquet"
    pseudobulk_atac_file = sample_processed_data_dir / "RE_pseudobulk.parquet"

    pseudo_bulk_rna_df.to_parquet(pseudobulk_rna_file, engine="pyarrow", compression="snappy")
    pseudo_bulk_atac_df.to_parquet(pseudobulk_atac_file, engine="pyarrow", compression="snappy")
    
def get_threshold(sample_filtering_settings, setting_name):
    setting_value = sample_filtering_settings[setting_name].values[0]
    print(f"{setting_name}: {setting_value}")
    
    return setting_value
    
if __name__ == "__main__":
    args = parse_args()

    PROJECT_DIR = Path(args.project_dir)
    RAW_DATA_DIR = Path(args.raw_data_dir)
    PROCESSED_DATA_DIR = Path(args.processed_data_dir)
    SAMPLE_NAME = args.sample_name

    tss_path = Path(args.tss_path)
    rna_count_file = Path(args.rna_count_file) if args.rna_count_file else None
    atac_count_file = Path(args.atac_count_file) if args.atac_count_file else None
    raw_h5_file = Path(args.raw_h5_file) if args.raw_h5_file else None
    tf_list_file = Path(args.tf_list_file) if args.tf_list_file else None
    frag_path = Path(args.frag_path) if args.frag_path else None

    SAMPLE_DATA_DIR = RAW_DATA_DIR / SAMPLE_NAME
    SAMPLE_PROCESSED_DATA_DIR = PROCESSED_DATA_DIR / SAMPLE_NAME
    
    filtering_setting_df = pd.read_csv(PROJECT_DIR / "dev" / "notebooks" / "muon_preprocessing" /"qc_filtering_settings.tsv", sep="\t")
    sample_filtering_settings = filtering_setting_df[filtering_setting_df["Sample"] == SAMPLE_NAME]    
    
    # ----- RNA QC thresholds -----
    MIN_CELLS_PER_GENE = get_threshold(sample_filtering_settings, "Min Cells per Gene")
    MIN_GENES_PER_CELL = get_threshold(sample_filtering_settings, "Min Genes per Cell")
    MAX_GENES_PER_CELL = get_threshold(sample_filtering_settings, "Max Genes per Cell")
    MIN_TOTAL_COUNTS = get_threshold(sample_filtering_settings, "Min Total Counts")
    MAX_TOTAL_COUNTS = get_threshold(sample_filtering_settings, "Max Total Counts")
    MAX_PCT_COUNTS_MT = get_threshold(sample_filtering_settings, "Max Pct MT")

    # ----- ATAC QC thresholds -----
    MIN_CELLS_PER_PEAK = get_threshold(sample_filtering_settings, "Min Cells per Peak")
    MIN_PEAKS_PER_CELL = get_threshold(sample_filtering_settings, "Min Peaks per Cell")
    MAX_PEAKS_PER_CELL = get_threshold(sample_filtering_settings, "Max Peaks per Cell")
    MIN_TOTAL_PEAK_COUNTS = get_threshold(sample_filtering_settings, "Min Total Peak Counts")
    MAX_TOTAL_PEAK_COUNTS = get_threshold(sample_filtering_settings, "Max Total Peak Counts")

    if not SAMPLE_PROCESSED_DATA_DIR.exists():
        SAMPLE_PROCESSED_DATA_DIR.mkdir(parents=True)
    
    mdata = load_raw_data(SAMPLE_NAME, SAMPLE_DATA_DIR, rna_count_file, atac_count_file, raw_h5_file)

    mdata.write(SAMPLE_PROCESSED_DATA_DIR / f"{SAMPLE_NAME}.h5mu")
    
    data_processor = MudataProcessor(
        mdata=mdata,
        processed_data_dir=SAMPLE_PROCESSED_DATA_DIR,
        sample_name=SAMPLE_NAME,
        tss_path=tss_path,
        tf_list_file=tf_list_file
    )
    
    # RNA QC and Preprocessing
    data_processor.rna_qc_filter(
        min_cells_per_gene = MIN_CELLS_PER_GENE,
        min_genes_per_cell = MIN_GENES_PER_CELL,
        max_genes_per_cell = MAX_GENES_PER_CELL,
        min_total_counts_per_cell = MIN_TOTAL_COUNTS,
        max_total_counts_per_cell = MAX_TOTAL_COUNTS,
        max_pct_counts_mt = MAX_PCT_COUNTS_MT,
        norm_target_sum = 1e4,
        min_rna_disp = 0.5,
        filter_hvgs = False,
        tf_list_file = None,
        fig_dir=SAMPLE_PROCESSED_DATA_DIR / "preprocessing_figures" / "rna_qc",
        )
    
    data_processor.rna_pca_and_neighbors(
        data_processor.rna, 
        n_pcs=20,
        n_neighbors=10,
        fig_dir=SAMPLE_PROCESSED_DATA_DIR / "preprocessing_figures" / "rna_qc",
        )
    
    # ATAC QC and Preprocessing
    data_processor.atac_qc_filter(
        min_cells_per_peak=MIN_CELLS_PER_PEAK,
        min_peaks_per_cell=MIN_PEAKS_PER_CELL,
        max_peaks_per_cell=MAX_PEAKS_PER_CELL,
        min_total_counts_per_cell=MIN_TOTAL_PEAK_COUNTS,
        max_total_counts_per_cell=MAX_TOTAL_PEAK_COUNTS,
        min_atac_disp=0.5,
        promoter_upstream=1000,
        promoter_downstream=100,
        distal_max=200_000,
        filter_hvgs=False,
        fig_dir=SAMPLE_PROCESSED_DATA_DIR / "preprocessing_figures" / "atac_qc",
        )
    
    data_processor.nucleosome_signal(
        frag_path=frag_path, 
        fig_dir=SAMPLE_PROCESSED_DATA_DIR / "preprocessing_figures" / "atac_qc"
        )
    data_processor.tss_enrichment(
        frag_path=frag_path, 
        n_tss=500, 
        extend_upstream=1000, 
        extend_downstream=1000,
        fig_dir=SAMPLE_PROCESSED_DATA_DIR / "preprocessing_figures" / "atac_qc"
        )
    
    # Save the processed data
    save_processed_data(data_processor.mdata, SAMPLE_PROCESSED_DATA_DIR)
    
    # Integrate the RNA and ATAC modalities using MOFA+
    integrate_rna_atac(data_processor.mdata, SAMPLE_PROCESSED_DATA_DIR, SAMPLE_NAME, fig_dir=SAMPLE_PROCESSED_DATA_DIR / "integration")
    
    # Create metacells
    create_metacells(data_processor.mdata, SAMPLE_PROCESSED_DATA_DIR, hops=2)
    
