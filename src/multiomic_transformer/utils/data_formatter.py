from pathlib import Path

import pybedtools
import torch
from multiomic_transformer.utils.gene_canonicalizer import GeneCanonicalizer
from multiomic_transformer.utils.downloads import *

import json
import logging
import os
import re
import numpy as np
import pickle

from multiomic_transformer.utils.peaks import format_peaks, find_genes_near_peaks

logging.basicConfig(level=logging.INFO, format='%(message)s')

class TrainingDataFormatter:
    def __init__(
        self, 
        project_dir: Path,
        dataset_name: str,
        organism_code: str,
        sample_names: list[str],
        chrom_list: list[str],
        processed_data_dir: Path | None = None,
        output_dir: Path | None = None,
        ):
        
        self.project_dir = project_dir
        self.dataset_name = dataset_name
        self.organism_code = organism_code
        self.sample_names = sample_names
        self.chrom_list = chrom_list
        self.processed_data_dir = processed_data_dir
        self.output_dir = output_dir

        self.canon = None
        
        self.window_size_kb = 1
        
        self._setup_file_paths()
        self._ensure_genome_info_files_exist()
        
        self._file_setup_done = False
        
        self._handle_assertions()
        
        self.genes = None
        self.tfs = None
        self.tgs = None
        
        self.peaks = None
        self.peak_locs_df = None

    
    def load_pseudobulk_rna_df(self, sample_name: str):
        rna_file = Path(self.file_paths["processed"]["base_dir"]) / sample_name / "TG_pseudobulk.parquet"
        rna_df = pd.read_parquet(rna_file)
        self.canonicalize_gene_names(rna_df)
        
        self.genes = rna_df.index.tolist()
        
        # Creates the TF and TG files and stores the TF and TG names as a list
        self.tfs, self.tgs = self._split_genes_into_tfs_and_tgs(self.genes)
        
        return rna_df
    
    def load_pseudobulk_atac_df(self, sample_name: str):
        assert sample_name in self.sample_names, f"Sample name {sample_name} not in provided sample names: {self.sample_names}"
        
        atac_file = Path(self.file_paths["processed"]["base_dir"]) / sample_name / "RE_pseudobulk.parquet"
        atac_df = pd.read_parquet(atac_file)
        atac_df.index = atac_df.index.map(self.normalize_peak_format)
        
        self.peaks = atac_df.index.tolist()
        
        # Formats the peak locations and create a BED file for calculating peak-gene distances.
        self._create_peak_bed_file(atac_df, sample_name)

        return atac_df
    
    def aggregate_pseudobulk_datasets(self, force_recalculate=False):
        """
        Aggregate pseudobulk datasets across all samples and chromosomes.

        Parameters
        ----------
        force_recalculate : bool, optional
            Whether to recompute everything (default is False)

        Returns
        -------
        total_TG_pseudobulk_global : pd.DataFrame
            Global TG pseudobulk across all samples
        pseudobulk_chrom_dict : dict[str, dict]
            Per-chromosome aggregates of TG pseudobulk and RE pseudobulk
        """


        def _canon_series_same_len(s: pd.Series) -> pd.Series:
            """Canonicalize to same length; replace non-mapped with '' (caller filters)."""
            cs = self.canon.canonicalize_series(s.astype(str))
            return cs.fillna("")

        def _agg_sum(dfs: list[pd.DataFrame]) -> pd.DataFrame:
            if not dfs:
                raise ValueError("No DataFrames provided to aggregate.")
            if len(dfs) == 1:
                return dfs[0]
            return pd.concat(dfs).groupby(level=0).sum()

        def _agg_first(dfs: list[pd.DataFrame]) -> pd.DataFrame:
            if not dfs:
                raise ValueError("No DataFrames provided to aggregate.")
            if len(dfs) == 1:
                return dfs[0]
            return pd.concat(dfs).groupby(level=0).first()
        
        sample_level_peak_to_gene_dist_dfs = []
        for sample_name in self.sample_names:
            processed_data_dir = self.file_paths["processed"]["base_dir"]
            peak_to_gene_dist_file = processed_data_dir / sample_name / "peak_to_gene_dist.parquet"
            
            peak_to_gene_dist_df = pd.read_parquet(peak_to_gene_dist_file, engine="pyarrow")
            sample_level_peak_to_gene_dist_dfs.append(peak_to_gene_dist_df)
        
        total_tg_pseudobulk_path = self.file_paths["processed"]["tg_pseudobulk_global"]
        pseudobulk_chrom_dict_path = self.file_paths["processed"]["base_dir"] / "pseudobulk_chrom_dict.pkl"
        
        # Decide whether to recompute everything
        need_recalc = (
            force_recalculate
            or not total_tg_pseudobulk_path.is_file()
            or not pseudobulk_chrom_dict_path.is_file()
        )
        if need_recalc:
            logging.info("  - Loading processed pseudobulk datasets:")
            logging.info(f"   - Sample names: {self.sample_names}")
            logging.info(f"   - Looking for processed samples in {self.file_paths['processed']['base_dir']}")

            # ---- 1) Build per-sample TG pseudobulk ----
            per_sample_TG: dict[str, pd.DataFrame] = {}
            for sample_name in self.sample_names:
                sample_raw_dir = self.file_paths["processed"]["base_dir"] / sample_name
                tg_path = sample_raw_dir / "TG_pseudobulk.parquet"
                TG_pseudobulk = pd.read_parquet(tg_path, engine="pyarrow")
                TG_pseudobulk = self._canon_index_sum(TG_pseudobulk)
                per_sample_TG[sample_name] = TG_pseudobulk

            # Global TG pseudobulk across all samples
            total_TG_pseudobulk_global = _agg_sum(list(per_sample_TG.values()))

            # ---- 2) Build per-chromosome aggregates ----
            pseudobulk_chrom_dict: dict[str, dict] = {}
            logging.info("  - Aggregating per-chromosome pseudobulk datasets:")
            for chrom_id in self.chrom_list:
                logging.info(f"   - Aggregating data for {chrom_id}")

                TG_pseudobulk_samples = []
                RE_pseudobulk_samples = []
                peaks_df_samples = []
                
                def make_chrom_gene_tss_df(gene_tss_file: Union[str, Path], chrom_id: str, genome_dir: Union[str, Path]) -> pd.DataFrame:
                    gene_tss_bed = pybedtools.BedTool(gene_tss_file)
                    gene_tss_df = (
                        gene_tss_bed
                        .filter(lambda x: x.chrom == chrom_id)
                        .saveas(os.path.join(genome_dir, f"{chrom_id}_gene_tss.bed"))
                        .to_dataframe()
                        .sort_values(by="start", ascending=True)
                        )
                    gene_tss_df["name"] = gene_tss_df["name"].astype(str).str.upper()
                    gene_tss_df = gene_tss_df.drop_duplicates(subset=["name"], keep="first")
                    return gene_tss_df
    
                # gene TSS for this chromosome
                genome_dir = self.file_paths["genome"]["genome_dir"]
                chrom_tss_path = genome_dir / f"{chrom_id}_gene_tss.bed"
                if not chrom_tss_path.is_file():
                    gene_tss_chrom = make_chrom_gene_tss_df(
                        gene_tss_file=self.file_paths["genome"]["gene_tss_file"],
                        chrom_id=chrom_id,
                        genome_dir=genome_dir,
                    )
                else:
                    gene_tss_chrom = pd.read_csv(
                        chrom_tss_path,
                        sep="\t",
                        header=None,
                        usecols=[0, 1, 2, 3],
                    )
                    gene_tss_chrom = gene_tss_chrom.rename(
                        columns={0: "chrom", 1: "start", 2: "end", 3: "name"}
                    )

                gene_tss_chrom["name"] = _canon_series_same_len(gene_tss_chrom["name"])
                gene_tss_chrom = gene_tss_chrom[gene_tss_chrom["name"] != ""]
                gene_tss_chrom = gene_tss_chrom.drop_duplicates(subset=["name"], keep="first")
                genes_on_chrom = gene_tss_chrom["name"].tolist()

                for sample_name in self.sample_names:
                    sample_raw_dir = self.file_paths["processed"]["base_dir"] / sample_name

                    # RE pseudobulk: peaks x metacells (loaded from per-sample raw directory)
                    re_path = sample_raw_dir / "RE_pseudobulk.parquet"
                    RE_pseudobulk = pd.read_parquet(re_path, engine="pyarrow")

                    # TG: restrict to genes on this chrom
                    TG_chr_specific = per_sample_TG[sample_name].loc[
                        per_sample_TG[sample_name].index.intersection(genes_on_chrom)
                    ]

                    # RE: restrict to this chrom (handle both chr:start-end and chr-start-end formats)
                    mask_colon = RE_pseudobulk.index.str.startswith(f"{chrom_id}:")
                    mask_dash = RE_pseudobulk.index.str.startswith(f"{chrom_id}-")
                    RE_chr_specific = RE_pseudobulk[mask_colon | mask_dash]
                    
                    logging.debug(f"      - Sample {sample_name}, {chrom_id}: {len(RE_chr_specific)} peaks matched")
                    if len(RE_chr_specific) > 0:
                        logging.debug(f"      - First few peaks: {RE_chr_specific.index[:3].tolist()}")

                    # Build peaks df from RE index
                    # Handle both colon-separated (chr:start-end) and dash-separated (chr-start-end) formats
                    # Normalize all peaks to chr:start-end format for consistent storage
                    peaks_df = (
                        RE_chr_specific.index.to_series()
                        .str.split("[-:]", n=2, expand=True, regex=True)
                        .rename(columns={0: "chrom", 1: "start", 2: "end"})
                    )
                    peaks_df["start"] = peaks_df["start"].astype(int)
                    peaks_df["end"] = peaks_df["end"].astype(int)
                    peaks_df["peak_id"] = RE_chr_specific.index.to_series()

                    TG_pseudobulk_samples.append(TG_chr_specific)
                    RE_pseudobulk_samples.append(RE_chr_specific)
                    peaks_df_samples.append(peaks_df)

                # Aggregate across samples for this chromosome
                total_TG_pseudobulk_chr = _agg_sum(TG_pseudobulk_samples)
                total_RE_pseudobulk_chr = _agg_sum(RE_pseudobulk_samples)
                total_peaks_df = _agg_first(peaks_df_samples)

                pseudobulk_chrom_dict[chrom_id] = {
                    "total_TG_pseudobulk_chr": total_TG_pseudobulk_chr,
                    "total_RE_pseudobulk_chr": total_RE_pseudobulk_chr,
                    "total_peaks_df": total_peaks_df,
                }

            # ---- 3) Save aggregates ----
            total_TG_pseudobulk_global.to_parquet(
                total_tg_pseudobulk_path,
                engine="pyarrow",
                compression="snappy",
            )
            with open(pseudobulk_chrom_dict_path, "wb") as f:
                pickle.dump(pseudobulk_chrom_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            # Load both from disk
            logging.info("  - Found existing global and per-chrom pseudobulk; loading...")
            total_TG_pseudobulk_global = pd.read_parquet(
                total_tg_pseudobulk_path,
                engine="pyarrow",
            )
            with open(pseudobulk_chrom_dict_path, "rb") as f:
                pseudobulk_chrom_dict = pickle.load(f)

        return total_TG_pseudobulk_global, pseudobulk_chrom_dict
    
    def create_chrom_files(self):
        total_TG_pseudobulk_global, pseudobulk_chrom_dict = self.aggregate_pseudobulk_datasets(force_recalculate=False)
        
        # genome-wide TF expression for all metacells (columns)
        genome_wide_tf_expression = (
            total_TG_pseudobulk_global
            .reindex(self.tfs)           # ensure row order matches your TF list
            .fillna(0)
            .values.astype("float32")
        )
        tf_tensor_all = torch.from_numpy(genome_wide_tf_expression)  # [T, C]
    
        common_tf_vocab_file = self.file_paths["training_cache"]["common"]["tf_vocab"]
    
        # ensure common TF vocab exists, else initialize from tf_names
        if not os.path.exists(common_tf_vocab_file):
            with open(common_tf_vocab_file, "w") as f:
                json.dump({n: i for i, n in enumerate(self.tfs)}, f)

        with open(common_tf_vocab_file) as f:
            tf_vocab = json.load(f)

        # align TF tensor to vocab order (and get kept names/ids)
        tf_tensor_all_aligned, tf_names_kept, tf_ids = self.align_to_vocab(
            self.tfs, tf_vocab, tf_tensor_all, label="TF"
        )
        
        for chrom_id in self.chrom_list:
            logging.info(f"Creating chromosome-specific files for {chrom_id}")
            chrom_specific_dir = self.file_paths["training_cache"][chrom_id]["dir"]
            chrom_specific_dir.mkdir(parents=True, exist_ok=True)
            
            TG_pseudobulk_samples = []
            RE_pseudobulk_samples = []
            peaks_df_samples = []
            
            gene_tss_chrom = self._load_or_create_chrom_gene_tss_df(chrom_id=chrom_id)
            gene_tss_chrom = gene_tss_chrom.drop_duplicates(subset=["name"], keep="first")
            genes_on_chrom = gene_tss_chrom["name"].tolist()
            
            total_TG_pseudobulk_chr = pseudobulk_chrom_dict[chrom_id]["total_TG_pseudobulk_chr"]
            total_RE_pseudobulk_chr = pseudobulk_chrom_dict[chrom_id]["total_RE_pseudobulk_chr"]
            total_peaks_df = pseudobulk_chrom_dict[chrom_id]["total_peaks_df"]
            
            vals = total_TG_pseudobulk_chr.values.astype("float32")
            
            tg_names = total_TG_pseudobulk_chr.index.tolist()
            
            # Genome-wide TF expression for all samples
            genome_wide_tf_expression = total_TG_pseudobulk_global.reindex(self.tfs).fillna(0).values.astype("float32")
            metacell_names = total_TG_pseudobulk_global.columns.tolist()
        
                
    def align_to_vocab(self, names: list[str], vocab: dict[str, int], tensor_all: torch.Tensor, label: str = "genes") -> tuple[torch.Tensor, list[str], list[int]]:
        """
        Restrict to the subset of names that exist in the global vocab.
        Returns:
        aligned_tensor : [num_kept, C] (chromosome-specific subset)
        kept_names     : list[str] of kept names (order = aligned_tensor rows)
        kept_ids       : list[int] global vocab indices for kept names
        """
        kept_ids = []
        kept_names = []
        aligned_rows = []

        for i, n in enumerate(names):
            vid = vocab.get(n)
            if vid is not None:
                kept_ids.append(vid)
                kept_names.append(n)
                aligned_rows.append(tensor_all[i])

        if not kept_ids:
            raise ValueError(f"No {label} matched the global vocab.")

        aligned_tensor = torch.stack(aligned_rows, dim=0)  # [num_kept, num_cells]

        return aligned_tensor, kept_names, kept_ids

    
    def create_peak_to_tg_distance_file(
        self, 
        sample_name: str,
        max_peak_distance=1e6, 
        distance_factor_scale=25000, 
        force_recalculate=False,
        filter_to_nearest_gene=False,
        promoter_bp=None
        ) -> pd.DataFrame:
        
        if not force_recalculate and self.file_paths["samples"][sample_name]["peak_to_gene_dist"].is_file():
            logging.info(f"Peak to TSS distance file already exists for sample {sample_name} at {self.file_paths['samples'][sample_name]['peak_to_gene_dist']}. Skipping recalculation.")
            return pd.read_parquet(self.file_paths["samples"][sample_name]["peak_to_gene_dist"])
        
        else:
            logging.info(f"Calculating peak to TSS distances for sample {sample_name} and saving to {self.file_paths['samples'][sample_name]['peak_to_gene_dist']}.")
            peak_bed_file = self.file_paths["samples"][sample_name]["peaks"]
            tss_bed_file = self.file_paths["genome"]["gene_tss"]
            
            peak_bed = pybedtools.BedTool(str(peak_bed_file))
            tss_bed = pybedtools.BedTool(str(tss_bed_file))

            # Step 3: Find peaks near TSS and compute distances
            genes_near_peaks = find_genes_near_peaks(peak_bed, tss_bed, tss_distance_cutoff=max_peak_distance)
            
            genes_near_peaks = genes_near_peaks.rename(columns={"gene_id": "target_id"})
            genes_near_peaks["target_id"] = genes_near_peaks["target_id"].str.upper()
            
            if "TSS_dist" not in genes_near_peaks.columns:
                raise ValueError("Expected column 'TSS_dist' missing from find_genes_near_peaks output.")
            
            # Step 4: Compute the distance score
            genes_near_peaks["TSS_dist_score"] = np.exp(-genes_near_peaks["TSS_dist"] / float(distance_factor_scale))
            
            # Step 5: Drop rows where the peak is too far from the gene
            genes_near_peaks = genes_near_peaks[genes_near_peaks["TSS_dist"] <= max_peak_distance]
            
            if promoter_bp is not None:
                # Subset to keep genes that are near gene promoters
                genes_near_peaks = genes_near_peaks[genes_near_peaks["TSS_dist"] <= int(promoter_bp)]
                
            if filter_to_nearest_gene:
                # Filter to use the gene closest to the peak
                genes_near_peaks = (genes_near_peaks.sort_values(["TSS_dist_score","TSS_dist","target_id"],
                                    ascending=[False, True, True], kind="mergesort")
                        .drop_duplicates(subset=["peak_id"], keep="first"))
            
            # Save the peak to TSS distance file for this sample
            peak_to_tss_dist_file = self.file_paths["samples"][sample_name]["peak_to_gene_dist"]
            peak_to_tss_dist_file.parent.mkdir(parents=True, exist_ok=True)
            genes_near_peaks.to_parquet(peak_to_tss_dist_file, index=False)
        
            return genes_near_peaks
    
    @staticmethod
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
    
    def build_global_tg_vocab(self):
        """
        Builds a global TG vocab from the TSS file with contiguous IDs [0..N-1].
        Overwrites existing vocab if it's missing or non-contiguous.
        
        Parameters
        ----------
        gene_tss_file : str | Path
            Gene TSS bed file with columns 'chrom', 'start', 'end', 'name'
        vocab_file : str | Path
            File to write the global TG vocab to
        
        Returns
        -------
        dict[str, int]
            Global TG vocab with contiguous IDs [0..N-1]
        """
        gene_tss_file = self.file_paths["genome"]["gene_tss"]
        vocab_file = self.file_paths["training_cache"]["common"]["tg_vocab"]
        
        # 1) Load all genes genome-wide (bed: chrom start end name)
        gene_tss_bed = pybedtools.BedTool(gene_tss_file)
        gene_tss_df = gene_tss_bed.to_dataframe().sort_values(by="start", ascending=True)

        gene_tss_df["name"] = self.canon.canonicalize_series(gene_tss_df["name"])
        tss_genes = set(gene_tss_df["name"])
        tg_set = set(self.tgs)

        names = sorted(tss_genes & tg_set)

        logging.info(f"  - Writing global TG vocab with {len(names)} genes")

        # 3) Build fresh contiguous mapping
        vocab = {name: i for i, name in enumerate(names)}
        self.atomic_json_dump(vocab, vocab_file)

        return vocab

    def _setup_file_paths(self):
        if self.output_dir is None:
            self.output_dir = self.project_dir / "output" / self.dataset_name

        assert self.dataset_name in self.output_dir.name, \
            f"Output directory {self.output_dir} must contain the dataset name {self.dataset_name}"

        if not self.output_dir.is_dir():
            logging.info(f"Creating output directory: {self.output_dir}")
            os.makedirs(self.output_dir)

        # Initialize dictionary
        self.file_paths = {
            "genome": {},
            "processed": {},
            "training_cache": {},
            "common": {},
            "samples": {}
        }

        # ----- GENOME AND ANNOTATION FILES -----
        genome_dir = self.project_dir / "data" / "genome_data" / "reference_genome" / self.organism_code
        gene_annotation_dir = self.project_dir / "data" / "genome_data" / "genome_annotation" / self.organism_code

        self.file_paths["genome"] = {
            "genome_dir": genome_dir,
            "gene_annotation_dir": gene_annotation_dir,
            "gene_tss": gene_annotation_dir / f"{self.organism_code}_gene_tss.bed",
            "chrom_sizes": genome_dir / f"{self.organism_code}.chrom.sizes",
            "tf_info": self.project_dir / "data" / "databases" / "motif_information" / self.organism_code / "TF_Information_all_motifs.txt"
        }

        # ----- PROCESSED DATA DIRECTORY -----
        if self.processed_data_dir is None:
            self.processed_data_dir = self.project_dir / "data" / "processed" / self.dataset_name

        self.file_paths["processed"] = {
            "base_dir": self.processed_data_dir,
            "tg_pseudobulk_global": self.processed_data_dir / "total_TG_pseudobulk_global.parquet"
        }

        # ----- TRAINING CACHE -----
        training_data_cache = self.project_dir / "data" / "training_data_cache"
        sample_cache_dir = training_data_cache / self.dataset_name
        common_data = sample_cache_dir / "common"
        
        if not sample_cache_dir.is_dir():
            os.makedirs(sample_cache_dir)

        self.file_paths["training_cache"] = {
            "base_dir": training_data_cache,
            "dataset_dir": sample_cache_dir,
            "common": {
                    "dir": common_data,
                    "tf_vocab": common_data / "tf_vocab.json",
                    "tg_vocab": common_data / "tg_vocab.json",
            },
            "tf_tensor": sample_cache_dir / "tf_tensor_all.pt",
            "tf_names": sample_cache_dir / "tf_names.json",
            "tf_ids": sample_cache_dir / "tf_ids.pt",
            "metacell_names": sample_cache_dir / "metacell_names.json",
        }

        # ----- TF-TG COMBO FILES -----
        tf_tg_combos_dir = self.processed_data_dir / "tf_tg_combos"

        self.file_paths["processed"]["tf_tg_combos"] = {
            "dir": tf_tg_combos_dir,
            "total_genes": tf_tg_combos_dir / "total_genes.csv",
            "tf_list": tf_tg_combos_dir / "tf_list.csv",
            "tg_list": tf_tg_combos_dir / "tg_list.csv",
        }

        # ----- SAMPLE-LEVEL FILES -----
        for sample_name in self.sample_names:
            sample_dir = self.processed_data_dir / sample_name

            self.file_paths["samples"][sample_name] = {
                "dir": sample_dir,
                "peaks": sample_dir / "peaks.bed",
                "peak_to_gene_dist": sample_dir / "peak_to_gene_dist.parquet",
                "sliding_window": sample_dir / "sliding_window.parquet",
                "tf_tg_reg_potential": sample_dir / "tf_tg_regulatory_potential.parquet",
            }
            
        # ----- CHROMOSOME-LEVEL FILES -----
        for chrom_id in self.chrom_list:
            chrom_cache_dir = sample_cache_dir / chrom_id
            # Chromosome-specific cache files
            atac_tensor_path: Path =            chrom_cache_dir / f"atac_window_tensor_all_{chrom_id}.pt"
            tg_tensor_path: Path =              chrom_cache_dir / f"tg_tensor_all_{chrom_id}.pt"
            sample_tg_name_file: Path =         chrom_cache_dir / f"tg_names_{chrom_id}.json"
            genome_window_file: Path =          chrom_cache_dir / f"{chrom_id}_windows_{self.window_size_kb}kb.bed"
            sample_window_map_file: Path =      chrom_cache_dir / f"window_map_{chrom_id}.json"
            peak_to_tss_dist_path: Path =       chrom_cache_dir / f"genes_near_peaks_{chrom_id}.parquet"
            dist_bias_file: Path =              chrom_cache_dir / f"dist_bias_{chrom_id}.pt"
            tg_id_file: Path =                  chrom_cache_dir / f"tg_ids_{chrom_id}.pt"
            manifest_file: Path =               chrom_cache_dir / f"manifest_{chrom_id}.json"
            chrom_peak_bed_file: Path =         chrom_cache_dir / f"peak_tmp_{chrom_id}.bed"
            tss_bed_file: Path =                chrom_cache_dir / f"tss_tmp_{chrom_id}.bed"
            
            self.file_paths["training_cache"].update({
                chrom_id: {
                    "dir": chrom_cache_dir,
                    "atac_tensor": atac_tensor_path,
                    "tg_tensor": tg_tensor_path,
                    "tg_names": sample_tg_name_file,
                    "genome_windows": genome_window_file,
                    "window_map": sample_window_map_file,
                    "peak_to_tss_dist": peak_to_tss_dist_path,
                    "dist_bias": dist_bias_file,
                    "tg_ids": tg_id_file,
                    "manifest": manifest_file,
                    "chrom_peak_bed": chrom_peak_bed_file,
                    "tss_bed": tss_bed_file,
                }
            })
            
        self.atomic_json_dump(self.file_paths, self.output_dir / "file_paths.json")

        self._file_setup_done = True

    def _ensure_genome_info_files_exist(self):
        
        # Set up the file paths if they haven't been set up yet
        if not self._file_setup_done:
            self._setup_file_paths()
            
        genome_dir = self.file_paths["genome"]["genome_dir"]
        chrom_sizes_file = self.file_paths["genome"]["chrom_sizes"]
        gene_tss_file = self.file_paths["genome"]["gene_tss"]
        gene_annotation_dir = self.file_paths["genome"]["gene_annotation_dir"]
                
        if not os.path.isdir(genome_dir):
            os.makedirs(genome_dir)
        
        download_genome_fasta(
            organism_code=self.organism_code,
            save_dir=genome_dir
        )
            
        if not os.path.isfile(chrom_sizes_file):
            download_chrom_sizes(
                organism_code=self.organism_code,
                save_dir=genome_dir
            )
            
        # Check organism code
        if self.organism_code == "mm10":
            ensemble_dataset_name = "mmusculus_gene_ensembl"
        elif self.organism_code == "hg38":
            ensemble_dataset_name = "hsapiens_gene_ensembl"
        else:
            raise ValueError(f"Organism not recognized: {self.organism_code} (must be 'mm10' or 'hg38').")
        
        if not os.path.isfile(gene_tss_file):
            download_gene_tss_file(
                save_file=gene_tss_file,
                gene_dataset_name=ensemble_dataset_name,
            )
        
        # Download organism-specific NCBI gene info and Ensembl GTF
        download_ncbi_gene_info(
            organism_code=self.organism_code, 
            out_path=gene_annotation_dir
            )
        
        # Download GTF file and set the information for the NCBI files based on the organism code
        if self.organism_code == "mm10":
            species_taxid = "10090"
            species_file_name = "Mus_musculus"
            gtf_assembly = "GRCm39.115"
            download_ensembl_gtf(
                organism_code=self.organism_code, 
                release=115, 
                assembly="GRCm39", 
                decompress=False, 
                out_dir=gene_annotation_dir
                )
        
        elif self.organism_code == "hg38":
            species_taxid = "9606"
            species_file_name = "Homo_sapiens"
            gtf_assembly = "GRCh38.113"
            download_ensembl_gtf(
                organism_code=self.organism_code, 
                release=113, 
                assembly="GRCh38", 
                decompress=False,
                out_dir=gene_annotation_dir
                )
        
        gtf_file = gene_annotation_dir / f"{species_file_name}.{gtf_assembly}.gtf.gz"
        ncbi_file = gene_annotation_dir / f"{species_file_name}.gene_info.gz"
        
        # Create the gene canonicalizer
        gc = GeneCanonicalizer(use_mygene=False)
        gc.load_gtf(str(gtf_file))
        gc.load_ncbi_gene_info(str(ncbi_file), species_taxid=species_taxid)
        
        self.canon = gc
        logging.info(f"Map sizes: {gc.coverage_report()}")

    def atomic_json_dump(self, obj, path: Path):
        """Safe JSON dump, avoids race conditions by making a tmp file first, then updating the name"""
        def convert(o):
            if isinstance(o, Path):
                return str(o)
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2, default=convert)
        os.replace(tmp, path)
    
    def canonicalize_gene_names(self, rna_df):
        rna_df.index = pd.Index(
            self.canon.canonicalize_series(pd.Series(rna_df.index, dtype=object)).array
        )
    
    def check_cached_file_exist(self):
        missing_files = []
        
        # Check if all required output files exist
        sample_specific_required_files = [
            self.file_paths["training_cache"]["tf_tensor"],
            self.file_paths["training_cache"]["tf_ids"],
            self.file_paths["training_cache"]["tf_names"],
            self.file_paths["training_cache"]["metacell_names"],
            self.file_paths["training_cache"]["common"]["tf_vocab"],
            self.file_paths["training_cache"]["common"]["tg_vocab"],
        ]
        for file in sample_specific_required_files:
            if not file.is_file():
                missing_files.append(file.name)
        
        for chrom_id in self.chrom_list:
            chrom_cache_dir = self.file_paths["training_cache"]["dataset_dir"] / chrom_id
            chrom_specific_required_files = [
                chrom_cache_dir / f"atac_window_tensor_all_{chrom_id}.pt",
                chrom_cache_dir / f"tg_tensor_all_{chrom_id}.pt",
                chrom_cache_dir / f"tg_names_{chrom_id}.json",
                chrom_cache_dir / f"{chrom_id}_windows_{self.window_size_kb}kb.bed",
                chrom_cache_dir / f"window_map_{chrom_id}.json",
                chrom_cache_dir / f"genes_near_peaks_{chrom_id}.parquet",
                chrom_cache_dir / f"dist_bias_{chrom_id}.pt",
                chrom_cache_dir / f"tg_ids_{chrom_id}.pt",
                chrom_cache_dir / f"manifest_{chrom_id}.json",
            ]
            for file in chrom_specific_required_files:
                if not file.is_file():
                    missing_files.append(file.name)
        
        if len(missing_files) > 0:
            logging.info(f"Missing required files: {missing_files}")
            return False
        else:
            logging.info("All required files are present.")
            return True
    
    def _split_genes_into_tfs_and_tgs(self, genes):
        """Creates TF and TG combo files based on the provided gene list and the reference TF list."""
        
        tf_name_col = "TF_Name"
        
        def _canon(x: str) -> str:
            """
            Strips version suffix and uppercases a given string.
            """
            
            # strip version suffix and uppercase
            s = str(x).strip()
            s = re.sub(r"\.\d+$", "", s)
            return s.upper()
        
        # --- normalize incoming genes ---
        genes_norm = sorted({_canon(g) for g in genes if pd.notna(g)})

        # --- load TF reference file robustly (auto-detect column if needed) ---
        tf_list_file = self.file_paths["genome"]["tf_info"]
        tf_list_file = Path(tf_list_file)
        tf_ref = pd.read_csv(tf_list_file, sep=None, engine="python")  # auto-detect delim
        if tf_name_col and tf_name_col in tf_ref.columns:
            tf_col = tf_name_col
        else:
            # attempt to auto-detect a sensible TF column
            lower = {c.lower(): c for c in tf_ref.columns}
            for cand in ("tf_name", "tf", "symbol", "gene_symbol", "gene", "name"):
                if cand in lower:
                    tf_col = lower[cand]
                    break
            else:
                # if exactly one column, use it
                if tf_ref.shape[1] == 1:
                    tf_col = tf_ref.columns[0]
                else:
                    raise ValueError(
                        f"Could not locate TF name column in {tf_list_file}. "
                        f"Available columns: {list(tf_ref.columns)}"
                    )

        known_tfs = {_canon(x) for x in tf_ref[tf_col].dropna().astype(str).tolist()}

        # --- new sets from this call ---
        tfs_new = sorted(set(genes_norm) & known_tfs)
        tgs_new = sorted(set(genes_norm) - set(tfs_new))

        total_file = self.file_paths["processed"]["tf_tg_combos"]["total_genes"]
        tf_file    = self.file_paths["processed"]["tf_tg_combos"]["tf_list"]
        tg_file    = self.file_paths["processed"]["tf_tg_combos"]["tg_list"]
        
        def _read_list(path: Path, col: str) -> list[str]:
            """
            Reads a list of elements from a CSV file.
            """
            if path.is_file():
                df = pd.read_csv(path)
                if col not in df.columns and df.shape[1] == 1:
                    # tolerate unnamed single column
                    return sorted({_canon(v) for v in df.iloc[:, 0].astype(str)})
                return sorted({_canon(v) for v in df[col].dropna().astype(str)})
            return []

        total_existing = _read_list(total_file, "Gene")
        tf_existing    = _read_list(tf_file, "TF")
        tg_existing    = _read_list(tg_file, "TG")

        total = sorted(set(total_existing) | set(genes_norm))
        tfs   = sorted(set(tf_existing)    | set(tfs_new))
        tgs   = sorted((set(tg_existing) | set(tgs_new)) - set(tfs))
        
        tfs = sorted(set(self.canon.canonicalize_series(pd.Series(tfs)).tolist()))
        tgs = sorted(set(self.canon.canonicalize_series(pd.Series(tgs)).tolist()))

        pd.DataFrame({"Gene": total}).to_csv(total_file, index=False)
        pd.DataFrame({"TF": tfs}).to_csv(tf_file, index=False)
        pd.DataFrame({"TG": tgs}).to_csv(tg_file, index=False)

        return tfs, tgs
    
    def _create_peak_bed_file(self, atac_df, sample_name):
        self.peak_locs_df = format_peaks(pd.Series(atac_df.index)).rename(columns={"chromosome": "chrom"})
        peak_bed_file = self.file_paths["samples"][sample_name]["peaks"]
        
        if not os.path.isfile(peak_bed_file):
            # Write the peak BED file
            peak_bed_file.parent.mkdir(parents=True, exist_ok=True)
            pybedtools.BedTool.from_dataframe(
                self.peak_locs_df[["chrom", "start", "end", "peak_id"]]
            ).saveas(peak_bed_file)
    
    def _build_peak_locs_from_index(
        self,
        peak_index: pd.Index,
        coerce_chr_prefix: bool = True,
    ) -> pd.DataFrame:
        """
        Parse peak ids like 'chr1:100-200' into chrom, start, end, and peak_id columns

        """
        rows = []
        for pid in map(str, peak_index):
            try:
                chrom_part, se = pid.split(":", 1)
                if coerce_chr_prefix and not chrom_part.startswith("chr"):
                    chrom = f"chr{chrom_part}"
                else:
                    chrom = chrom_part
                start, end = se.split("-", 1)
                start, end = int(start), int(end)
                if start > end:
                    start, end = end, start
                rows.append((chrom, start, end, pid))
            except Exception as ex:
                logging.warning(f"Skipping malformed peak ID '{pid}': {ex}")
                continue
        return pd.DataFrame(rows, columns=["chrom", "start", "end", "peak_id"])

    def _canon_index_sum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Canonicalize df.index with GeneCanonicalizer and sum duplicate rows."""
        if df.empty:
            return df
        mapped = self.canon.canonicalize_series(pd.Series(df.index, index=df.index))
        out = df.copy()
        out.index = mapped.values
        out = out[out.index != ""]  # drop unmapped
        if not out.index.is_unique:
            out = out.groupby(level=0).sum()
        return out

    def _load_or_create_chrom_gene_tss_df(self, chrom_id: str) -> pd.DataFrame:
        chrom_tss_path = genome_dir / f"{chrom_id}_gene_tss.bed"
        if not chrom_tss_path.is_file():
            genome_dir = self.file_paths["genome"]["genome_dir"]
            gene_tss_file = self.file_paths["genome"]["gene_tss"]
            
            gene_tss_bed = pybedtools.BedTool(gene_tss_file)
            gene_tss_df = (
                gene_tss_bed
                .filter(lambda x: x.chrom == chrom_id)
                .saveas(os.path.join(genome_dir, f"{chrom_id}_gene_tss.bed"))
                .to_dataframe()
                .sort_values(by="start", ascending=True)
                )
            gene_tss_df = gene_tss_df.drop_duplicates(subset=["name"], keep="first")
        else:
            gene_tss_df = pd.read_csv(chrom_tss_path, sep="\t", header=None, usecols=[0, 1, 2, 3])
            gene_tss_df = gene_tss_df.rename(
                columns={0: "chrom", 1: "start", 2: "end", 3: "name"}
            )
        return gene_tss_df

    def _handle_assertions(self):
        tf_file = self.file_paths["genome"]["tf_info"]
        
        assert self.project_dir is not None, "Project directory is not specified"
        assert self.organism_code is not None, "Organism code is not specified. Please specify 'mm10' or 'hg38'"
        assert self.organism_code in ["mm10", "hg38"], f"Unsupported organism code: {self.organism_code}"
        assert tf_file.is_file(), f"TF information file not found: {tf_file}"
        assert self.processed_data_dir.is_dir(), f"Processed data directory not found: {self.processed_data_dir}"