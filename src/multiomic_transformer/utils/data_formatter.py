from pathlib import Path
from typing import Iterable, List

import pandas as pd
import pybedtools
import torch

import json
import logging
import os
import re
import numpy as np
import scipy.sparse as sp
import pickle
import random

import sys
PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
SRC_DIR = str(Path(PROJECT_DIR) / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from multiomic_transformer.utils.gene_canonicalizer import GeneCanonicalizer
from multiomic_transformer.utils.downloads import *
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
        self.window_size = 1000
        self.distance_scale_factor = 20000
        self.max_peak_tg_distance = 100000
        
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
        assert sample_name in self.sample_names, \
            f"Sample name {sample_name} not in provided sample names: {self.sample_names}"
        
        rna_file = Path(self.file_paths["processed"]["base_dir"]) / sample_name / "TG_pseudobulk.parquet"
        
        assert rna_file.is_file(), \
            f"RNA pseudobulk file not found for sample {sample_name} at expected path: {rna_file}"
        
        rna_df = pd.read_parquet(rna_file)
        self.canonicalize_gene_names(rna_df)
        
        self.genes = rna_df.index.tolist()
        
        # Creates the TF and TG files and stores the TF and TG names as a list
        self.tfs, self.tgs = self.split_genes_into_tfs_and_tgs(self.genes)
        
        return rna_df
    
    def load_pseudobulk_atac_df(self, sample_name: str):
        assert sample_name in self.sample_names, \
            f"Sample name {sample_name} not in provided sample names: {self.sample_names}"
        
        atac_file = Path(self.file_paths["processed"]["base_dir"]) / sample_name / "RE_pseudobulk.parquet"
        
        assert atac_file.is_file(), \
            f"ATAC pseudobulk file not found for sample {sample_name} at expected path: {atac_file}"
        
        atac_df = pd.read_parquet(atac_file)
        assert atac_df is not None, \
            f"Failed to load ATAC pseudobulk file for sample {sample_name}"
        atac_df.index = atac_df.index.map(self.normalize_peak_format)
        
        self.peaks = atac_df.index.tolist()
        
        # Formats the peak locations and create a BED file for calculating peak-gene distances.
        self.create_peak_bed_file(atac_df, sample_name)

        return atac_df
    
    def aggregate_pseudobulk_datasets(self, force_recalculate: bool = False):
        """
        Aggregate pseudobulk datasets across all samples and chromosomes.

        Parameters
        ----------
        force_recalculate : bool, optional
            Whether to recompute everything.

        Returns
        -------
        total_TG_pseudobulk_global : pd.DataFrame
            Global TG pseudobulk across all samples.
        pseudobulk_chrom_dict : dict[str, dict]
            Per-chromosome aggregates of TG pseudobulk, RE pseudobulk, and peak metadata.
        """
        total_tg_pseudobulk_path = self.file_paths["processed"]["tg_pseudobulk_global"]
        pseudobulk_chrom_dict_path = self.file_paths["processed"]["base_dir"] / "pseudobulk_chrom_dict.pkl"

        self._aggregate_peak_to_gene_dist()

        if not self._should_recalculate_pseudobulk(
            total_tg_pseudobulk_path=total_tg_pseudobulk_path,
            pseudobulk_chrom_dict_path=pseudobulk_chrom_dict_path,
            force_recalculate=force_recalculate,
        ):
            logging.info("  - Found existing global and per-chrom pseudobulk; loading...")
            return self._load_pseudobulk_aggregates(
                total_tg_pseudobulk_path,
                pseudobulk_chrom_dict_path,
            )

        logging.info("  - Loading processed pseudobulk datasets:")
        logging.info(f"   - Sample names: {self.sample_names}")
        logging.info(f"   - Looking for processed samples in {self.file_paths['processed']['base_dir']}")

        per_sample_tg = self._load_per_sample_tg_pseudobulk()
        total_tg_pseudobulk_global = self._sum_dataframes(per_sample_tg.values())

        pseudobulk_chrom_dict = self._build_chromosome_pseudobulk_dict(per_sample_tg)

        self._save_pseudobulk_aggregates(
            total_tg_pseudobulk_global=total_tg_pseudobulk_global,
            pseudobulk_chrom_dict=pseudobulk_chrom_dict,
            total_tg_pseudobulk_path=total_tg_pseudobulk_path,
            pseudobulk_chrom_dict_path=pseudobulk_chrom_dict_path,
        )

        return total_tg_pseudobulk_global, pseudobulk_chrom_dict

    def _should_recalculate_pseudobulk(
        self,
        total_tg_pseudobulk_path: Path,
        pseudobulk_chrom_dict_path: Path,
        force_recalculate: bool,
    ) -> bool:
        return (
            force_recalculate
            or not total_tg_pseudobulk_path.is_file()
            or not pseudobulk_chrom_dict_path.is_file()
        )

    def _load_pseudobulk_aggregates(
        self,
        total_tg_pseudobulk_path: Path,
        pseudobulk_chrom_dict_path: Path,
    ) -> tuple[pd.DataFrame, dict[str, dict]]:
        total_tg_pseudobulk_global = pd.read_parquet(
            total_tg_pseudobulk_path,
            engine="pyarrow",
        )
        with open(pseudobulk_chrom_dict_path, "rb") as f:
            pseudobulk_chrom_dict = pickle.load(f)
        return total_tg_pseudobulk_global, pseudobulk_chrom_dict


    def _save_pseudobulk_aggregates(
        self,
        total_tg_pseudobulk_global: pd.DataFrame,
        pseudobulk_chrom_dict: dict[str, dict],
        total_tg_pseudobulk_path: Path,
        pseudobulk_chrom_dict_path: Path,
    ) -> None:
        total_tg_pseudobulk_global.to_parquet(
            total_tg_pseudobulk_path,
            engine="pyarrow",
            compression="snappy",
        )
        with open(pseudobulk_chrom_dict_path, "wb") as f:
            pickle.dump(pseudobulk_chrom_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


    def _load_per_sample_tg_pseudobulk(self) -> dict[str, pd.DataFrame]:
        per_sample_tg: dict[str, pd.DataFrame] = {}

        for sample_name in self.sample_names:
            tg_path = self.file_paths["processed"]["base_dir"] / sample_name / "TG_pseudobulk.parquet"
            tg_pseudobulk = pd.read_parquet(tg_path, engine="pyarrow")
            tg_pseudobulk = self._canon_index_sum(tg_pseudobulk)
            per_sample_tg[sample_name] = tg_pseudobulk

        return per_sample_tg


    def _build_chromosome_pseudobulk_dict(
        self,
        per_sample_tg: dict[str, pd.DataFrame],
    ) -> dict[str, dict]:
        pseudobulk_chrom_dict: dict[str, dict] = {}

        logging.info("  - Aggregating per-chromosome pseudobulk datasets:")
        for chrom_id in self.chrom_list:
            logging.info(f"   - Aggregating data for {chrom_id}")
            pseudobulk_chrom_dict[chrom_id] = self._aggregate_single_chromosome(
                chrom_id=chrom_id,
                per_sample_tg=per_sample_tg,
            )

        return pseudobulk_chrom_dict


    def _aggregate_single_chromosome(
        self,
        chrom_id: str,
        per_sample_tg: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        genes_on_chrom = self._get_genes_on_chromosome(chrom_id)

        tg_pseudobulk_samples = []
        re_pseudobulk_samples = []
        peaks_df_samples = []

        for sample_name in self.sample_names:
            re_pseudobulk = self._load_re_pseudobulk(sample_name)

            tg_chr_specific = per_sample_tg[sample_name].loc[
                per_sample_tg[sample_name].index.intersection(genes_on_chrom)
            ]

            re_chr_specific = self._filter_re_to_chromosome(re_pseudobulk, chrom_id)
            peaks_df = self._build_peaks_df(re_chr_specific.index)

            logging.debug(
                f"      - Sample {sample_name}, {chrom_id}: {len(re_chr_specific)} peaks matched"
            )
            if len(re_chr_specific) > 0:
                logging.debug(f"      - First few peaks: {re_chr_specific.index[:3].tolist()}")

            tg_pseudobulk_samples.append(tg_chr_specific)
            re_pseudobulk_samples.append(re_chr_specific)
            peaks_df_samples.append(peaks_df)

        return {
            "total_TG_pseudobulk_chr": self._sum_dataframes(tg_pseudobulk_samples),
            "total_RE_pseudobulk_chr": self._sum_dataframes(re_pseudobulk_samples),
            "total_peaks_df": self._first_dataframes(peaks_df_samples),
        }


    def _load_re_pseudobulk(self, sample_name: str) -> pd.DataFrame:
        re_path = self.file_paths["processed"]["base_dir"] / sample_name / "RE_pseudobulk.parquet"
        return pd.read_parquet(re_path, engine="pyarrow")


    def _get_genes_on_chromosome(self, chrom_id: str) -> list[str]:
        gene_tss_chrom = self._load_or_create_chrom_gene_tss_df(chrom_id)

        gene_tss_chrom["name"] = self.canon.canonicalize_series(
            gene_tss_chrom["name"].astype(str)
        ).fillna("")
        gene_tss_chrom = gene_tss_chrom[gene_tss_chrom["name"] != ""]
        gene_tss_chrom = gene_tss_chrom.drop_duplicates(subset=["name"], keep="first")

        return gene_tss_chrom["name"].tolist()


    def _filter_re_to_chromosome(self, re_pseudobulk: pd.DataFrame, chrom_id: str) -> pd.DataFrame:
        idx = re_pseudobulk.index.astype(str)
        mask = idx.str.startswith(f"{chrom_id}:") | idx.str.startswith(f"{chrom_id}-")
        return re_pseudobulk.loc[mask]


    def _build_peaks_df(self, peak_index: pd.Index) -> pd.DataFrame:
        peaks_df = (
            peak_index.to_series()
            .str.split("[-:]", n=2, expand=True, regex=True)
            .rename(columns={0: "chrom", 1: "start", 2: "end"})
        )
        peaks_df["start"] = peaks_df["start"].astype(int)
        peaks_df["end"] = peaks_df["end"].astype(int)
        peaks_df["peak_id"] = peak_index.astype(str)
        return peaks_df


    @staticmethod
    def _sum_dataframes(dfs) -> pd.DataFrame:
        return pd.concat(dfs).groupby(level=0).sum()


    @staticmethod
    def _first_dataframes(dfs) -> pd.DataFrame:
        return pd.concat(dfs).groupby(level=0).first()
    
    def create_chrom_files(self, total_TG_pseudobulk_global, pseudobulk_chrom_dict, force_recalculate=False):

        global_tf_tensor_path = self.file_paths["training_cache"]["tf_tensor"]
        global_tf_ids_path = self.file_paths["training_cache"]["tf_ids"]
        global_tf_names_path = self.file_paths["training_cache"]["tf_names"]
        global_metacell_path = self.file_paths["training_cache"]["metacell_names"]
        total_peak_gene_dist_file = self.file_paths["training_cache"]["peak_to_gene_dist_global"]
        common_tf_vocab_file = self.file_paths["training_cache"]["common"]["tf_vocab"]
        common_tg_vocab_file = self.file_paths["training_cache"]["common"]["tg_vocab"]

        total_peak_gene_dist_df = pd.read_parquet(total_peak_gene_dist_file, engine="pyarrow")

        # genome-wide TF expression for all metacells (columns)
        genome_wide_tf_expression = (
            total_TG_pseudobulk_global
            .reindex(self.tfs)           # ensure row order matches your TF list
            .fillna(0)
            .values.astype("float32")
        )
        tf_tensor_all = torch.from_numpy(genome_wide_tf_expression)  # [T, C]
        
        tf_vocab = self.load_or_dump_common_tf_vocab()
        tg_vocab = self.load_or_dump_common_tg_vocab()

        # align TF tensor to vocab order (and get kept names/ids)
        tf_tensor_all_aligned, tf_names_kept, tf_ids = self.align_to_vocab(
            self.tfs, tf_vocab, tf_tensor_all, label="TF"
        )
        
        # Save the TF tensor, TF IDs, TF names, and metacell names for global use across chromosomes
        torch.save(tf_tensor_all_aligned, global_tf_tensor_path)
        torch.save(torch.tensor(tf_ids, dtype=torch.long), global_tf_ids_path)
        
        self.atomic_json_dump(tf_names_kept, global_tf_names_path)
        self.atomic_json_dump(total_TG_pseudobulk_global.columns.tolist(), global_metacell_path)
        
        logging.info(f"Saved TF tensor with {len(tf_names_kept)} TFs and {tf_tensor_all_aligned.shape[1]} metacells.")
        logging.info(f"  - Number of chromosomes: {len(self.chrom_list)}: {self.chrom_list}")
        logging.info(f"  - Processing chromosomes for dataset: {self.dataset_name}")
        
        # Chromosome-specific files
        for chrom_id in self.chrom_list:
            chrom_specific_dir = self.file_paths["training_cache"][chrom_id]["dir"]
            atac_tensor_path = self.file_paths["training_cache"][chrom_id]["atac_tensor"]
            tg_tensor_path = self.file_paths["training_cache"][chrom_id]["tg_tensor"]
            dist_bias_file = self.file_paths["training_cache"][chrom_id]["dist_bias"]
            tg_id_file = self.file_paths["training_cache"][chrom_id]["tg_ids"]
            sample_tg_name_file = self.file_paths["training_cache"][chrom_id]["tg_names"]
            sample_window_map_file = self.file_paths["training_cache"][chrom_id]["window_map"]
            peak_to_tss_dist_path = self.file_paths["training_cache"][chrom_id]["peak_to_tss_dist"]
            manifest_file = self.file_paths["training_cache"][chrom_id]["manifest"]
            
            # Check if all required output files exist
            required_files = [
                global_tf_tensor_path,
                global_tf_ids_path,
                global_tf_names_path,
                global_metacell_path,
                atac_tensor_path,
                tg_tensor_path,
                sample_tg_name_file,
                sample_window_map_file,
                peak_to_tss_dist_path,
                dist_bias_file,
                tg_id_file,
                manifest_file,
            ]
            
            if not force_recalculate and all(os.path.isfile(f) for f in required_files):
                logging.info(f"  - All required output files exist for {chrom_id}. Skipping preprocessing.")
                continue
            
            chrom_specific_dir.mkdir(parents=True, exist_ok=True)
            
            logging.info(f"Creating chromosome-specific files for {chrom_id}")
            total_TG_pseudobulk_chr = pseudobulk_chrom_dict[chrom_id]["total_TG_pseudobulk_chr"]
            total_RE_pseudobulk_chr = pseudobulk_chrom_dict[chrom_id]["total_RE_pseudobulk_chr"]
            total_peaks_df = pseudobulk_chrom_dict[chrom_id]["total_peaks_df"]
            
            # Downcast the TG expression to float32
            TG_expression = total_TG_pseudobulk_chr.values.astype("float32")
            chr_tg_names = total_TG_pseudobulk_chr.index.tolist()
            
            chrom_peak_ids = set(total_peaks_df["peak_id"].astype(str))
            
            # Create genome windows
            logging.debug(f"  - Creating genomic windows for {chrom_id}")
            genome_windows = self.create_or_load_genomic_windows(
                window_size=self.window_size,
                chrom_id=chrom_id,
                chrom_sizes_file=self.file_paths["genome"]["chrom_sizes"],
                genome_window_file=self.file_paths["training_cache"][chrom_id]["genome_windows"],
                force_recalculate=force_recalculate,
                promoter_only=None
            )
            num_windows = int(genome_windows.shape[0])
            
            genes_near_peaks = total_peak_gene_dist_df[total_peak_gene_dist_df["peak_id"].astype(str).isin(chrom_peak_ids)].copy()                
            genes_near_peaks.to_parquet(peak_to_tss_dist_path, engine="pyarrow", compression="snappy")
            
            window_map = self.make_peak_to_window_map(
                peaks_bed=total_peaks_df,
                windows_bed=genome_windows,
                peaks_as_windows=False,
            )

            _, tg_tensor_all, atac_window_tensor_all = self.precompute_input_tensors(
                output_dir=str(chrom_specific_dir),
                genome_wide_tf_expression=genome_wide_tf_expression,
                genome_wide_tg_expression=TG_expression,
                total_RE_pseudobulk_chr=total_RE_pseudobulk_chr,
                window_map=window_map,
                windows=genome_windows,
            )
            
            # Skip this chromosome if no peaks matched
            if tg_tensor_all is None:
                logging.warning(f"{chrom_id}: No peaks matched between window_map and total_RE_pseudobulk_chr; skipping this chromosome.")
                continue
            
            # Match TFs and TGs to global vocab
            tg_tensor_all, tg_names_kept, tg_ids = self.align_to_vocab(chr_tg_names, tg_vocab, tg_tensor_all, label="TG")
            
            # Build distance bias [num_windows x num_tg_kept] aligned to kept TGs
            logging.debug(f"  - Building distance bias")
            dist_bias, new_window_map, kept_window_indices = self.build_distance_bias(
                genes_near_peaks=genes_near_peaks,
                window_map=window_map,
                tg_names_kept=tg_names_kept,
                num_windows=num_windows,
                dtype=torch.float32,
                mode="logsumexp",
                prune_empty_windows=True
            )
            
            if kept_window_indices is not None:
                keep_t = torch.tensor(kept_window_indices, dtype=torch.long)
                atac_window_tensor_all = atac_window_tensor_all.index_select(0, keep_t)
                genome_windows = genome_windows.iloc[kept_window_indices].reset_index(drop=True)
            else:
                kept_window_indices = list(range(num_windows))

            # ----- Writing Output Files -----
            torch.save(atac_window_tensor_all, atac_tensor_path)
            torch.save(tg_tensor_all, tg_tensor_path)
            self.atomic_json_dump(new_window_map, sample_window_map_file)
            self.atomic_json_dump(tg_names_kept, sample_tg_name_file)
            torch.save(torch.tensor(tg_ids, dtype=torch.long), tg_id_file)
            torch.save(dist_bias, dist_bias_file)
            
            # Manifest of general sample info and file paths
            manifest = {
                "dataset_name": self.dataset_name,
                "chrom": chrom_id,
                "num_windows": len(kept_window_indices),
                "num_tfs": int(len(tf_names_kept)),
                "num_tgs": int(len(tg_names_kept)),
                "Distance tau": 20000,
                "Max peak-TG distance": 100000,
                "paths": {
                    "tf_tensor_all": str(global_tf_tensor_path),
                    "tg_tensor_all": str(tg_tensor_path),
                    "atac_window_tensor_all": str(atac_tensor_path),
                    "dist_bias": str(dist_bias_file),
                    "tf_ids": str(global_tf_ids_path),
                    "tg_ids": str(tg_id_file),
                    "tf_names": str(global_tf_names_path),
                    "tg_names": str(sample_tg_name_file),
                    "common_tf_vocab": str(common_tf_vocab_file),
                    "common_tg_vocab": str(common_tg_vocab_file),
                    "window_map": str(sample_window_map_file),
                    "genes_near_peaks": str(peak_to_tss_dist_path),
                }
            }
            with open(manifest_file, "w") as f:
                json.dump(manifest, f, indent=2)

        logging.info("Preprocessing complete. Wrote per-sample/per-chrom data for MultiomicTransformerDataset.")
            
    def load_or_dump_common_tf_vocab(self):
        return self._load_or_create_vocab(
            self.file_paths["training_cache"]["common"]["tf_vocab"],
            self.tfs,
            "TF",
        )
        
    def load_or_dump_common_tg_vocab(self):
        return self._load_or_create_vocab(
            self.file_paths["training_cache"]["common"]["tg_vocab"],
            self.tgs,
            "TG",
        )
            
    def _load_or_create_vocab(self, path: Path, names: list[str], label: str) -> dict[str, int]:
        if not path.exists():
            vocab = {name: i for i, name in enumerate(names)}
            self.atomic_json_dump(vocab, path)
            logging.info(f"Initialized {label} vocab with {len(vocab)} entries")
            return vocab

        with open(path) as f:
            return json.load(f)

    def create_or_load_genomic_windows(
        self,
        window_size: int,
        chrom_id: str,
        genome_window_file: str | Path,
        chrom_sizes_file: str | Path,
        force_recalculate: bool = False,
        promoter_only: bool = False,
    ):
        """
        Create or load fixed-size genomic windows for a chromosome.

        When `promoter_only=True`, skip windows entirely and return an empty
        DataFrame with the expected schema so callers don't break.

        Parameters
        ----------
        window_size : int
            Size of genomic windows to create.
        chrom_id : str
            Chromosome identifier.
        genome_window_file : Union[str, Path]
            Path to save/load genomic windows to/from.
        chrom_sizes_file : Union[str, Path]
            Path to chromosome sizes file (e.g. UCSC chrom.sizes.txt).
        force_recalculate : bool, optional
            Whether to recompute genomic windows even if they already exist.
            Defaults to False.
        promoter_only : bool, optional
            Whether to skip creating genomic windows and return an empty DataFrame.
            Defaults to False.

        Returns
        -------
        pd.DataFrame
            DataFrame with genomic windows, with columns "chrom", "start", "end", and "win_idx".
        """
        
        # Promoter-centric evaluation: no windows needed
        if promoter_only:
            return pd.DataFrame(columns=["chrom", "start", "end", "win_idx"])

        if not os.path.exists(genome_window_file) or force_recalculate:
            genome_windows = pybedtools.bedtool.BedTool().window_maker(g=chrom_sizes_file, w=window_size)
            # Ensure consistent column names regardless of BedTool defaults
            chrom_windows = (
                genome_windows
                .filter(lambda x: x.chrom == chrom_id)
                .saveas(genome_window_file)
                .to_dataframe(names=["chrom", "start", "end"])
            )
            logging.debug(f"  - Created {chrom_windows.shape[0]} windows")
        else:
            logging.debug("\nLoading existing genomic windows")
            chrom_windows = pybedtools.BedTool(genome_window_file).to_dataframe(names=["chrom", "start", "end"])

        chrom_windows = chrom_windows.reset_index(drop=True)
        chrom_windows["win_idx"] = chrom_windows.index
        return chrom_windows
        
    def make_peak_to_window_map(self, peaks_bed: pd.DataFrame, windows_bed: pd.DataFrame, peaks_as_windows: bool = True,) -> dict[str, int]:
        """
        Map each peak to the window it overlaps the most.
        Ensures the BedTool 'name' field is exactly the `peak_id` column.
        Parameters
        ----------
        peaks_bed : pd.DataFrame
            DataFrame of peaks, with columns "chrom", "start", "end", "peak_id".
        windows_bed : pd.DataFrame
            DataFrame of genomic windows, with columns "chrom", "start", "end", "win_idx".
        peaks_as_windows : bool, optional
            Whether to treat peaks as genomic windows. Defaults to True.
        Returns
        -------
        dict[str, int]
            Mapping of peak_id to win_idx, with the peak_id as key and the win_idx as value.
        """
        
        pb = peaks_bed.copy()
        required = ["chrom", "start", "end", "peak_id"]
        missing = [c for c in required if c not in pb.columns]
        if missing:
            raise ValueError(f"peaks_bed missing columns: {missing}")

        pb["chrom"] = pb["chrom"].astype(str)
        pb["start"] = pb["start"].astype(int)
        pb["end"]   = pb["end"].astype(int)
        pb["peak_id"] = pb["peak_id"].astype(str).str.strip()
        
        if peaks_as_windows or windows_bed is None or windows_bed.empty:
            # stable order mapping: as they appear
            return {pid: int(i) for i, pid in enumerate(pb["peak_id"].tolist())}

        # BedTool uses the 4th column as "name. make it explicitly 'peak_id'
        pb_for_bed = pb[["chrom", "start", "end", "peak_id"]].rename(columns={"peak_id": "name"})
        bedtool_peaks   = pybedtools.BedTool.from_dataframe(pb_for_bed)

        # windows: enforce expected columns & dtypes
        wb = windows_bed.copy()
        wr = ["chrom", "start", "end", "win_idx"]
        miss_w = [c for c in wr if c not in wb.columns]
        if miss_w:
            raise ValueError(f"windows_bed missing columns: {miss_w}")
        wb["chrom"] = wb["chrom"].astype(str)
        wb["start"] = wb["start"].astype(int)
        wb["end"]   = wb["end"].astype(int)
        wb["win_idx"] = wb["win_idx"].astype(int)
        bedtool_windows = pybedtools.BedTool.from_dataframe(wb[["chrom", "start", "end", "win_idx"]])

        overlaps = {}
        for iv in bedtool_peaks.intersect(bedtool_windows, wa=True, wb=True):
            # left fields (peak): chrom, start, end, name
            peak_id = iv.name  # guaranteed to be the 'name' we set = peak_id
            # right fields (window): ... chrom, start, end, win_idx (as the last field)
            win_idx = int(iv.fields[-1])

            peak_start, peak_end = int(iv.start), int(iv.end)
            win_start, win_end   = int(iv.fields[-3]), int(iv.fields[-2])
            overlap_len = max(0, min(peak_end, win_end) - max(peak_start, win_start))

            if overlap_len <= 0:
                continue
            overlaps.setdefault(peak_id, []).append((overlap_len, win_idx))

        # resolve ties by max-overlap then random
        mapping = {}
        for pid, lst in overlaps.items():
            if not lst: 
                continue
            max_ol = max(lst, key=lambda x: x[0])[0]
            candidates = [w for ol, w in lst if ol == max_ol]
            mapping[str(pid)] = int(random.choice(candidates))

        return mapping
   
    def precompute_input_tensors(
        self,
        output_dir: str,
        genome_wide_tf_expression: np.ndarray,   # [num_TF, num_cells]
        genome_wide_tg_expression: np.ndarray,                   # [num_TG_chr, num_cells]
        total_RE_pseudobulk_chr: pd.DataFrame,                 # pd.DataFrame: rows=peak_id, cols=metacells
        window_map: dict,
        windows: pd.DataFrame,                            # pd.DataFrame with shape[0] = num_windows
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        """
        Builds & saves:
        - tf_tensor_all.pt                        [num_TF, num_cells]
        - tg_tensor_all_{chr}.pt                  [num_TG_chr, num_cells]
        - atac_window_tensor_all_{chr}.pt         [num_windows, num_cells]

        Returns:
        (tf_tensor_all, tg_tensor_all, atac_window_tensor_all)
        """
        os.makedirs(output_dir, exist_ok=True)

        # ---- TF tensor ----
        tf_tensor_all = torch.as_tensor(
            np.asarray(genome_wide_tf_expression, dtype=np.float32), dtype=dtype
        )

        # ---- TG tensor ----
        tg_tensor_all = torch.as_tensor(
            np.asarray(genome_wide_tg_expression, dtype=np.float32), dtype=dtype
        )

        total_RE_pseudobulk_chr.index = (
            total_RE_pseudobulk_chr.index.astype(str).str.strip()
        )

        # ---- ATAC window tensor ----
        num_windows = int(windows.shape[0])
        num_peaks   = int(total_RE_pseudobulk_chr.shape[0])
        
        logging.debug(f"  - precompute_input_tensors: {num_windows} windows, {num_peaks} peaks")
        logging.debug(f"  - window_map has {len(window_map)} entries")
        if len(window_map) > 0:
            sample_peaks = list(window_map.keys())[:3]
            logging.debug(f"  - Sample peak IDs from window_map: {sample_peaks}")
        if num_peaks > 0:
            sample_re_peaks = total_RE_pseudobulk_chr.index[:3].tolist()
            logging.debug(f"  - Sample peak IDs from RE_pseudobulk: {sample_re_peaks}")

        rows, cols, vals = [], [], []
        peak_to_idx = {p: i for i, p in enumerate(total_RE_pseudobulk_chr.index)}
        for peak_id, win_idx in window_map.items():
            peak_idx = peak_to_idx.get(peak_id)
            if peak_idx is not None and 0 <= win_idx < num_windows:
                rows.append(win_idx)
                cols.append(peak_idx)
                vals.append(1.0)

        if not rows:
            logging.warning("No peaks from window_map matched rows in total_RE_pseudobulk_chr. Returning None.")
            logging.warning(f"  - Checked {len(window_map)} window_map entries against {num_peaks} RE peaks")
            return None, None, None

        W = sp.csr_matrix((vals, (rows, cols)), shape=(num_windows, num_peaks))
        atac_window = W @ total_RE_pseudobulk_chr.values  # [num_windows, num_cells]

        atac_window_tensor_all = torch.as_tensor(
            atac_window.astype(np.float32), dtype=dtype
        )

        return tf_tensor_all, tg_tensor_all, atac_window_tensor_all
                
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
    
    def build_distance_bias(
        self,
        genes_near_peaks: pd.DataFrame,
        window_map: Dict[str, int],
        tg_names_kept: Iterable[str],
        num_windows: int,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        mode: str = "logsumexp",   # "max" | "sum" | "mean" | "logsumexp"
        prune_empty_windows: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, int], List[int]]]:
        """
        Build a [num_windows x num_tg_kept] (or pruned) distance-bias tensor aligned to the kept TGs.

        If prune_empty_windows is True, windows with no peaks (no entries in genes_near_peaks)
        are removed and indices are compacted.

        Returns:
            If prune_empty_windows is False:
                dist_bias
            If True:
                dist_bias, new_window_map, kept_window_indices

            where:
                - dist_bias: [num_kept_windows, num_tg_kept]
                - new_window_map: peak_id -> new_window_idx (only for kept windows)
                - kept_window_indices: list mapping new_window_idx -> original_window_idx
        """
        tg_names_kept = list(tg_names_kept)
        num_tg_kept = len(tg_names_kept)
        tg_index_map = {tg: i for i, tg in enumerate(tg_names_kept)}

        from collections import defaultdict
        scores_map = defaultdict(list)

        # Collect all scores for each (window, TG)
        for _, row in genes_near_peaks.iterrows():
            win_idx = window_map.get(row["peak_id"])
            tg_idx  = tg_index_map.get(row["target_id"])
            if win_idx is not None and tg_idx is not None:
                scores_map[(win_idx, tg_idx)].append(float(row["TSS_dist_score"]))

        # If not pruning, keep all windows regardless of whether they have peaks
        if not prune_empty_windows:
            dist_bias = torch.zeros((num_windows, num_tg_kept), dtype=dtype, device=device)
            for (win_idx, tg_idx), scores in scores_map.items():
                scores_tensor = torch.tensor(scores, dtype=dtype, device=device)
                if mode == "max":
                    dist_bias[win_idx, tg_idx] = scores_tensor.max()
                elif mode == "sum":
                    dist_bias[win_idx, tg_idx] = scores_tensor.sum()
                elif mode == "mean":
                    dist_bias[win_idx, tg_idx] = scores_tensor.mean()
                elif mode == "logsumexp":
                    dist_bias[win_idx, tg_idx] = torch.logsumexp(scores_tensor, dim=0)
                else:
                    raise ValueError(f"Unknown pooling mode: {mode}")
            return dist_bias

        # --- Pruned version: only keep windows that appear in scores_map ---
        used_windows = sorted({win for (win, _) in scores_map.keys()})
        num_kept = len(used_windows)
        old2new = {w: i for i, w in enumerate(used_windows)}

        dist_bias = torch.zeros((num_kept, num_tg_kept), dtype=dtype, device=device)

        for (win_idx, tg_idx), scores in scores_map.items():
            new_win_idx = old2new[win_idx]
            scores_tensor = torch.tensor(scores, dtype=dtype, device=device)

            if mode == "max":
                dist_bias[new_win_idx, tg_idx] = scores_tensor.max()
            elif mode == "sum":
                dist_bias[new_win_idx, tg_idx] = scores_tensor.sum()
            elif mode == "mean":
                dist_bias[new_win_idx, tg_idx] = scores_tensor.mean()
            elif mode == "logsumexp":
                dist_bias[new_win_idx, tg_idx] = torch.logsumexp(scores_tensor, dim=0)
            else:
                raise ValueError(f"Unknown pooling mode: {mode}")

        # Build a new window_map in the compressed index space
        new_window_map: Dict[str, int] = {}
        used_set = set(used_windows)
        for peak_id, old_idx in window_map.items():
            if old_idx in used_set:
                new_window_map[peak_id] = old2new[old_idx]

        kept_window_indices = used_windows

        return dist_bias, new_window_map, kept_window_indices
    
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
    
    def build_global_tg_vocab(self, dataset_tgs):
        """
        Builds a global TG vocab from the TSS file with contiguous IDs [0..N-1],
        ordered by chromosome then genomic start position.
        """
        gene_tss_file = self.file_paths["genome"]["gene_tss"]
        vocab_file = self.file_paths["training_cache"]["common"]["tg_vocab"]

        gene_tss_bed = pybedtools.BedTool(gene_tss_file)
        gene_tss_df = gene_tss_bed.to_dataframe().sort_values(by="start", ascending=True)

        # 2) Canonical symbol list (MUST match downstream normalization)
        gene_tss_df["name"] = self.canon.canonicalize_series(gene_tss_df["name"])
        tss_genes = set(gene_tss_df["name"])

        if dataset_tgs is not None:
            dataset_tgs = {self.canon.canonical_symbol(g) for g in dataset_tgs}
            names = sorted(tss_genes & dataset_tgs)
        else:
            names = sorted(tss_genes)
        logging.info(f"  - Writing global TG vocab with {len(names)} genes")

        # 3) Build fresh contiguous mapping
        vocab = {name: i for i, name in enumerate(names)}

        # 4) Atomic overwrite
        tmp = str(vocab_file) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(vocab, f)
        os.replace(tmp, vocab_file)

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
            "peak_to_gene_dist_global": sample_cache_dir / "peak_to_gene_dist_global.parquet",
        }
        
        if not self.file_paths["training_cache"]["common"]["dir"].is_dir():
            os.makedirs(self.file_paths["training_cache"]["common"]["dir"])
        

        # ----- TF-TG COMBO FILES -----
        tf_tg_combos_dir = self.processed_data_dir / "tf_tg_combos"
        
        if not tf_tg_combos_dir.is_dir():
            os.makedirs(tf_tg_combos_dir)

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
            genome_window_file: Path =          chrom_cache_dir / f"{chrom_id}_windows.bed"
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
                chrom_cache_dir / f"{chrom_id}_windows.bed",
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
            return missing_files
        else:
            logging.info("All required files are present.")
            return missing_files
    
    def split_genes_into_tfs_and_tgs(self, genes):
        """Creates TF and TG combo files based on the provided gene list and the reference TF list."""
        
        tf_name_col = "TF_Name"
        
        # --- normalize incoming genes ---
        genes_norm = sorted({self._canon(g) for g in genes if pd.notna(g)})

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

        known_tfs = {self._canon(x) for x in tf_ref[tf_col].dropna().astype(str).tolist()}

        # --- new sets from this call ---
        tfs_new = sorted(set(genes_norm) & known_tfs)
        tgs_new = sorted(set(genes_norm) - set(tfs_new))

        total_file = self.file_paths["processed"]["tf_tg_combos"]["total_genes"]
        tf_file    = self.file_paths["processed"]["tf_tg_combos"]["tf_list"]
        tg_file    = self.file_paths["processed"]["tf_tg_combos"]["tg_list"]

        total_existing = self._read_list(total_file, "Gene")
        tf_existing    = self._read_list(tf_file, "TF")
        tg_existing    = self._read_list(tg_file, "TG")

        total = sorted(set(total_existing) | set(genes_norm))
        tfs   = sorted(set(tf_existing)    | set(tfs_new))
        tgs   = sorted((set(tg_existing) | set(tgs_new)) - set(tfs))
        
        tfs = sorted(set(self.canon.canonicalize_series(pd.Series(tfs)).tolist()))
        tgs = sorted(set(self.canon.canonicalize_series(pd.Series(tgs)).tolist()))

        pd.DataFrame({"Gene": total}).to_csv(total_file, index=False)
        pd.DataFrame({"TF": tfs}).to_csv(tf_file, index=False)
        pd.DataFrame({"TG": tgs}).to_csv(tg_file, index=False)

        return tfs, tgs
    
    def create_peak_bed_file(self, atac_df, sample_name):
        self.peak_locs_df = format_peaks(pd.Series(atac_df.index)).rename(columns={"chromosome": "chrom"})
        peak_bed_file = self.file_paths["samples"][sample_name]["peaks"]
        
        if not os.path.isfile(peak_bed_file):
            # Write the peak BED file
            peak_bed_file.parent.mkdir(parents=True, exist_ok=True)
            pybedtools.BedTool.from_dataframe(
                self.peak_locs_df[["chrom", "start", "end", "peak_id"]]
            ).saveas(peak_bed_file)
    
    def _aggregate_peak_to_gene_dist(self):
        total_peak_gene_dist_file = self.file_paths["training_cache"]["peak_to_gene_dist_global"]
        sample_level_peak_to_gene_dist_dfs = []
        for sample_name in self.sample_names:
            peak_to_gene_dist_file = self.file_paths["samples"][sample_name]["peak_to_gene_dist"]
            
            peak_to_gene_dist_df = pd.read_parquet(peak_to_gene_dist_file, engine="pyarrow")
            sample_level_peak_to_gene_dist_dfs.append(peak_to_gene_dist_df)
            
        total_peak_gene_dist_df = pd.concat(sample_level_peak_to_gene_dist_dfs)
        total_peak_gene_dist_df.to_parquet(total_peak_gene_dist_file, engine="pyarrow", compression="snappy")
    
    def _read_list(self, path: Path, col: str) -> list[str]:
        """
        Reads a list of elements from a CSV file.
        """
        if path.is_file():
            df = pd.read_csv(path)
            if col not in df.columns and df.shape[1] == 1:
                # tolerate unnamed single column
                return sorted({self._canon(v) for v in df.iloc[:, 0].astype(str)})
            return sorted({self._canon(v) for v in df[col].dropna().astype(str)})
        return []
    
    def _canon(self, x: str) -> str:
        """
        Strips version suffix and uppercases a given string.
        """
        
        # strip version suffix and uppercase
        s = str(x).strip()
        s = re.sub(r"\.\d+$", "", s)
        return s.upper()
    
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
        chrom_tss_path = self.file_paths["genome"]["genome_dir"] / f"{chrom_id}_gene_tss.bed"

        if chrom_tss_path.is_file():
            df = pd.read_csv(chrom_tss_path, sep="\t", header=None, usecols=[0, 1, 2, 3])
            df = df.rename(columns={0: "chrom", 1: "start", 2: "end", 3: "name"})
            df = df.drop_duplicates(subset=["name"], keep="first")
            return df

        gene_tss_bed = pybedtools.BedTool(self.file_paths["genome"]["gene_tss"])
        df = (
            gene_tss_bed
            .filter(lambda x: x.chrom == chrom_id)
            .saveas(str(chrom_tss_path))
            .to_dataframe()
            .sort_values("start")
            .drop_duplicates(subset=["name"], keep="first")
        )
        
        return df

    def _handle_assertions(self):
        tf_file = self.file_paths["genome"]["tf_info"]
        
        assert self.project_dir is not None, "Project directory is not specified"
        assert self.organism_code is not None, "Organism code is not specified. Please specify 'mm10' or 'hg38'"
        assert self.organism_code in ["mm10", "hg38"], f"Unsupported organism code: {self.organism_code}"
        assert tf_file.is_file(), f"TF information file not found: {tf_file}"
        assert self.processed_data_dir.is_dir(), f"Processed data directory not found: {self.processed_data_dir}"