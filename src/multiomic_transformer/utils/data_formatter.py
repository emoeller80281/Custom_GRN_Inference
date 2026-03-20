from pathlib import Path

import pybedtools
from multiomic_transformer.utils.gene_canonicalizer import GeneCanonicalizer
from multiomic_transformer.utils.downloads import *

import json
import logging
import os
import re
import numpy as np

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

        names = sorted(tss_genes & self.tgs)

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

    def _handle_assertions(self):
        tf_file = self.file_paths["genome"]["tf_info"]
        
        assert self.project_dir is not None, "Project directory is not specified"
        assert self.organism_code is not None, "Organism code is not specified. Please specify 'mm10' or 'hg38'"
        assert self.organism_code in ["mm10", "hg38"], f"Unsupported organism code: {self.organism_code}"
        assert tf_file.is_file(), f"TF information file not found: {tf_file}"
        assert self.processed_data_dir.is_dir(), f"Processed data directory not found: {self.processed_data_dir}"