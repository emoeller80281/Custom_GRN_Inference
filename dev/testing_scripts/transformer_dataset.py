import os
import torch
import torch.distributed as dist
import numpy as np
import pandas as pd
import pybedtools
from grn_inference import utils
from typing import Optional
from scipy import sparse
import logging

class TransformerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        atac_data_filename: str,
        rna_data_filename: str,
        chrom_id: str,
        num_cells: int,
        window_size: int,
        genome_dir: str,
        gene_tss_filepath: str,
        chrom_size_filepath: str,
        output_dir: str,
        force_recalculate: Optional[bool] = False,
        ):
        self.atac_data_filename = atac_data_filename
        self.rna_data_filename = rna_data_filename
        self.chrom_id = chrom_id
        self.num_cells = num_cells
        self.window_size = window_size
        self.genome_dir = genome_dir
        self.gene_tss_filepath = gene_tss_filepath
        self.chrom_size_filepath = chrom_size_filepath
        self.output_dir = output_dir
        self.force_recalculate = force_recalculate
        
        mesc_atac_data, mesc_atac_peak_loc_df = self._load_atac_dataset()
        mesc_rna_data = self._load_rna_data()

        gene_tss_df = self._load_or_create_gene_tss_df()

        logging.info(f"Using {gene_tss_df['name'].nunique()} genes (TSS df)")
        
        # Restrict gene TSS dataframe to only use the selected chromosome
        gene_tss_df = gene_tss_df[gene_tss_df["chrom"] == self.chrom_id]
        
        mesc_rna_data = mesc_rna_data[mesc_rna_data.index.isin(gene_tss_df["name"])]
        logging.info(f"Using {mesc_rna_data.index.nunique()} genes (scRNA-seq data df)")
        
        mm10_windows = self._create_or_load_genomic_windows()

        genes_near_peaks = self._calculate_peak_to_tg_distance_score(mesc_atac_peak_loc_df, gene_tss_df)
        
        # Restrict the genes near peaks dataframe to only using TGs from genes on one chromosome
        genes_near_peaks = genes_near_peaks[(genes_near_peaks["gene_chr"] == self.chrom_id) & (genes_near_peaks["peak_chr"] == self.chrom_id)]
        logging.info(f"Using {genes_near_peaks['target_id'].nunique()} genes (genes_near_peaks_df)")

        if not os.path.isfile(os.path.join(self.output_dir, "tmp/homer_peaks.txt")):
            self._create_homer_peaks_file(genes_near_peaks)
            
        homer_results = self._load_homer_tf_to_peak_results()


        tfs, peaks, genes = self._get_unique_tfs_peaks_genes(homer_results, genes_near_peaks)
        tf_i, peak_i, gene_i = self._create_tf_peak_gene_mapping_dicts(tfs, peaks, genes)
        
        logging.info("Preparing sparse components (TF×Peak and Peak×Gene)")
        self.homer_tf_peak_sparse = self._cast_homer_tf_to_peak_df_sparse(homer_results, tf_i, peak_i)
        self.gene_distance_sparse = self._cast_peak_to_tg_distance_sparse(genes_near_peaks, peak_i, gene_i)
        self.peaks_by_window = self._assign_peaks_to_windows(mesc_atac_peak_loc_df, peaks, peak_i, mm10_windows)

        self.gene_ids = torch.arange(len(genes))
        
        self.shared_barcodes, mesc_atac_data_shared, mesc_rna_data_shared = self._find_shared_barcodes(mesc_atac_data, mesc_rna_data)

        rna_tfs  = mesc_rna_data_shared.reindex(index=tfs).fillna(0).astype("float32")          # rows: TFs, cols: cells
        atac_peaks  = mesc_atac_data_shared.reindex(index=peaks).fillna(0).astype("float32")   # rows: peaks, cols: cells
        
        assert set(rna_tfs.columns) == set(atac_peaks.columns), "RNA/ATAC barcode sets differ"
        
        self.tf_expr_arr  = rna_tfs.values          # shape [TF, n_cells]
        self.peak_acc_arr = atac_peaks.values          # shape [P,  n_cells]
        self.cell_col_idx   = {c:i for i,c in enumerate(rna_tfs.columns)}
        
    def __getitem__(self, cell_idx):
        cell_col = self.cell_col_idx[cell_idx]
        
        tf_expr = self.tf_expr_arr[:, cell_col]
        peak_acc = self.peak_acc_arr[:, cell_col]
        
        tf_peak_binding = self.homer_tf_peak_sparse.multiply(tf_expr[:, None]).tocsr()
        
        tf_windows = []
        gene_biases = []
        for window_peak_idx in self.peaks_by_window:
            active = window_peak_idx[peak_acc[window_peak_idx] > 0]
            if active.size == 0:
                continue

            w = peak_acc[active]
            tf_win = (tf_peak_binding[:, active] @ w).astype(np.float32)         # [TF]
            
            # Regularize
            m, s = tf_win.mean(), tf_win.std() + 1e-6
            tf_win = np.clip((tf_win - m) / s, -3.0, 3.0)

            # gene×window bias (parameter-free)
            pg_sub = self.gene_distance_sparse[active, :]  # [|p|, G]
            bias   = np.asarray(pg_sub.max(axis=0)).ravel().astype(np.float32)    # [G]

            tf_windows.append(tf_win)
            gene_biases.append(bias)

        return {
            "cell_id": cell_idx,
            "tf_windows": tf_windows,        # List[np.ndarray [TF]]
            "gene_biases": gene_biases,      # List[np.ndarray [G]]
        }
            
    def collate_fn(batch, pad_to_max=True, device="cuda"):
        # ragged -> padded tensors
        B = len(batch)
        Wlens = [len(b["tf_windows"]) for b in batch]
        Wmax = max(Wlens) if pad_to_max else None
        TF  = batch[0]["tf_windows"][0].shape[0] if Wmax else None
        G   = batch[0]["gene_biases"][0].shape[0] if Wmax else None

        if pad_to_max:
            tf_pad   = torch.zeros((B, Wmax, TF), dtype=torch.float32)
            bias_pad = torch.zeros((B, Wmax, G),  dtype=torch.float32)
            kpm      = torch.ones((B, Wmax),      dtype=torch.bool)  # True=PAD

            for i, sample in enumerate(batch):
                wlen = len(sample["tf_windows"])
                if wlen == 0: continue
                tf_stack   = torch.from_numpy(np.stack(sample["tf_windows"], axis=0))   # [W', TF]
                bias_stack = torch.from_numpy(np.stack(sample["gene_biases"], axis=0))  # [W', G]
                tf_pad[i, :wlen]   = tf_stack
                bias_pad[i, :wlen] = bias_stack
                kpm[i, :wlen] = False

            return (
                tf_pad.to(device, non_blocking=True),      # [B, Wmax, TF]
                bias_pad.to(device, non_blocking=True),    # [B, Wmax, G]
                kpm.to(device, non_blocking=True),         # [B, Wmax]
            )
        else:
            # no padding path (variable lengths)
            tf_list   = [torch.from_numpy(np.stack(s["tf_windows"])).to(device) for s in batch]
            bias_list = [torch.from_numpy(np.stack(s["gene_biases"])).to(device) for s in batch]
            return tf_list, None, bias_list
    
    def __len__(self):
        return len(self.shared_barcodes)

    def _load_or_create_gene_tss_df(self):
        gene_tss_outfile = os.path.join(self.genome_dir, "mm10_ch1_gene_tss.bed")
        if not os.path.isfile(gene_tss_outfile) or self.force_recalculate:
            gene_tss_bed = pybedtools.BedTool(self.gene_tss_filepath)
            
            gene_tss_df = (
                gene_tss_bed
                .filter(lambda x: x.chrom == self.chrom_id)
                .saveas(gene_tss_outfile)
                .to_dataframe()
                .sort_values(by="start", ascending=True)
                )
        else:
            gene_tss_df = pybedtools.BedTool(gene_tss_outfile).to_dataframe().sort_values(by="start", ascending=True)
            
        return gene_tss_df

    def _load_atac_dataset(self):
        mesc_atac_data = pd.read_parquet(self.atac_data_filename).set_index("peak_id")
        mesc_atac_peak_loc = mesc_atac_data.index

        # format the peaks to be in bed_format
        mesc_atac_peak_loc_df = utils.format_peaks(mesc_atac_peak_loc)
        mesc_atac_peak_loc_df = mesc_atac_peak_loc_df[mesc_atac_peak_loc_df["chromosome"] == self.chrom_id]
        mesc_atac_peak_loc_df = mesc_atac_peak_loc_df.rename(columns={"chromosome":"chrom"})

        # TEMPORARY Restrict to one chromosome for testing
        mesc_atac_data = mesc_atac_data[mesc_atac_data.index.isin(mesc_atac_peak_loc_df.peak_id)]
        
        return mesc_atac_data, mesc_atac_peak_loc_df

    def _load_rna_data(self):
        logging.info("Reading in the scRNA-seq dataset")
        mesc_rna_data = pd.read_parquet(self.rna_data_filename).set_index("gene_id")
        return mesc_rna_data

    def _create_or_load_genomic_windows(self, organism_code: str = "mm10"):
        genome_window_file = os.path.join(self.genome_dir, f"{organism_code}_{self.chrom_id}_windows_{self.window_size // 1000}kb.bed")
        if not os.path.exists(genome_window_file) or self.force_recalculate:
            
            logging.info("Creating genomic windows")
            genome_windows = pybedtools.bedtool.BedTool().window_maker(g=self.chrom_size_filepath, w=self.window_size)
            windows_df = (
                genome_windows
                .filter(lambda x: x.chrom == self.chrom_id)  # TEMPORARY Restrict to one chromosome for testing
                .saveas(genome_window_file)
                .to_dataframe()
            )
        else:
            
            logging.info("Loading existing genomic windows")
            windows_df = pybedtools.BedTool(genome_window_file).to_dataframe()
            
        return windows_df

    def _calculate_peak_to_tg_distance_score(self, mesc_atac_peak_loc_df, gene_tss_df):
        if not os.path.isfile(os.path.join(self.output_dir, "genes_near_peaks.parquet")) or self.force_recalculate:
            if "peak_tmp.bed" not in os.listdir(self.output_dir) or "tss_tmp.bed" not in os.listdir(self.output_dir) or self.force_recalculate:
            
                logging.info("Calculating peak to TG distance score")
                peak_bed = pybedtools.BedTool.from_dataframe(
                    mesc_atac_peak_loc_df[["chrom", "start", "end", "peak_id"]]
                    ).saveas(os.path.join(self.output_dir, "peak_tmp.bed"))

                tss_bed = pybedtools.BedTool.from_dataframe(
                    gene_tss_df[["chrom", "start", "end", "name"]]
                    ).saveas(os.path.join(self.output_dir, "tss_tmp.bed"))
                
            peak_bed = pybedtools.BedTool(os.path.join(self.output_dir, "peak_tmp.bed"))
            tss_bed = pybedtools.BedTool(os.path.join(self.output_dir, "tss_tmp.bed"))
        

            genes_near_peaks = utils.find_genes_near_peaks(peak_bed, tss_bed)

            # Restrict to peaks within 1 Mb of a gene TSS
            genes_near_peaks = genes_near_peaks[genes_near_peaks["TSS_dist"] <= 1e6]

            # Scale the TSS distance score by the exponential scaling factor
            genes_near_peaks = genes_near_peaks.copy()
            genes_near_peaks["TSS_dist_score"] = np.exp(-genes_near_peaks["TSS_dist"] / 250000)

            genes_near_peaks.to_parquet(os.path.join(self.output_dir, "genes_near_peaks.parquet"), compression="snappy", engine="pyarrow")
        else:
            genes_near_peaks = pd.read_parquet(os.path.join(self.output_dir, "genes_near_peaks.parquet"), engine="pyarrow")
        
        return genes_near_peaks

    def _create_homer_peaks_file(self, genes_near_peaks):
        logging.info("Building Homer peaks file")
        homer_peaks = genes_near_peaks[["peak_id", "peak_chr", "peak_start", "peak_end"]]
        homer_peaks = homer_peaks.rename(columns={
            "peak_id":"PeakID", 
            "peak_chr":"chr",
            "peak_start":"start",
            "peak_end":"end"
            })
        homer_peaks["strand"] = ["."] * len(homer_peaks)
        homer_peaks["start"] = round(homer_peaks["start"].astype(int),0)
        homer_peaks["end"] = round(homer_peaks["end"].astype(int),0)
        homer_peaks = homer_peaks.drop_duplicates(subset="PeakID")

        os.makedirs(os.path.join(self.output_dir, "tmp"), exist_ok=True)
        homer_peak_path = os.path.join(self.output_dir, "tmp/homer_peaks.txt")
        homer_peaks.to_csv(homer_peak_path, sep="\t", header=False, index=False)

    def _load_homer_tf_to_peak_results(self):
        assert os.path.exists(os.path.join(self.output_dir, "homer_tf_to_peak.parquet")), \
            "ERROR: Homer TF to peak output parquet file required"
            
        homer_results = pd.read_parquet(os.path.join(self.output_dir, "homer_tf_to_peak.parquet"), engine="pyarrow")
        homer_results = homer_results.reset_index(drop=True)
        homer_results["source_id"] = homer_results["source_id"].str.capitalize()
        
        return homer_results

    def _find_shared_barcodes(self, atac_df, rna_df):
        atac_cell_barcodes = atac_df.columns.to_list()
        rna_cell_barcodes = rna_df.columns.to_list()
        atac_in_rna_shared_barcodes = [i for i in atac_cell_barcodes if i in rna_cell_barcodes]

        # Make sure that the cell names are in the same order and in both datasets
        shared_barcodes = sorted(set(atac_in_rna_shared_barcodes))[:self.num_cells]

        atac_df_shared = atac_df[shared_barcodes]
        rna_df_shared = rna_df[shared_barcodes]
        
        return shared_barcodes, atac_df_shared, rna_df_shared

    def _get_unique_tfs_peaks_genes(self, homer_results, genes_near_peaks):  
        tfs   = homer_results["source_id"].astype(str).str.capitalize().unique()
        
        peaks = np.unique(np.concatenate([
            homer_results["peak_id"].astype(str).values,
            genes_near_peaks["peak_id"].astype(str).values
        ]))
        
        genes = genes_near_peaks["target_id"].astype(str).unique()
        
        return tfs, peaks, genes

    def _create_tf_peak_gene_mapping_dicts(self, tfs, peaks, genes):
        tf_i   = {t:i for i,t in enumerate(tfs)}
        peak_i = {p:i for i,p in enumerate(peaks)}
        gene_i = {g:i for i,g in enumerate(genes)}
        
        return tf_i, peak_i, gene_i

    def _cast_homer_tf_to_peak_df_sparse(self, homer_results, tf_i, peak_i):
        homer_results = homer_results[["source_id","peak_id","homer_binding_score"]].copy()
        homer_results["source_id"] = homer_results["source_id"].astype(str).str.capitalize().map(tf_i)
        homer_results["peak_id"]   = homer_results["peak_id"].astype(str).map(peak_i)
        homer_results = homer_results.dropna(subset=["source_id","peak_id"])
        homer_tf_peak_sparse = sparse.coo_matrix(
            (homer_results["homer_binding_score"].astype(np.float32).values,
            (homer_results["source_id"].astype(int).values, homer_results["peak_id"].astype(int).values)),
            shape=(len(tf_i), len(peak_i))
        ).tocsr()

        return homer_tf_peak_sparse

    def _cast_peak_to_tg_distance_sparse(self, genes_near_peaks, peak_i, gene_i):
        genes_near_peaks = genes_near_peaks[["peak_id","target_id","TSS_dist_score"]].copy()
        genes_near_peaks["peak_id"]   = genes_near_peaks["peak_id"].astype(str).map(peak_i)
        genes_near_peaks["target_id"] = genes_near_peaks["target_id"].astype(str).map(gene_i)
        genes_near_peaks = genes_near_peaks.dropna(subset=["peak_id","target_id"])
        gene_distance_sparse = sparse.coo_matrix(
            (genes_near_peaks["TSS_dist_score"].astype(np.float32).values,
            (genes_near_peaks["peak_id"].astype(int).values, genes_near_peaks["target_id"].astype(int).values)),
            shape=(len(peak_i), len(gene_i))
        ).tocsr()
        
        return gene_distance_sparse

    def _assign_peaks_to_windows(self, mesc_atac_peak_loc_df, peaks, peak_i, windows_df):
        logging.info("Assigning peaks to windows")

        # Make a Series with peak start/end by peak_id
        peak_coord_df = mesc_atac_peak_loc_df.loc[mesc_atac_peak_loc_df["peak_id"].isin(peaks), ["peak_id","chrom","start","end"]].copy()
        peak_coord_df = peak_coord_df[peak_coord_df["chrom"] == self.chrom_id]
        coord_map = peak_coord_df.set_index("peak_id")[["start","end"]].to_dict(orient="index")

        w = int((windows_df["end"] - windows_df["start"]).mode().iloc[0])
        win_lut = {}  # window_idx -> window_id string
        for _, row in windows_df.iterrows():
            k = row["start"] // w
            win_lut[k] = f'{row["chrom"]}:{row["start"]}-{row["end"]}'
        nW = len(win_lut)

        rng = np.random.default_rng(0)
        def __assign_best_window(start, end, w):
            i0 = start // w
            i1 = (end - 1) // w
            if i1 < i0: i1 = i0
            best_k = i0
            best_ov = -1
            ties = []
            for k in range(i0, i1 + 1):
                bs, be = k * w, (k + 1) * w
                ov = max(0, min(end, be) - max(start, bs))
                if ov > best_ov:
                    best_ov, best_k = ov, k
                    ties = [k]
                elif ov == best_ov:
                    ties.append(k)
            return best_k if len(ties) == 1 else int(rng.choice(ties))

        # map each peak to a window_idx (or -1 if we lack coords)
        peak_to_window_idx = np.full(len(peaks), -1, dtype=np.int32)
        for p, idx in peak_i.items():
            info = coord_map.get(p)
            if info is None: 
                continue
            peak_to_window_idx[idx] = __assign_best_window(int(info["start"]), int(info["end"]), w)

        # build list of peak indices per window
        peaks_by_window = [np.where(peak_to_window_idx == k)[0] for k in range(nW)]
        
        return peaks_by_window

    def _build_train_gene_stats(self, rna_data_shared_barcodes, train_cells, genes, device):
        n_genes = len(genes)
        sum_y   = torch.zeros(n_genes, dtype=torch.float32)
        sum_y2  = torch.zeros(n_genes, dtype=torch.float32)
        count   = torch.zeros(n_genes, dtype=torch.int32)
        
        def __get_true_tg_expr_vector_from_data(rna_data, genes):
            dup = pd.Index(genes).duplicated()
            assert not dup.any(), f"Duplicate gene IDs in prediction axis at: {np.where(dup)[0][:10]}"

            # Align counts to prediction order from the gene to index mapping (same length and order as genes)
            true_counts = rna_data.reindex(genes)

            # build mask for missing genes (not present in RNA)
            mask = ~true_counts.isna().to_numpy()

            # Handle missing genes using a masked loss 
            y_true_vec = true_counts.to_numpy(dtype=float)        # shape (n_genes,)

            y_true = torch.tensor(y_true_vec, dtype=torch.float32).unsqueeze(0)   # [1, n_genes]
            mask_t = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)            # [1, n_genes]
        
            return y_true, mask_t

        for c in train_cells:                     # use *global* train_cells, not sharded
            y, m = __get_true_tg_expr_vector_from_data(rna_data_shared_barcodes[c], genes)
            y = y.squeeze(0).to(torch.float32)    # [n_genes]
            m = m.squeeze(0)                      # [n_genes], bool
            sum_y  += torch.where(m, y, 0).cpu()
            sum_y2 += torch.where(m, y*y, 0).cpu()
            count  += m.to(torch.int32).cpu()

        count_f = count.to(torch.float32)
        mu = sum_y / torch.clamp(count_f, min=1.0)           # [n_genes]
        var = sum_y2 / torch.clamp(count_f, min=1.0) - mu*mu
        var = torch.clamp(var, min=0.0)
        sd  = torch.sqrt(var)

        # For genes never observed in training, make them neutral so they don't explode the loss
        never_seen = (count == 0)
        mu[never_seen] = 0.0
        sd[never_seen] = 1.0

        seen = (count > 0).sum().item()

        # Keep ggenes that are seen in the training dataset
        seen_genes_mask = (count.to(torch.int32) > 0).to(device)  # [n_genes], True for 532 seen genes
        mu = mu.to(device); sd = sd.to(device)
        sd = torch.clamp(sd, min=1e-6)

        seen = seen_genes_mask.to(device)  # genes seen in TRAIN only

        # Pack to a single tensor for convenient broadcast
        stats = torch.stack([mu, sd], dim=0)      # [2, n_genes]

        # Move to device and broadcast
        stats = stats.to(device)
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(stats, src=0)

        mu, sd = stats[0], stats[1]                   # both [n_genes], on device

        # after you compute mu, sd from TRAIN only
        mu = mu.to(device)
        sd = torch.clamp(sd.to(device), min=1e-2)
        
        return mu, sd, seen_genes_mask
