import logging

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
)
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import wandb
import time

class TFTGRegulationModel(nn.Module):
    def __init__(
        self,
        pretrained_tf_peak_model,
        d_model,
        num_heads=4,
        dropout=0.1,
        tf_peak_chunk_size=256,
        keep_tf_peak_model_in_eval=False,
    ):
        super().__init__()

        self.tf_peak_model = pretrained_tf_peak_model
        self.tf_peak_chunk_size = tf_peak_chunk_size

        # Optional device-resident TF embedding table, populated by
        # set_tf_embedding_table(). Registered non-persistent so it never enters
        # state_dict -- checkpoints keep exactly the keys they already have, and
        # load_state_dict(strict=True) is unaffected in both directions.
        self.register_buffer("tf_embedding_table", None, persistent=False)
        self.register_buffer("tf_mask_table", None, persistent=False)
        # Real (unpadded) protein length per TF, used to crop chunks -- see forward().
        self.register_buffer("tf_length_table", None, persistent=False)

        # See train() below. Default False preserves the behaviour every existing
        # checkpoint was trained under.
        self.keep_tf_peak_model_in_eval = keep_tf_peak_model_in_eval

        self.peak_feature_proj = nn.Sequential(
            nn.Linear(4, d_model),  # binding, accessibility, distance_scaled, distance_weight
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.tf_expr_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.tg_expr_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.tg_query_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.peak_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(d_model)

        # peak_context + tf_expr + tg_expr
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    @staticmethod
    def pool_cell_logits(
        cell_logits,
        cell_mask=None,
        mode="lse",
        temperature=1.0,
    ):
        """
        cell_logits: [E, C]
        cell_mask: [E, C], True for real cells, False for padded cells
        """

        if cell_mask is None:
            cell_mask = torch.ones_like(cell_logits, dtype=torch.bool)

        if mode == "mean":
            masked_logits = cell_logits.masked_fill(~cell_mask, 0.0)
            denom = cell_mask.sum(dim=1).clamp_min(1)
            return masked_logits.sum(dim=1) / denom

        elif mode == "max":
            masked_logits = cell_logits.masked_fill(~cell_mask, float("-inf"))
            return masked_logits.max(dim=1).values

        elif mode == "lse":
            masked_logits = cell_logits.masked_fill(~cell_mask, float("-inf"))
            n_cells = cell_mask.sum(dim=1).clamp_min(1)

            return temperature * (
                torch.logsumexp(masked_logits / temperature, dim=1)
                - torch.log(n_cells.float())
            )

        else:
            raise ValueError(f"Unknown pooling mode: {mode}")

    def train(self, mode: bool = True):
        """Standard train()/eval(), except the frozen TF-DNA submodule can be pinned.

        nn.Module.train() recurses, so Lightning calling .train() each epoch flips the
        pretrained TF-DNA model into train mode even though its parameters have
        requires_grad=False and it is only ever called under no_grad. Freezing the
        weights does not freeze the *mode*: its three BatchNorm1d layers then normalise
        by batch statistics over whatever (TF, peak) pairs happen to share a chunk, and
        keep overwriting their own running_mean/running_var.

        Measured on the real mm10 checkpoint, that is not a rounding difference --
        eval-mode and train-mode binding logits differ by mean 1.14 (logit sd 6.13),
        2.1% of pairs cross the 0.5 probability boundary, and the running means drift
        1.7-12.6% over 200 batches. It also means the TF-TG model trains against batch
        statistics and is then evaluated against running statistics.

        Pinning eval fixes that inconsistency and makes the padding-skip/crop fast path
        legal during training (measured 1822 -> 311 ms/step), but it changes what a newly
        trained model sees, so it is opt-in: every existing checkpoint was trained with
        this False.
        """
        super().train(mode)
        if self.keep_tf_peak_model_in_eval:
            self.tf_peak_model.eval()
        return self

    def set_tf_embedding_table(self, tf_embeddings_tensor, tf_mask_tensor):
        """
        Hold the TF protein embeddings on the model's device.

        Without this the dataloader ships a [T, D] embedding per edge across PCIe --
        roughly 1 GB per batch of 512 at T ~= 4000, almost all of it duplicated,
        because a batch touches only a few hundred distinct TFs. The whole table is
        under 2 GB, so keeping it resident and gathering on-device removes that
        transfer entirely. Pass tf_idx to forward() once this is set.
        """
        device = next(self.parameters()).device
        self.tf_embedding_table = tf_embeddings_tensor.to(device).float()
        self.tf_mask_table = tf_mask_tensor.to(device).bool()
        self.tf_length_table = self.tf_mask_table.sum(dim=1).long()
        return self

    # Crop widths are quantized to this ladder so torch.compile sees a handful of shapes
    # rather than one per batch.
    #
    # Rung count matters more than rung placement, because the edge universe is built
    # TF-major (MultiIndex.from_product([tfs, tgs])), so a batch holds a single TF and the
    # crop changes as the run walks from TF to TF. Every distinct (crop, n_chunks) pair
    # costs an Inductor compile and a CUDA-graph recording, and once the total exceeds
    # torch._dynamo.config.cache_size_limit they evict each other and recompile forever:
    # a 5-rung ladder at tf_peak_chunk_size=256 measured 15-38 s/batch spikes recurring
    # indefinitely, against 0.14-0.57 s/batch on already-compiled shapes.
    #
    # Three rungs give up almost nothing: mean crop 869 vs 836 for mm10 (6.43x vs 6.69x)
    # and 903 vs 882 for hg38 (4.43x vs 4.54x), with only 1.1% / 1.6% of TFs falling
    # through to the full table width.
    TF_CROP_LADDER = (512, 1024, 2048)

    def _tf_lengths_per_slot(self, tf_idx, tf_mask_edge, use_resident_table, E, P):
        """Real TF protein length for each of the E*P (edge, peak) slots, as [E*P]."""
        if use_resident_table:
            tf_len_edge = self.tf_length_table[tf_idx]          # [E]
        else:
            tf_len_edge = tf_mask_edge.sum(dim=1).long()        # [E]
        return tf_len_edge.repeat_interleave(P)

    def _chunk_crop_lengths(self, tf_len_sorted, n_chunks, width, table_len):
        """Per-chunk crop width: the chunk's longest TF, rounded up the ladder."""
        chunk_max = tf_len_sorted.view(n_chunks, width).amax(dim=1)
        # One sync for the whole loop instead of one per chunk.
        crops = []
        for longest in chunk_max.tolist():
            rung = next(
                (r for r in self.TF_CROP_LADDER if r >= longest),
                table_len,
            )
            crops.append(min(rung, table_len))
        return crops

    def forward(
        self,
        tf_embedding=None,
        tf_mask=None,
        peak_sequences=None,
        peak_accessibility=None,
        peak_distance=None,
        tf_expression=None,
        tg_expression=None,
        cell_mask=None,
        peak_mask=None,
        pooling_mode: str = "lse",
        pooling_temperature: float = 1.0,
        tf_idx=None,
    ):
        """
        Bag-level forward pass.

        This computes TF-DNA binding once per TF-TG edge and peak,
        then reuses those binding scores across sampled cells.

        Parameters
        ----------
        tf_embedding : [E, T, D], optional
            Pre-gathered embeddings. Omit and pass `tf_idx` instead to gather from the
            device-resident table registered by set_tf_embedding_table().
        tf_mask : [E, T], optional
        peak_sequences : [E, P, L, 4]
        peak_accessibility : [E, C, P]
        peak_distance : [E, P]
        tf_expression : [E, C]
        tg_expression : [E, C]
        cell_mask : [E, C]
        peak_mask : [E, P], optional
        tf_idx : [E], optional
            Row indices into the resident embedding table. Used only when
            `tf_embedding` is None.

        Returns
        -------
        edge_logits : [E]
        cell_logits : [E, C]
        """

        if tf_embedding is None:
            if tf_idx is None:
                raise ValueError("forward() needs either tf_embedding or tf_idx.")
            if self.tf_embedding_table is None:
                raise ValueError(
                    "tf_idx was supplied but no embedding table is resident. "
                    "Call set_tf_embedding_table() first."
                )
            # Gathered per chunk below rather than materialising [E, T, D] here, so the
            # peak memory matches the pre-gathered path instead of exceeding it.
            tf_idx = tf_idx.to(self.tf_embedding_table.device).long().reshape(-1)
            if tf_mask is None:
                tf_mask = self.tf_mask_table

        if not torch.is_floating_point(peak_sequences):
            peak_sequences = peak_sequences.float()

        E, C = cell_mask.shape
        _, P, L, nuc_dim = peak_sequences.shape
        EC = E * C

        # ------------------------------------------------------------
        # 1. Cell-invariant edge-level tensors
        # ------------------------------------------------------------
        # These are repeated across cells in your current dataloader.
        # Use only the first cell to avoid C-fold redundant TF-DNA inference.
        tf_embedding_edge = tf_embedding         # [E, T, D], or None when using tf_idx
        tf_mask_edge = tf_mask                  # [E, T], or the full table when using tf_idx
        T = tf_mask_edge.shape[1]               # padded protein length of the table
        peak_sequences_edge = peak_sequences    # [E, P, L, 4]
        peak_distance_edge = peak_distance        # [E, P]

        use_resident_table = tf_embedding_edge is None

        def gather_tf_chunk(edge_idx):
            """Embeddings and masks for a chunk of edge indices, [chunk, T, D] / [chunk, T]."""
            if use_resident_table:
                table_rows = tf_idx[edge_idx]
                return self.tf_embedding_table[table_rows], self.tf_mask_table[table_rows]
            return tf_embedding_edge[edge_idx], tf_mask_edge[edge_idx]

        if peak_mask is not None:
            peak_mask_edge = peak_mask            # [E, P]
        else:
            peak_mask_edge = None

        # ------------------------------------------------------------
        # 2a. Frozen TF-DNA binding model: [E, P]
        # ------------------------------------------------------------

        # Flatten the peaks into a single batch dimension of ExP
        peak_seq_flat = peak_sequences_edge.reshape(E * P, L, nuc_dim)

        chunk_size = self.tf_peak_chunk_size
        if chunk_size is None or chunk_size <= 0:
            chunk_size = E * P

        # Every edge is padded out to max_peaks_real, which is set by the single TG with
        # the most nearby peaks, so most slots in a batch are padding -- typically 60-75%
        # of them. Binding for padded slots is thrown away by the masked_fill below, so
        # at inference time we score only the real peaks.
        #
        # This is gated on eval mode on purpose. tf_peak_model's peak encoder contains
        # nn.BatchNorm1d, and there is no train() override keeping the frozen submodule
        # in eval, so during TF-TG training those layers normalise by *batch* statistics
        # -- which makes the binding output depend on exactly which rows share a chunk.
        # Skipping rows or repacking chunks there would silently change training results
        # and stop previously-trained models from being reproducible. In eval mode
        # BatchNorm uses its running statistics, so the fast path is bitwise identical.
        skip_padded_peaks = peak_mask_edge is not None and not self.tf_peak_model.training

        with torch.no_grad():
            # zeros, not empty: on the fast path the padded slots are never written, and
            # a defined value keeps them finite going into the sigmoid before masking
            binding_logits_flat = torch.zeros(
                E * P,
                device=peak_sequences_edge.device,
                dtype=peak_sequences_edge.dtype,
            )

            if skip_padded_peaks:
                mask_flat = peak_mask_edge.reshape(-1)
                n_valid = int(mask_flat.sum())

                if n_valid > 0:
                    # Every shape here has to be stable across batches or torch.compile
                    # re-traces. An earlier version selected slots with .nonzero() and
                    # sized the chunk exactly to the work; both shapes then tracked
                    # n_valid, which is data-dependent and near-unique per batch. Dynamo
                    # guarded on the nonzero() output size and recompiled every batch --
                    # measured ~45 s/batch against 0.72 s once warm, i.e. slower than
                    # never compiling at all.
                    #
                    # So: order the slots with a fixed-shape stable argsort (valid first)
                    # instead of nonzero(), and round the chunk width up to CHUNK_QUANTUM.
                    # That leaves only (width, n_chunks) varying, over a handful of
                    # combinations. Capping the width at chunk_size bounds memory; the
                    # quantum as a floor bounds the wasted rows when n_valid << chunk_size
                    # -- the case that padding up to whole chunk_size blocks got wrong.
                    #
                    # Rows past n_valid address genuinely padded slots, so their logits
                    # are meaningless, but binding_score is masked_fill'd by peak_mask
                    # below before it is ever used. Computing them is wasted work, never
                    # wrong work, and it is bounded by one quantum per batch.
                    CHUNK_QUANTUM = 256
                    quantized = ((n_valid + CHUNK_QUANTUM - 1) // CHUNK_QUANTUM) * CHUNK_QUANTUM
                    width = min(chunk_size, max(CHUNK_QUANTUM, quantized))
                    n_chunks = (n_valid + width - 1) // width
                    total = n_chunks * width

                    # Order slots by (valid first, then TF protein length). The validity
                    # key is what the chunking above needs; the length key is what makes
                    # the crop below worth doing. TF embeddings are padded to the longest
                    # protein in the table -- 5,588 tokens for mm10 against a median of
                    # 474 -- and tf_encoder plus the cross-attention run over every padded
                    # position, so cost is ~linear in the padded length. Grouping slots of
                    # similar length means each chunk can be cropped near its own longest
                    # TF instead of the table's.
                    tf_len_slot = self._tf_lengths_per_slot(
                        tf_idx, tf_mask_edge, use_resident_table, E, P
                    )
                    sort_key = torch.where(
                        mask_flat, tf_len_slot, tf_len_slot + (T + 1)
                    )
                    chunk_idx_source = torch.argsort(sort_key, stable=True)
                    if total > chunk_idx_source.numel():
                        # Only when chunk_size is not a multiple of the quantum. The pad
                        # repeats an already-masked slot, so it stays harmless.
                        chunk_idx_source = torch.cat(
                            [
                                chunk_idx_source,
                                chunk_idx_source[-1].expand(total - chunk_idx_source.numel()),
                            ]
                        )

                    # Longest real TF in each chunk, rounded up to the crop ladder. Read
                    # in one .tolist() before the loop rather than per chunk: each read
                    # of a device tensor into Python is a sync and a graph break.
                    chunk_crop_lengths = self._chunk_crop_lengths(
                        tf_len_slot[chunk_idx_source[:total]], n_chunks, width, T
                    )

                    for chunk_i, start in enumerate(range(0, total, width)):
                        sel = chunk_idx_source[start : start + width]
                        edge_idx = sel // P

                        tf_embedding_chunk, tf_mask_chunk = gather_tf_chunk(edge_idx)

                        # Drop trailing positions that are padding for every row in this
                        # chunk. Masks are strict prefixes and the embeddings are zero
                        # past the mask, so the dropped columns contribute nothing to the
                        # masked attention or to masked_mean_pool -- exact, not an
                        # approximation.
                        crop = chunk_crop_lengths[chunk_i]
                        if crop < tf_embedding_chunk.shape[1]:
                            tf_embedding_chunk = tf_embedding_chunk[:, :crop]
                            tf_mask_chunk = tf_mask_chunk[:, :crop]

                        logits_chunk = self.tf_peak_model(
                            tf_embedding=tf_embedding_chunk,
                            tf_mask=tf_mask_chunk,
                            peak_embedding=peak_seq_flat[sel],
                        )

                        # Copy values out before next compiled-model invocation.
                        # index_copy_ requires matching dtypes, and under autocast the
                        # TF-DNA model returns reduced precision while this buffer is
                        # fp32, so cast explicitly -- the .copy_ used on the path below
                        # would have done it implicitly.
                        binding_logits_flat.index_copy_(
                            0, sel, logits_chunk.to(binding_logits_flat.dtype)
                        )
            else:
                # Original path, unchanged: score every slot including padding, with the
                # same ragged final chunk. Used for training and whenever no peak mask
                # is supplied.
                for start in range(0, E * P, chunk_size):
                    end = min(start + chunk_size, E * P)

                    flat_idx = torch.arange(start, end, device=peak_sequences_edge.device)
                    edge_idx = flat_idx // P

                    tf_embedding_chunk, tf_mask_chunk = gather_tf_chunk(edge_idx)
                    peak_seq_chunk = peak_seq_flat[start:end]

                    logits_chunk = self.tf_peak_model(
                        tf_embedding=tf_embedding_chunk,
                        tf_mask=tf_mask_chunk,
                        peak_embedding=peak_seq_chunk,
                    )

                    # Copy values out before next compiled-model invocation
                    binding_logits_flat[start:end].copy_(logits_chunk)

        binding_logits = binding_logits_flat.reshape(E, P)
        
        # ------------------------------------------------------------
        # 2b. Mask and expand TF-peak binding scores across cells
        # ------------------------------------------------------------
        # Sigmoid to convert logits to probabilities
        binding_score = torch.sigmoid(binding_logits)  # [E, P]

        # If a peak mask is provided, set binding scores of masked peaks to 0
        if peak_mask_edge is not None:
            binding_score = binding_score.masked_fill(~peak_mask_edge, 0.0)

        # Reuse TF-peak binding score across cells
        binding_score = binding_score[:, None, :].expand(E, C, P)  # [E, C, P]

        # ------------------------------------------------------------
        # 3. Distance features
        # ------------------------------------------------------------
        abs_distance = peak_distance_edge.abs()
        distance_scaled = torch.clamp(abs_distance / 250_000.0, 0.0, 1.0)   # [E, P]
        distance_weight = torch.exp(-abs_distance / 50_000.0)               # [E, P]

        if peak_mask_edge is not None:
            distance_scaled = distance_scaled.masked_fill(~peak_mask_edge, 0.0)
            distance_weight = distance_weight.masked_fill(~peak_mask_edge, 0.0)

        distance_scaled = distance_scaled[:, None, :].expand(E, C, P) # [E, C, P]
        distance_weight = distance_weight[:, None, :].expand(E, C, P) # [E, C, P]

        # ------------------------------------------------------------
        # 4. Cell-specific peak features
        # ------------------------------------------------------------
        if peak_mask_edge is not None:
            peak_accessibility = peak_accessibility.masked_fill(
                ~peak_mask_edge[:, None, :],
                0.0,
            )
            
        assert binding_score.shape == peak_accessibility.shape, (
            f"binding_score {binding_score.shape} != peak_accessibility {peak_accessibility.shape}"
        )
        assert distance_scaled.shape == peak_accessibility.shape, (
            f"distance_scaled {distance_scaled.shape} != peak_accessibility {peak_accessibility.shape}"
        )
        assert distance_weight.shape == peak_accessibility.shape, (
            f"distance_weight {distance_weight.shape} != peak_accessibility {peak_accessibility.shape}"
        )

        peak_features = torch.stack(
            [
                binding_score,
                peak_accessibility,
                distance_scaled,
                distance_weight,
            ],
            dim=-1,
        )  # [E, C, P, 4]

        peak_features = peak_features.reshape(EC, P, 4)  # [E*C, P, 4]
        peak_tokens = self.peak_feature_proj(peak_features)  # [E*C, P, d_model]

        # ------------------------------------------------------------
        # 5. Expression tokens
        # ------------------------------------------------------------
        tf_expr_token = self.tf_expr_proj(
            tf_expression.reshape(EC, 1)
        )  # [E*C, d_model]

        tg_expr_token = self.tg_expr_proj(
            tg_expression.reshape(EC, 1)
        )  # [E*C, d_model]

        tg_query_input = tf_expr_token + tg_expr_token

        tg_query = self.tg_query_proj(tg_query_input).unsqueeze(1)  # [E*C, 1, d_model]

        # ------------------------------------------------------------
        # 6. TG query attends to linked peak tokens
        # ------------------------------------------------------------
        key_padding_mask = None

        if peak_mask_edge is not None:
            key_padding_mask = peak_mask_edge[:, None, :].expand(E, C, P)
            key_padding_mask = ~key_padding_mask.reshape(EC, P)  # True = ignore

        peak_context, _ = self.peak_attention(
            query=tg_query,
            key=peak_tokens,
            value=peak_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        peak_context = self.norm(peak_context.squeeze(1))  # [E*C, d_model]

        # ------------------------------------------------------------
        # 7. Cell-level logits
        # ------------------------------------------------------------
        final = torch.cat(
            [
                peak_context,
                tf_expr_token,
                tg_expr_token,
            ],
            dim=-1,
        )  # [E*C, d_model * 3]

        cell_logits = self.classifier(final).squeeze(-1)  # [E*C]
        cell_logits = cell_logits.reshape(E, C)           # [E, C]

        # ------------------------------------------------------------
        # 8. Pool cell logits into edge logits
        # ------------------------------------------------------------
        edge_logits = self.pool_cell_logits(
            cell_logits,
            cell_mask=cell_mask,
            mode=pooling_mode,
            temperature=pooling_temperature,
        )  # [E]

        return edge_logits, cell_logits
    
class LitTFTGRegulationModel(pl.LightningModule):
    def __init__(
        self,
        model: TFTGRegulationModel,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        pos_weight: float | None = None,
        pooling_mode: str = "lse",
        pooling_temperature: float = 1.0,
        logit_clamp: float | None = 20.0,
        enable_timing_sync: bool = False,
        warmup_steps: int = 0,
    ):
        super().__init__()

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        # Linear LR warmup over the first `warmup_steps` optimizer steps. 0 disables it,
        # which is the historical behaviour. Needed once the effective batch grows: the
        # ReduceLROnPlateau below only reacts after val/loss has already stalled, so it
        # cannot protect the first few hundred steps, which is exactly where a large batch
        # at a scaled-up LR diverges.
        self.warmup_steps = int(warmup_steps)
        self.pooling_mode = pooling_mode
        self.pooling_temperature = pooling_temperature
        self.logit_clamp = logit_clamp
        self.enable_timing_sync = enable_timing_sync

        self.save_hyperparameters(ignore=["model"])

        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)
            self.register_buffer("pos_weight", pos_weight_tensor)
        else:
            self.pos_weight = None

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()

        self.val_probs = []
        self.val_targets = []
        # Validation batches dropped because every logit was NaN/Inf. Counted per epoch
        # and surfaced in on_validation_epoch_end -- a non-zero count means the model has
        # started diverging, which is worth seeing rather than silently averaging over.
        self._n_empty_val_batches = 0
        self._prev_batch_end_time = None
        self._epoch_start_time = None
        self._step_start_time = None
        self._backward_start_time = None
        self._timing_window_size = 50
        self._timing_windows = {
            "load": [],
            "h2d": [],
            "forward": [],
            "backward": [],
            "step": [],
        }
        self._latest_timing_avgs = {}

    def _sync_if_cuda(self, device=None) -> None:
        if self.enable_timing_sync and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    def _record_timing(self, name: str, value: float) -> None:
        window = self._timing_windows[name]
        window.append(value)
        if len(window) > self._timing_window_size:
            window.pop(0)

        self._latest_timing_avgs[name] = sum(window) / len(window)

    def forward(self, batch):
        
        return self.model(
            tf_embedding=batch.get("tf_embedding", None),
            tf_mask=batch.get("tf_mask", None),
            peak_sequences=batch["peak_sequences"],
            peak_accessibility=batch["peak_accessibility"],
            peak_distance=batch["peak_distance"],
            tf_expression=batch["tf_expression"],
            tg_expression=batch["tg_expression"],
            cell_mask=batch["cell_mask"],
            peak_mask=batch.get("peak_mask", None),
            pooling_mode=self.pooling_mode,
            pooling_temperature=self.pooling_temperature,
            tf_idx=batch.get("tf_idx", None),
        )

    def _loss(self, logits, labels):
        if self.pos_weight is not None:
            return nn.functional.binary_cross_entropy_with_logits(
                logits,
                labels,
                pos_weight=self.pos_weight,
            )

        return nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
        )

    def _shared_step(self, batch, stage: str):
        labels = batch["label"].float()

        forward_start = None
        if stage == "train":
            self._sync_if_cuda()
            forward_start = time.perf_counter()

        edge_logits, _ = self.forward(batch)

        if forward_start is not None:
            self._sync_if_cuda()
            forward_time = time.perf_counter() - forward_start
            self._record_timing("forward", forward_time)

        if self.logit_clamp is not None:
            edge_logits = edge_logits.clamp(min=-self.logit_clamp, max=self.logit_clamp)

        loss = self._loss(edge_logits, labels)
        probs = torch.sigmoid(edge_logits)

        if stage == "train":
            acc = self.train_acc(probs, labels.int())
        elif stage == "val":

            valid_mask = torch.isfinite(probs) & torch.isfinite(labels)

            probs = probs[valid_mask]
            labels = labels[valid_mask]

            if probs.numel() == 0:
                # Every logit in this batch was NaN/Inf, so the finite mask emptied it.
                # torchmetrics cannot update from an empty tensor -- BinaryAccuracy
                # reshapes to [0, -1], which raises "cannot reshape tensor of 0 elements".
                # Left unguarded, one bad batch kills the whole run at the epoch boundary,
                # which is how job 3788646 lost 8 hours and 7 good epochs. Skip the batch
                # and let on_validation_epoch_end aggregate whatever else was finite.
                acc = None
                self._n_empty_val_batches += 1
            else:
                acc = self.val_acc(probs, labels.int())

                self.val_probs.append(probs.detach().float().cpu())
                self.val_targets.append(labels.detach().int().cpu())
        else:
            raise ValueError(f"Unknown stage: {stage}")

        self.log(
            f"{stage}/loss",
            loss,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=(stage != "train"),
        )

        if acc is not None:
            self.log(
                f"{stage}/acc",
                acc,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=(stage != "train"),
            )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if not self.training:
            return batch

        if self._prev_batch_end_time is not None:
            load_time = time.perf_counter() - self._prev_batch_end_time
            self._record_timing("load", load_time)

        return batch

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if not self.training:
            return super().transfer_batch_to_device(batch, device, dataloader_idx)

        start_time = time.perf_counter()
        batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        self._sync_if_cuda(device)
        h2d_time = time.perf_counter() - start_time
        self._record_timing("h2d", h2d_time)
        return batch
    
    def on_train_epoch_start(self):
        for k in self._timing_windows:
            self._timing_windows[k].clear()
        self._latest_timing_avgs.clear()
        self._prev_batch_end_time = None
        self._epoch_start_time = time.perf_counter()
        

    def on_train_batch_start(self, batch, batch_idx):
        self._sync_if_cuda()
        self._step_start_time = time.perf_counter()

    def on_before_backward(self, loss):
        self._sync_if_cuda()
        self._backward_start_time = time.perf_counter()

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure,
    ):
        start_time = self._backward_start_time or time.perf_counter()
        result = super().optimizer_step(
            epoch,
            batch_idx,
            optimizer,
            optimizer_closure,
        )
        self._sync_if_cuda()
        backward_opt_time = time.perf_counter() - start_time
        self._backward_start_time = None
        self._record_timing("backward", backward_opt_time)
        return result

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self._step_start_time is None:
            return

        self._sync_if_cuda()
        step_time = time.perf_counter() - self._step_start_time
        self._step_start_time = None
        self._record_timing("step", step_time)
        self._prev_batch_end_time = time.perf_counter()

        if batch_idx % 50 == 0:
            for name, avg_value in self._latest_timing_avgs.items():
                self.log(
                    f"train/{name}_time_avg",
                    avg_value,
                    on_step=True,
                    on_epoch=False,
                    prog_bar=False,
                    logger=True,
                    sync_dist=False,
                )

    def on_train_epoch_end(self):
        if self._epoch_start_time is None:
            return

        self._sync_if_cuda()
        epoch_time = time.perf_counter() - self._epoch_start_time
        epoch_time_mins = epoch_time / 60.0
        self._epoch_start_time = None

        self.log(
            "train/epoch_time_min",
            epoch_time_mins,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=False,
        )

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, stage="val")
        
        

    def on_validation_epoch_start(self):
        self.val_probs.clear()
        self.val_targets.clear()
        self._n_empty_val_batches = 0

    def on_validation_epoch_end(self):
        if self._n_empty_val_batches:
            logging.warning(
                f"{self._n_empty_val_batches} validation batch(es) were entirely "
                "non-finite and were skipped. The model is producing NaN/Inf logits -- "
                "treat this epoch's val metrics as unreliable and check for divergence."
            )
        if not self.val_probs:
            logging.error(
                "Every validation batch was non-finite; no val metrics for this epoch."
            )
            return
        
        probs = torch.cat(self.val_probs, dim=0).view(-1)
        targets = torch.cat(self.val_targets, dim=0).view(-1).int()

        self.val_probs.clear()
        self.val_targets.clear()

        targets = np.asarray(targets).astype(float).ravel()
        probs = np.asarray(probs).astype(float).ravel()

        finite_mask = np.isfinite(targets) & np.isfinite(probs)

        targets = targets[finite_mask].astype(int)
        probs = probs[finite_mask]

        if len(targets) == 0:
            auroc = np.nan
            auprc = np.nan
        elif len(np.unique(targets)) < 2:
            auroc = np.nan
            auprc = np.nan
        else:
            auroc = roc_auc_score(targets, probs)
            auprc = average_precision_score(targets, probs)

        self.log("val/auroc", auroc, prog_bar=True, sync_dist=True)
        self.log("val/auprc", auprc, prog_bar=True, sync_dist=True)

        if not getattr(self.logger, "experiment", None):
            return

        if not self.trainer.is_global_zero:
            return

        try:
            if len(np.unique(targets)) < 2:
                return

            pre, rec, _ = precision_recall_curve(targets, probs)
            fpr, tpr, _ = roc_curve(targets, probs)

            def _sample_curve(x, y, n=100):
                
                if len(x) <= n:
                    return x, y
                idx = np.linspace(0, len(x) - 1, n).astype(int)
                return x[idx], y[idx]

            rec_s, pre_s = _sample_curve(rec, pre, n=100)
            fpr_s, tpr_s = _sample_curve(fpr, tpr, n=100)

            self.logger.experiment.log({
                "val/pr_curve": wandb.plot.line_series(
                    [rec_s],
                    [pre_s],
                    keys=["precision"],
                    xname="Recall",
                ),
                "val/roc_curve": wandb.plot.line_series(
                    [fpr_s],
                    [tpr_s],
                    keys=["TPR"],
                    xname="FPR",
                ),
            })
        except Exception as e:
            print("[WARN] PR/ROC curve error:", e)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None, **kwargs):
        """Linear LR warmup, applied only while inside the warmup window.

        Deliberately stops touching param_group["lr"] once warmup is over, so
        ReduceLROnPlateau owns the LR from then on -- the two would otherwise fight, with
        warmup overwriting every reduction the scheduler made.
        """
        if self.warmup_steps > 0 and self.trainer.global_step < self.warmup_steps:
            scale = float(self.trainer.global_step + 1) / float(self.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = scale * self.lr

        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.1, 
            patience=5, 
            threshold=1e-4, 
            threshold_mode='rel', 
            cooldown=3, 
            min_lr=1e-7, 
            eps=1e-08
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",      # Adjust LR per 'epoch' or 'step'
                "frequency": 1,           # How often to step the scheduler
                "monitor": "val/loss",    # Metric to track for ReduceLROnPlateau
            },
        }

# ----- Utility Functions -----
@torch.no_grad()
def move_batch_to_device(batch, device):
    moved = {
        "peak_sequences": batch["peak_sequences"].to(device, non_blocking=True),
        "peak_accessibility": batch["peak_accessibility"].to(device, non_blocking=True),
        "peak_distance": batch["peak_distance"].to(device, non_blocking=True),
        "tf_expression": batch["tf_expression"].to(device, non_blocking=True),
        "tg_expression": batch["tg_expression"].to(device, non_blocking=True),
        "label": batch["label"].to(device, non_blocking=True),
    }

    # tf_embedding/tf_mask are absent when the dataset returns indices instead, in
    # which case tf_idx is what the model needs on-device to gather from its table.
    for key in ("tf_embedding", "tf_mask", "tf_idx", "cell_mask", "peak_mask"):
        if key in batch:
            moved[key] = batch[key].to(device, non_blocking=True)

    return moved

