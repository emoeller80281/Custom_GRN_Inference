import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import numpy as np
import math
from typing import Optional, Union, Tuple, Sequence, cast, List
import scipy.sparse as sp

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S = x.size(1)
        return x + self.pe[:S].to(x.dtype).unsqueeze(0)

class MultiomicTransformer(nn.Module):
    def __init__(
        self,
        *,
        n_genes: int,           # number of genes to predict (query length)
        tf_in_dim: int,         # input feature size for TF token
        d_model: int,
        nhead: int,
        dff: int,
        dropout: float,
        n_layers: int,
        kernel_stride_size: int,
        max_windows: int = 4096,
    ):
        super().__init__()

        # Linear projections of TF, TG, and the windows, with dropout
        self.d_model = d_model
        self.tf_channels = 64
        self.log_attn_bias_scale = nn.Parameter(torch.tensor(math.log(0.1), dtype=torch.float32))
        
        self.kernel_stride_size = kernel_stride_size
        
        # Projects the TF vector to a smaller dimension, then 
        self.window_from_tf = nn.Sequential(
            nn.Linear(tf_in_dim, self.tf_channels, bias=False),  # TF -> 32
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.tf_channels, self.d_model, bias=True),
            nn.LayerNorm(self.d_model)
        )

        # Pool windows to reduce the dimensionality, helps with computational efficiency
        # at the cost of decreased resolution
        self.pool = nn.AvgPool1d(kernel_size=self.kernel_stride_size, stride=self.kernel_stride_size)

        # Positional encoding over window sequence
        # Helps to keep track of where the peak accessibility is in relation to genes
        self.posenc = SinusoidalPositionalEncoding(self.d_model, max_len=max_windows)

        # Encoder stack
        # Each window attends to other windows + feedforward layer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        
        # Build an encoder with multiple layers
        self.win_encoder = nn.TransformerEncoder(
            enc_layer, 
            num_layers=n_layers, 
            enable_nested_tensor=False
            )
        
        # Learn an embedding per TF and scale it by expression to form tokens
        self.tf_id_embed = nn.Embedding(tf_in_dim, d_model)        # [n_tfs, d_model]
        tf_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dff,
                                              dropout=dropout, batch_first=True, norm_first=True)
        self.tf_encoder = nn.TransformerEncoder(tf_layer, num_layers=max(1, n_layers//2), enable_nested_tensor=False)
        self.tf_ln = nn.LayerNorm(d_model)

        # Cross-attention
        # Creates an embedding of potential TGs to the same dimension of the model
        self.gene_embed = nn.Embedding(n_genes, self.d_model)  # queries
        
        # Layer normalize before passing to the decoder (keep )
        self.layer_norm_gene_queries = nn.LayerNorm(self.d_model)
        self.layer_norm_encoder_keys_values = nn.LayerNorm(self.d_model)
        
        # Each gene queries the window embeddings for context about its expression
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=nhead, dropout=dropout, batch_first=True)

        # Takes the attention layer and maps to a scalar prediction for each gene
        self.readout = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 1)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.weight is not None:
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def attach_sparse_sources(
        self,
        tf_peak_matrix,                 # SciPy CSR: tf_peak_matrix [TF x peak binding potential]
        peak_gene_matrix,               # SciPy CSR: peak_gene_matrix [peak x gene regulatory potential]
        peaks_by_window: Sequence,      # list/seq of np.ndarray peak indices per window
        col_of,                         # cell_id -> column index in RNA/ATAC matrices
        rna_arr, atac_arr,              # typically numpy / SciPy matrices
        device: Optional[torch.device] = None,
    ):
        """
        Stashes references (kept on CPU). Call once per process.
        """
        self.tf_peak_matrix = tf_peak_matrix
        self.peak_gene_matrix = peak_gene_matrix
        self.peaks_by_window = peaks_by_window
        self.col_of = col_of
        self.rna_arr = rna_arr
        self.atac_arr = atac_arr
        self.token_device = device or next(self.parameters()).device

    def _module_device(self) -> torch.device:
        return next(self.parameters()).device

    def _pool_mask(self, key_padding_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Downsample [batch_size, num_windows] -> [batch_size, num_windows_pooled] to match AvgPool1d(kernel=stride=K).
        Rule: a pooled position is PAD=True only if **all** covered positions were pad.
        """
        if key_padding_mask is None:
            return None
        
        batch_size, num_windows = key_padding_mask.shape
        
        # Trim to multiple of the kernel stride size to avoid ragged pooling
        num_windows_trimmed = (num_windows // self.kernel_stride_size) * self.kernel_stride_size
        
        # Group into [batch_size, num_pooled_groups, kernel_size]
        grouped = key_padding_mask[:, :num_windows_trimmed]  # keep bool
        grouped = grouped.contiguous().view(
            batch_size,
            num_windows_trimmed // self.kernel_stride_size,
            self.kernel_stride_size
        )
        pooled_mask = grouped.all(dim=-1)  # bool
        if num_windows_trimmed < num_windows:
            tail = key_padding_mask[:, num_windows_trimmed:]
            tail = tail.all(dim=-1, keepdim=True)
            pooled_mask = torch.cat([pooled_mask, tail], dim=1)
        return pooled_mask

    def build_tokens_streaming_for_cells(
        self,
        cell_ids: Sequence[int],
        clamp_val: float = 3.0,
        pad_to_max_in_batch: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """
        Returns:
        tokens: [B, W', d_model]
        key_padding_mask: None or [B, W'] (True = PAD)
        attn_bias: [B, W', G]  (additive bias for attention logits; larger => gene attends more to that window)
        """
        assert hasattr(self, "tf_peak_matrix") and hasattr(self, "peak_gene_matrix"), "Call attach_sparse_sources(...) first."
        device = self._module_device()

        kernel_size: int = self.pool.kernel_size if isinstance(self.pool.kernel_size, int) else self.pool.kernel_size[0]
        d_model_out: int = self.d_model

        per_cell_tokens: List[torch.Tensor] = []
        per_cell_bias:   List[torch.Tensor] = []
        pooled_lengths:  List[int] = []

        for cell_id in cell_ids:
            col_index = self.col_of[cell_id]

            # Sparse column slices (CPU)
            peak_acc = self.atac_arr[:, col_index]  # [P]

            # TF-peak binding scaled by TF expression (CSR)
            tf_peak_binding = self.tf_peak_matrix  # [TF, P]

            pooled_token_chunks: List[torch.Tensor] = []   # each [1, 1, d_model]
            pooled_bias_chunks:  List[torch.Tensor] = []   # each [1, 1, G]
            rolling_sum_token: Optional[torch.Tensor] = None   # [1, d_model]
            rolling_sum_bias:  Optional[torch.Tensor] = None   # [1, G]
            windows_accum = 0

            for window_peak_indices in self.peaks_by_window:
                # Cell-specific active peaks in this window (gate by accessibility)
                active_peak_idx = window_peak_indices[peak_acc[window_peak_indices] > 0]
                if active_peak_idx.size == 0:
                    continue

                # --------- TF-only token for this window ---------
                # Weighted average of TF contributions across peaks in this window
                weights = peak_acc[active_peak_idx].astype(np.float32)
                wsum = float(weights.sum())
                if wsum > 0:
                    weights = weights / wsum
                tf_window = (tf_peak_binding[:, active_peak_idx] @ weights)

                tf_window_t = torch.as_tensor(tf_window, device=device, dtype=torch.float32)
                # stabilize
                tf_window_t = (tf_window_t - tf_window_t.mean()) / (tf_window_t.std() + 1e-6)
                tf_window_t = torch.clamp(torch.nan_to_num(tf_window_t, nan=0.0), -clamp_val, clamp_val)

                with torch.amp.autocast(device_type="cuda", enabled=False):
                    token = self.window_from_tf(tf_window_t.unsqueeze(0))  # [1, d_model]

                # --------- Gene×window distance bias for this window ---------
                # Pull distances for active peaks: [|p|, G] on CPU, reduce by max over peaks
                # We treat larger values as "closer/better". If your matrix is "distance" where smaller is better,
                #       you may want to invert (e.g., bias = -distance) or normalize appropriately.
                pg_sub = self.peak_gene_matrix[active_peak_idx, :]  # sparse [|p|, G]
                # reduce: maximum over peaks
                # use .max(axis=0) on CSR: returns (1, G) matrix
                reduced  = pg_sub.max(axis=0)  # [G]
                
                if sp.issparse(reduced):
                    window_gene_bias_np = reduced.toarray().ravel()
                else:
                    window_gene_bias_np = np.asarray(reduced).ravel()

                window_gene_bias = torch.as_tensor(window_gene_bias_np, device=device, dtype=torch.float32).unsqueeze(0)  # [G]

                # --------- rolling pooling (tokens + bias) ---------
                rolling_sum_token = token if rolling_sum_token is None else (rolling_sum_token + token)           # [1, d_model]
                rolling_sum_bias  = window_gene_bias if rolling_sum_bias is None else (rolling_sum_bias + window_gene_bias)  # [1, G]
                windows_accum += 1

                if windows_accum == kernel_size:
                    pooled_token = (rolling_sum_token / float(kernel_size)).unsqueeze(1)    # [1, 1, d_model]
                    pooled_bias  = (rolling_sum_bias  / float(kernel_size)).unsqueeze(1)    # [1, 1, G]
                    pooled_token_chunks.append(pooled_token)
                    pooled_bias_chunks.append(pooled_bias)
                    rolling_sum_token, rolling_sum_bias = None, None
                    windows_accum = 0

                # free CPU temporaries
                del tf_window, tf_window_t, token, pg_sub, window_gene_bias_np, window_gene_bias

            # tail (< kernel_size)
            if windows_accum > 0 and (rolling_sum_token is not None) and (rolling_sum_bias is not None):
                pooled_token = (rolling_sum_token / float(windows_accum)).unsqueeze(1)  # [1, 1, d_model]
                pooled_bias  = (rolling_sum_bias  / float(windows_accum)).unsqueeze(1)  # [1, 1, G]
                pooled_token_chunks.append(pooled_token)
                pooled_bias_chunks.append(pooled_bias)

            if len(pooled_token_chunks) == 0:
                tokens_for_cell = torch.zeros((1, 1, d_model_out), device=device, dtype=torch.float32)
                bias_for_cell   = torch.zeros((1, 1, self.gene_embed.num_embeddings), device=device, dtype=torch.float32)
            else:
                tokens_for_cell = torch.cat(pooled_token_chunks, dim=1)  # [1, W', d_model]
                bias_for_cell   = torch.cat(pooled_bias_chunks,  dim=1)  # [1, W', G]

            per_cell_tokens.append(tokens_for_cell)
            per_cell_bias.append(bias_for_cell)
            pooled_lengths.append(tokens_for_cell.size(1))

            # free per-cell CSR
            del tf_peak_binding

        # no padding
        if not pad_to_max_in_batch:
            tokens   = torch.cat(per_cell_tokens, dim=0)  # [B, W', d_model]
            attn_bias = torch.cat(per_cell_bias,  dim=0)  # [B, W', G]
            return tokens, None, attn_bias

        # pad to max length in batch
        B = len(per_cell_tokens)
        Wmax = int(max(pooled_lengths))
        G = self.gene_embed.num_embeddings

        tokens    = torch.zeros((B, Wmax, d_model_out), device=device, dtype=torch.float32)
        bias_pads = torch.zeros((B, Wmax, G),           device=device, dtype=torch.float32)
        key_padding_mask = torch.ones((B, Wmax),        device=device, dtype=torch.bool)  # True=PAD

        for i, (tok_seq, bias_seq) in enumerate(zip(per_cell_tokens, per_cell_bias)):
            wlen = tok_seq.size(1)
            tokens[i, :wlen, :]    = tok_seq[0]
            bias_pads[i, :wlen, :] = bias_seq[0]
            key_padding_mask[i, :wlen] = False

        return tokens, key_padding_mask, bias_pads

    def build_tf_tokens_for_cells(self, cell_ids: Sequence[int], clamp_val: float = 3.0) -> torch.Tensor:
        """
        Returns TF tokens from expression only: [B, n_tfs, d_model]
        token(tf_k) = expr_k * TF_Embedding[k]
        """
        device = self._module_device()
        tf_count = self.tf_id_embed.num_embeddings
        tokens = []
        for cell_id in cell_ids:
            col = self.col_of[cell_id]
            tf_expr = self.rna_arr[:, col].astype(np.float32)  # shape [n_tfs]
            tf_expr_t = torch.as_tensor(tf_expr, device=device)
            # normalize a bit to stabilize
            tf_expr_t = (tf_expr_t - tf_expr_t.mean()) / (tf_expr_t.std() + 1e-6)
            tf_expr_t = torch.clamp(torch.nan_to_num(tf_expr_t, nan=0.0), -clamp_val, clamp_val)  # [n_tfs]
            # scale learned TF embeddings by expression => tokens [n_tfs, d_model]
            tf_tok = tf_expr_t.unsqueeze(1) * self.tf_id_embed.weight  # broadcast
            tokens.append(tf_tok.unsqueeze(0))  # [1, n_tfs, d_model]
        return torch.cat(tokens, dim=0)  # [B, n_tfs, d_model]


    def encode_windows(
        self, 
        windows: torch.Tensor, 
        key_padding_mask: Optional[torch.Tensor] = None,
        already_pooled: Optional[bool] = True
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        
        windows: Raw window features [batch, windows, window features]
        Returns: Encoded windows [batch, pooled windows, d_model]
        """
        # Linear project windows to [batch, windows, d_model]
        # Adds dropout
        x = windows
        
        # Pooling already occurs in the token builder. Can pool here if needed
        if not already_pooled:
            # AvgPool1d the windows
            x = x.transpose(1, 2)   # Transpose to get the windows as the last dim
            x = self.pool(x)        # AvgPool1d the windows (must be the last dimension)
            x = x.transpose(1, 2)    # Transpose back to the original shape
            pooled_mask = self._pool_mask(key_padding_mask)
        else:
            assert x.size(-1) == self.d_model, \
                "already_pooled=True requires d_model features"
            pooled_mask = key_padding_mask
        
        # Add the positional encoding  
        x = self.posenc(x)                
        
        # Run the transformer encoder
        # Attend to all other windows to gain global context
        encoded_windows = self.win_encoder(x, mask=None, src_key_padding_mask=pooled_mask)
        
        return encoded_windows, pooled_mask

    def forward(
        self,
        windows: torch.Tensor,                  # [B, W, d_model] if already pooled; else [B, W, TF]
        gene_ids: torch.Tensor,                 # [G]
        tf_tokens: Optional[torch.Tensor] = None,        # [B, T, d_model] (optional)
        key_padding_mask_win: Optional[torch.Tensor] = None,   # [B, W] (True=PAD)
        attn_bias_win: Optional[torch.Tensor] = None,           # [B, W, G]
        already_pooled: bool = True,
    ):
        B = windows.size(0)
        G = int(gene_ids.numel())
        H = self.cross_attn.num_heads

        # 1) Encode windows (positional encoding + Transformer + proper PAD mask)
        win_enc, pooled_mask = self.encode_windows(
            windows, key_padding_mask=key_padding_mask_win, already_pooled=already_pooled
        )                                           # [B, W', D], pooled_mask [B, W'] or None
        Wp = win_enc.size(1)

        # 2) Encode TF tokens if provided
        if tf_tokens is not None:
            tf_enc = self.tf_encoder(self.tf_ln(tf_tokens))     # [B, T, D]
            T = tf_enc.size(1)
            mem = torch.cat([win_enc, tf_enc], dim=1)           # [B, W'+T, D]
        else:
            T = 0
            mem = win_enc                                       # [B, W', D]

        mem = self.layer_norm_encoder_keys_values(mem)

        # 3) Gene queries
        q = self.gene_embed(gene_ids).unsqueeze(0).expand(B, G, -1)  # [B, G, D]
        q = self.layer_norm_gene_queries(q)

        # 4) Build additive attention bias over the fused memory (windows [+ TFs])
        fused_bias = None
        if attn_bias_win is not None:
            # [B, W, G] -> [B, G, W]
            win_bias = attn_bias_win.permute(0, 2, 1).to(mem.dtype)
            # z-norm per (B,G)
            win_bias = (win_bias - win_bias.mean(dim=2, keepdim=True)) / (win_bias.std(dim=2, keepdim=True) + 1e-6)
            scale = self.log_attn_bias_scale.exp()              # ensure positive scale
            win_bias = scale * win_bias                         # larger => more attention
            # pad mask for windows (if any) -> very negative
            if pooled_mask is not None:
                very_neg = -1e4
                win_bias = win_bias.masked_fill(pooled_mask.unsqueeze(1), very_neg)
            # append zeros for TF segment if present
            if T > 0:
                tf_bias = torch.zeros(B, G, T, device=mem.device, dtype=mem.dtype)
                fused_bias = torch.cat([win_bias, tf_bias], dim=2)      # [B, G, W'+T]
            else:
                fused_bias = win_bias                                   # [B, G, W']

        # 5) Convert to MultiheadAttention attn_mask (additive, per-head)
        attn_mask = None
        if fused_bias is not None:
            attn_mask = (fused_bias
                        .unsqueeze(1)                    # [B, 1, G, W'+T]
                        .expand(B, H, G, Wp + T)
                        .reshape(B * H, G, Wp + T))      # [B*H, G, W'+T]

        # 6) Cross-attention: genes query fused memory
        gene_ctx, _ = self.cross_attn(query=q, key=mem, value=mem, attn_mask=attn_mask)  # [B, G, D]
        pred = self.readout(gene_ctx).squeeze(-1)                                        # [B, G]
        return pred, gene_ctx
