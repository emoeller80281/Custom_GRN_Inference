import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import numpy as np
import math
from typing import Optional, Union, Tuple, Sequence

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
        # x: [B, S, D]
        S = x.size(1)
        return x + self.pe[:S].unsqueeze(0)
    
class CheckpointedEncoder(nn.Module):
    """
    Wrap a stack of encoder layers, checkpointing each layer during training.
    Works with batch_first=True (input [B, S, D]).
    """
    def __init__(self, layers, norm=None, use_checkpoint=True, use_reentrant=False):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm
        self.use_checkpoint = use_checkpoint
        self.use_reentrant = use_reentrant

    def forward(self, x):
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                # pass the flag explicitly and avoid the lambda
                x = checkpoint(layer, x, use_reentrant=self.use_reentrant)
            else:
                x = layer(x)
        return self.norm(x) if self.norm is not None else x

class MultiomicTransformer(nn.Module):
    def __init__(
        self,
        *,
        n_genes: int,           # number of genes to predict (query length)
        tg_in_dim: int,         # input feature size for TG token
        tf_in_dim: int,         # input feature size for TF token
        window_in_dim: int,     # per-window feature size (ATAC/etc.)
        d_model: int,
        nhead: int,
        dff: int,
        dropout: float,
        n_layers: int,
        kernel_stride_size: int,
        max_windows: int = 4096,
        use_checkpoint_encoder: bool = False,
    ):
        super().__init__()

        # Linear projections of TF, TG, and the windows, with dropout
        self.proj_tg = nn.Linear(tg_in_dim, d_model, bias=False)
        self.proj_tf = nn.Linear(tf_in_dim, d_model, bias=False)
        self.proj_window = nn.Sequential(
            nn.Linear(window_in_dim, d_model, bias=False),
            nn.Dropout(dropout),
        )

        # Pool windows to reduce the dimensionality, helps with computational efficiency
        # at the cost of decreased resolution
        self.pool = nn.AvgPool1d(kernel_size=kernel_stride_size, stride=kernel_stride_size)

        # Positional encoding over window sequence
        # Helps to keep track of where the peak accessibility is in relation to genes
        self.posenc = SinusoidalPositionalEncoding(d_model, max_len=max_windows)

        # Encoder stack
        # Each window attends to other windows + feedforward layer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        # Build an encoder with multiple layers
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Cross-attention
        # Creates an embedding of potential TGs to the same dimension of the model
        self.gene_embed = nn.Embedding(n_genes, d_model)  # queries
        
        # Each gene queries the window embeddings for context about its expression
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)

        # Takes the attention layer and maps to a scalar prediction for each gene
        self.readout = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
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
        H, D,                           # SciPy CSR: H [TF x P], D [P x G]
        peaks_by_window: Sequence,      # list/seq of np.ndarray peak indices per window
        col_of,                         # cell_id -> column index in RNA/ATAC matrices
        rna_arr, atac_arr,              # typically numpy / SciPy matrices
        device: Optional[torch.device] = None,
    ):
        """
        Stashes references (kept on CPU). Call once per process.
        """
        self.H = H
        self.D = D
        self.peaks_by_window = peaks_by_window
        self.col_of = col_of
        self.rna_arr = rna_arr
        self.atac_arr = atac_arr
        self.token_device = device or next(self.parameters()).device

    def build_tokens_streaming_for_cells(
        self,
        cell_ids: Sequence[int],
        clamp_val: float = 3.0,
        pad_to_max_in_batch: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            tokens: [B, W', d_model] if pad_to_max_in_batch=False
                    [B, Wmax, d_model] if pad_to_max_in_batch=True (padded with zeros)
            key_padding_mask: None if not padded,
                              else [B, Wmax] (True = PAD)
        Preserves streaming behavior: never materializes [W, d_model].
        """
        assert hasattr(self, "H") and hasattr(self, "D"), "Call attach_sparse_sources(...) first."
        device = self.token_device

        # pool kernel size (int)
        ks = self.pool.kernel_size if isinstance(self.pool.kernel_size, int) else self.pool.kernel_size[0]

        out_batches = []
        seq_lengths = []

        # For fallback zero token shape
        if isinstance(self.proj_window, nn.Sequential):
            d_model_out = self.proj_window[0].out_features
        else:
            d_model_out = self.proj_window.out_features

        for cell in cell_ids:
            col = self.col_of[cell]

            # Column slices (CPU)
            e = self.rna_arr[:, col]   # [TF]
            a = self.atac_arr[:, col]  # [P]

            # Scale sparse and keep CSR for slicing
            B = self.H.multiply(e[:, None]).tocsr()  # [TF, P]
            R = self.D.multiply(a[:, None]).tocsr()  # [P,  G]

            pooled_list = []        # will hold a few [1,1,d_model]; tiny
            sum_tok = None          # [1, d_model] on device
            cnt = 0

            for p_idx in self.peaks_by_window:
                # active peaks for this window in this cell
                p_active = p_idx[a[p_idx] > 0]
                if p_active.size == 0:
                    continue

                # (TF x |p|) @ (|p| x G) -> [TF, G], CPU dense
                Mw = (B[:, p_active] @ R[p_active, :]).toarray()

                # to GPU, stabilize per TF
                M = torch.as_tensor(Mw, device=device, dtype=torch.float32)
                # safety
                if not torch.isfinite(M).all():
                    M = torch.nan_to_num(M, 0.0)

                # standardize per TF across genes
                M = (M - M.mean(dim=1, keepdim=True)) / (M.std(dim=1, keepdim=True) + 1e-6)
                M = torch.clamp(torch.nan_to_num(M, 0.0), -clamp_val, clamp_val)

                # projections on device (keep fp32 to avoid overflow)
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    # M.t(): [G, TF] -> proj_tf: Linear(TF -> tf_channels)
                    Xtf  = self.proj_tf(M.t().unsqueeze(0))      # [1, G, tf_ch]
                    # transpose to [1, tf_ch, G] -> proj_tg: Linear(G -> tg_channels)
                    Xtg  = self.proj_tg(Xtf.transpose(1, 2))     # [1, tf_ch, tg_ch]
                    feat = Xtg.reshape(1, -1)                    # [1, tf_ch * tg_ch]
                    tok  = self.proj_window(feat)                # [1, d_model]

                # rolling sum for avg pooling over windows
                sum_tok = tok if sum_tok is None else (sum_tok + tok)
                cnt += 1

                if cnt == ks:
                    pooled = (sum_tok / float(ks)).unsqueeze(1)  # [1, 1, d_model]
                    pooled_list.append(pooled)
                    sum_tok = None
                    cnt = 0

                # free big intermediates promptly
                del M, Xtf, Xtg, feat, tok

            # tail (< ks)
            if cnt > 0 and sum_tok is not None:
                pooled = (sum_tok / float(cnt)).unsqueeze(1)     # [1, 1, d_model]
                pooled_list.append(pooled)
                sum_tok = None; cnt = 0

            if len(pooled_list) == 0:
                toks = torch.zeros((1, 1, d_model_out), device=device, dtype=torch.float32)
            else:
                toks = torch.cat(pooled_list, dim=1)             # [1, W', d_model]

            out_batches.append(toks)
            seq_lengths.append(toks.size(1))

            # free per-cell CSR on CPU
            del B, R

        # If no padding requested, all W' must match or this will raise (same as your current function)
        if not pad_to_max_in_batch:
            tokens = torch.cat(out_batches, dim=0)               # [B, W', d_model]
            return tokens, None

        # Otherwise pad to the longest within this batch and build a key_padding_mask
        Wmax = int(max(seq_lengths))
        Bsz  = len(out_batches)
        tokens = torch.zeros((Bsz, Wmax, d_model_out), device=device, dtype=torch.float32)
        key_padding_mask = torch.ones((Bsz, Wmax), device=device, dtype=torch.bool)  # True=PAD
        for i, t in enumerate(out_batches):
            w = t.size(1)
            tokens[i, :w, :] = t[0]
            key_padding_mask[i, :w] = False

        return tokens, key_padding_mask

    def encode_windows(self, windows: torch.Tensor) -> torch.Tensor:
        """
        
        windows: Raw window features [batch, windows, window features]
        Returns: Encoded windows [batch, pooled windows, d_model]
        """
        # Linear project windows to [batch, windows, d_model]
        # Adds dropout
        x = self.proj_window(windows)     
        
        # AvgPool1d the windows
        x = x.transpose(1, 2)   # Transpose to get the windows as the last dim
        x = self.pool(x)        # AvgPool1d the windows (must be the last dimension)
        x = x.tranpose(1, 2)    # Transpose back to the original shape
        
        # Add the positional encoding  
        x = self.posenc(x)                
        
        # Run the transformer encoder
        # Attend to all other windows to gain global context
        encoded_windows = self.encoder(x) 
        
        return encoded_windows

    def forward(
        self,
        windows: torch.Tensor,          # [B, S, Fw] window features (already pooled OR raw per-bin -> you pool before passing here)
        gene_ids: torch.Tensor,          # [G] LongTensor (0..n_genes-1)
        tf_token: Optional[torch.Tensor] = None,   # [B, Ftf]
        tg_token: Optional[torch.Tensor] = None,   # [B, Ftg]
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, S] True for PAD
        attn_mask: Optional[torch.Tensor] = None,         # [S, S] or [B, S, S]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            pred: [B, G] predictions per gene
            Z:    [B, G, D] attended gene representations (for analysis)
        """
        # Encode window sequence
        encoded_windows = self.encode_windows(windows)  # [B, S, D]

        batch_size = encoded_windows.size(0)    # Get the batch size from the window embeddings
        gene_indices = gene_ids.numel()         # Get the number of elements in gene_ids
        
        # Build gene queries
        gene_queries = (
            self.gene_embed(gene_ids)               # Create an embeddings for the genes         
            .unsqueeze(0)                           # Get the embeddings into the same size 
            .expand(batch_size, gene_indices, -1)
            )  # [B, G, D]

        # Run Cross-Attention
        # Each gene queries the window embeddings to find most relevant
        # windows for its expression
        gene_embeddings, _ = self.cross_attn(
            query=gene_queries, 
            key=encoded_windows, 
            value=encoded_windows,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask
            )  # [B, G, D]

        # Final prediction is one expression prediction per gene per batch
        pred = self.readout(gene_embeddings).squeeze(-1)  # [B, G]
        return pred, gene_embeddings
    

