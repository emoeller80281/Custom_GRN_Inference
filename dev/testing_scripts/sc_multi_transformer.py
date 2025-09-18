import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import numpy as np
import math
from typing import Optional, Union, Tuple, Sequence, cast

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
        self.d_model = d_model
        self.tf_channels = 32
        self.tg_channels = 64
        self.proj_tf = nn.Linear(tf_in_dim, self.tf_channels, bias=False)  # TF -> 32
        self.proj_tg = nn.Linear(tg_in_dim, self.tg_channels, bias=False)  # G  -> 64
        self.kernel_stride_size = kernel_stride_size

        self.proj_window = nn.Linear(self.tf_channels * self.tg_channels, d_model, bias=False)

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
        self.encoder = nn.TransformerEncoder(
            enc_layer, 
            num_layers=n_layers, 
            enable_nested_tensor=False
            )

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
        grouped = key_padding_mask[:, :num_windows_trimmed].float()  # [batch, num_windows_trimmed]
        grouped = grouped.view(
            batch_size, 
            num_windows_trimmed // self.kernel_stride_size, 
            self.kernel_stride_size
            )
        
        # A pooled position is pad if ALL windows in that group are pad
        pooled_mask = grouped.all(dim=-1)  # [batch, windows_trimmed // kernel size]
        
        # Handle tail (<pooling size): if exists, pad if all were pad
        if num_windows_trimmed < num_windows:
            tail = key_padding_mask[:, num_windows_trimmed:].all(dim=-1, keepdim=True)  # [batch,1]
            pooled_mask = torch.cat([pooled_mask, tail], dim=1)
            
        return pooled_mask

    def build_tokens_streaming_for_cells(
        self,
        cell_ids: Sequence[int],
        clamp_val: float = 3.0,
        pad_to_max_in_batch: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Builds pooled window tokens for the given cells, streaming over windows without
        materializing full [windows, d_model] intermediates.

        Returns:
            tokens:
                [batch_size, num_windows_pooled, d_model] if pad_to_max_in_batch=False
                [batch_size, max_windows_pooled_in_batch, d_model] if True (right-padded with zeros)
            key_padding_mask:
                None if not padded;
                otherwise [batch_size, max_windows_pooled_in_batch] with True = PAD (ignore), False = real token
        """
        assert hasattr(self, "tf_peak_matrix") and hasattr(self, "peak_gene_matrix"), "Call attach_sparse_sources(...) first."
        device = self._module_device()

        # Pool (reduce) windows in groups of this size
        kernel_size: int = self.pool.kernel_size if isinstance(self.pool.kernel_size, int) else self.pool.kernel_size[0]

        # Output feature size of the window projection (d_model)
        d_model_out: int = self.proj_window.out_features

        per_cell_token_sequences: list[torch.Tensor] = []
        seq_lengths_per_cell: list[int] = []

        for cell_id in cell_ids:
            # Get the column index for the cell in the RNA and ATAC data arrays
            col_index = self.col_of[cell_id]

            # Sparse column slices (CPU)
            rna_col = self.rna_arr[:, col_index]    # [TF]
            atac_col = self.atac_arr[:, col_index]  # [P]

            # Multiply the TF-peak binding potential by the RNA expression for the TFs
            tf_by_peak_scaled = self.tf_peak_matrix.multiply(rna_col[:, None]).tocsr()  # [TF, P]
            
            # Multiply the Peak-TG regulatory potential by the ATAC accessibility
            peak_by_gene_scaled = self.peak_gene_matrix.multiply(atac_col[:, None]).tocsr()  # [P, G]

            pooled_token_chunks: list[torch.Tensor] = []   # each chunk is [1, 1, d_model]
            rolling_sum_token: Optional[torch.Tensor] = None  # [1, d_model] on device
            windows_accumulated = 0

            # Generate the TF-TG embeddings for each window
            # This embeds the TF expression, TF-peak binding, peak-TG regulatory potential, and peak accessibility for each windwo
            for window_peak_indices in self.peaks_by_window:
                # Active peaks for this window in this cell
                active_peak_indices = window_peak_indices[atac_col[window_peak_indices] > 0]
                if active_peak_indices.size == 0:
                    continue
                
                # Matrix multiply the TF-peak binding with the peak-TG regulatory potential for windows with peaks
                # (TF x |p|) @ (|p| x G) -> [TF, G], dense on CPU
                tf_by_gene_cpu = (tf_by_peak_scaled[:, active_peak_indices] @ peak_by_gene_scaled[active_peak_indices, :]).toarray()

                # Move to GPU
                tf_by_gene = torch.as_tensor(tf_by_gene_cpu, device=device, dtype=torch.float32)

                # Remove infinite and nan values
                if not torch.isfinite(tf_by_gene).all():
                    tf_by_gene = torch.nan_to_num(tf_by_gene, nan=0.0, posinf=0.0, neginf=0.0)

                # Standardize per-TF across genes, clamp outliers
                tf_by_gene = (tf_by_gene - tf_by_gene.mean(dim=1, keepdim=True)) / (tf_by_gene.std(dim=1, keepdim=True) + 1e-6)
                tf_by_gene = torch.clamp(torch.nan_to_num(tf_by_gene, nan=0.0), -clamp_val, clamp_val)

                # Project to token (stay fp32 to avoid overflow)
                with torch.amp.autocast(device_type="cuda", enabled=False):
                    # Linear projections for TF and TG
                    tf_proj  = self.proj_tf(tf_by_gene.T.unsqueeze(0))        # [1, G, 32]
                    tg_proj  = self.proj_tg(tf_proj.transpose(1, 2))          # [1, 32, 64]
                    
                    # Flatten the TF x TG projections into a vector
                    fused    = tg_proj.reshape(1, -1)                          # [1, tf_channels * tg_channels]
                    assert fused.size(-1) == self.proj_window.in_features, \
                        f"fused={fused.size(-1)} vs proj_window.in_features={self.proj_window.in_features}"
                        
                    # Project the linear TF-TG embedding down to the model dimension
                    token    = self.proj_window(fused)                         # [1, d_model_out]

                # Rolling sum for manual average pooling across consecutive windows 
                # (Mimics Avg1DPool to aggregate consecutive windows)
                rolling_sum_token = token if rolling_sum_token is None else (rolling_sum_token + token)
                windows_accumulated += 1

                if windows_accumulated == kernel_size:
                    pooled_token = (rolling_sum_token / float(kernel_size)).unsqueeze(1)  # [1, 1, d_model]
                    pooled_token_chunks.append(pooled_token)
                    rolling_sum_token = None
                    windows_accumulated = 0

                # Free large intermediates promptly
                del tf_by_gene, tf_proj, tg_proj, fused, token

            # Divide the rolling sum by the number of windows for the cell to get the average
            if windows_accumulated > 0 and rolling_sum_token is not None:
                pooled_token = (rolling_sum_token / float(windows_accumulated)).unsqueeze(1)  # [1, 1, d_model]
                pooled_token_chunks.append(pooled_token)
                rolling_sum_token = None
                windows_accumulated = 0

            # Concatenate pooled chunks for this cell
            if len(pooled_token_chunks) == 0:
                tokens_for_cell = torch.zeros((1, 1, d_model_out), device=device, dtype=torch.float32)
            else:
                tokens_for_cell = torch.cat(pooled_token_chunks, dim=1)  # [1, num_windows_pooled, d_model]

            per_cell_token_sequences.append(tokens_for_cell)
            seq_lengths_per_cell.append(tokens_for_cell.size(1))

            # Free per-cell CSR on CPU
            del tf_by_peak_scaled, peak_by_gene_scaled

        # If no padding requested, require all pooled lengths to match and return directly
        if not pad_to_max_in_batch:
            tokens = torch.cat(per_cell_token_sequences, dim=0)  # [batch_size, num_windows_pooled, d_model]
            return tokens, None

        # Otherwise, right-pad to the longest pooled sequence in the batch and build a key_padding_mask
        max_windows_pooled_in_batch: int = int(max(seq_lengths_per_cell))   # Find the max number of pooled windows
        batch_size: int = len(per_cell_token_sequences)

        # Create a mask (1 if the position in the window is a true token or 0 if its padding)
        tokens = torch.zeros((batch_size, max_windows_pooled_in_batch, d_model_out), device=device, dtype=torch.float32)
        key_padding_mask = torch.ones((batch_size, max_windows_pooled_in_batch), device=device, dtype=torch.bool)  # True=PAD

        for row_index, cell_tokens in enumerate(per_cell_token_sequences):
            num_windows_pooled = cell_tokens.size(1)
            tokens[row_index, :num_windows_pooled, :] = cell_tokens[0]
            key_padding_mask[row_index, :num_windows_pooled] = False  # mark real tokens

        return tokens, key_padding_mask


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
        
        if x.size(-1) != self.d_model:
            x = self.proj_window(x)
        
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
        encoded_windows = self.encoder(x) 
        
        return encoded_windows, pooled_mask

    def forward(
        self,
        windows: torch.Tensor,          # [B, S, Fw] window features (already pooled OR raw per-bin -> you pool before passing here)
        gene_ids: torch.Tensor,          # [G] LongTensor (0..n_genes-1)
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, S] True for PAD
        attn_mask: Optional[torch.Tensor] = None,         # [S, S] or [B, S, S]
        already_pooled: Optional[bool] = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            pred: [B, G] predictions per gene
            Z:    [B, G, D] attended gene representations (for analysis)
        """
        # Encode window sequence
        encoded_windows, pooled_mask = self.encode_windows(windows, key_padding_mask, already_pooled)  # [B, S, D]

        batch_size = encoded_windows.size(0)    # Get the batch size from the window embeddings
        gene_indices = gene_ids.numel()         # Get the number of elements in gene_ids
        
        # Build gene queries
        gene_queries = (
            self.gene_embed(gene_ids)               # Create an embeddings for the genes         
            .unsqueeze(0)                           # Get the embeddings into the same size 
            .expand(batch_size, gene_indices, -1)
            )  # [B, G, D]
        
        # Layer normalize the window embeddings from the encoder and the gene
        # embeddings from the decoder to stabilize activations
        gene_queries_layernorm = self.layer_norm_gene_queries(gene_queries)
        encoded_windows_layernorm = self.layer_norm_encoder_keys_values(encoded_windows)

        # Run Cross-Attention
        # Each gene queries the window embeddings to find most relevant
        # windows for its expression
        gene_embeddings, _ = self.cross_attn(
            query=gene_queries_layernorm, 
            key=encoded_windows_layernorm, 
            value=encoded_windows_layernorm,
            key_padding_mask=(pooled_mask.to(encoded_windows_layernorm.device) if pooled_mask is not None else None),
            attn_mask=attn_mask
            )  # [B, G, D]

        # Final prediction is one expression prediction per gene per batch
        pred = self.readout(gene_embeddings).squeeze(-1)  # [B, G]
        return pred, gene_embeddings
    

