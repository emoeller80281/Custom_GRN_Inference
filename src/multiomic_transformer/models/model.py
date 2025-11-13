import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINEConv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
import torch.utils.checkpoint as cp
from typing import Optional
import numpy as np
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value
        
        # Linear layers for transforming inputs
        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None, attn_bias=None):
        # Expected: Q,K,V = [batch_size,H,L, d_k]
        assert Q.dim() == 4 and K.dim() == 4 and V.dim() == 4, \
            f"Q,K,V must be 4D: got {Q.shape}, {K.shape}, {V.shape}"
            
        batch_size,H,Lq,Dq = Q.shape
        Bk,Hk,Lk,Dk = K.shape
        Bv,Hv,Lv,Dv = V.shape
        assert (batch_size,H) == (Bk,Hk) == (Bv,Hv), f"Batch/heads mismatch: Q{(batch_size,H)} K{(Bk,Hk)} V{(Bv,Hv)}"
        assert Dq == Dk == Dv, f"Head dim mismatch: {Dq} vs {Dk} vs {Dv}"
        assert Lk == Lv, f"K/V length mismatch: Lk={Lk}, Lv={Lv}"
        
        # 0) sanitize inputs early
        Q = torch.nan_to_num(Q, nan=0.0, posinf=1e4, neginf=-1e4)
        K = torch.nan_to_num(K, nan=0.0, posinf=1e4, neginf=-1e4)
        V = torch.nan_to_num(V, nan=0.0, posinf=1e4, neginf=-1e4)
        if attn_bias is not None:
            attn_bias = torch.nan_to_num(attn_bias, nan=0.0, posinf=1e4, neginf=-1e4)

        # 1) compute logits in fp32 for numeric headroom
        scale = 1.0 / math.sqrt(self.d_k)
        logits = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale  # [B,H,Lq,Lk]

        # 2) add bias (clamped)
        if attn_bias is not None:
            logits = logits + attn_bias.float().clamp_(-1e4, 1e4)

        # 3) apply mask with dtype-aware large negative
        if mask is not None:
            large_neg = -1e4  # fp16/bf16-safe (instead of -1e9)
            # assume mask==0 means "masked"
            logits = logits.masked_fill(mask == 0, large_neg)

        # 4) softmax in fp32, then cast back to original dtype
        probs = torch.softmax(logits, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)
        out = torch.matmul(probs, V.float()).to(Q.dtype)
        return out
        
    def split_heads(self, x):
        
        # Reshape the input to have num_heads for multi-head attention
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None, attn_bias=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask, attn_bias)
        
        # Combine heads and apply output transformation
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEmbedding, self).__init__()

        self.d_model = d_model

        inv_freq = 1 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]
    
class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, mask=None, attn_bias=None):
        # query: [batch_size, len_query, d_model]
        # key_value: [batch_size, len_key_val, d_model]
        attn_out = self.attn(query, key_value, key_value, mask, attn_bias=attn_bias)
        attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=1e4, neginf=-1e4)
        res = query + 0.1 * self.dropout(attn_out)
        out = self.norm(res)
        return out

class TFtoTGShortcut(nn.Module):
    def __init__(
        self,
        d_model,
        use_motif_mask: bool = False,
        lambda_l1: float = 1e-4,
        lambda_l2: float = 0.0,
        topk: Optional[int] = None,
        dropout_p: float = 0.2,
        motif_mask_threshold: Optional[float] = None,  # in -log10(p)
        motif_prior_scale: float = 0.0,             # >0 to use soft prior
    ):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        self.use_motif_mask = use_motif_mask
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.topk = topk
        self.dropout_p = dropout_p

        self.motif_mask_threshold = motif_mask_threshold
        self.motif_prior_scale = motif_prior_scale

        # big negative for masking
        self.register_buffer("big_neg", torch.tensor(-1e4, dtype=torch.float32))

    def forward(self, tg_emb, tf_id_emb, tf_expr, motif_mask=None):
        """
        tg_emb      : [G, d_model]
        tf_id_emb   : [T, d_model]
        tf_expr     : [B, T]
        motif_mask  : [G, T] aggregated TF–TG motif scores in -log10(p)
                      (e.g. max or sum over peaks near each TG)
        """
        device = tg_emb.device
        tf_id_emb = tf_id_emb.to(device)
        tf_expr = tf_expr.to(device)

        # [G, T] similarity between TG & TF embeddings
        sim = torch.matmul(tg_emb, tf_id_emb.T) / math.sqrt(tf_id_emb.size(1))

        if self.use_motif_mask and motif_mask is not None:
            motif_mask = motif_mask.to(device)

            # 1) Hard structural mask: require sufficient motif evidence
            if self.motif_mask_threshold is not None:
                allowed = motif_mask >= self.motif_mask_threshold  # bool [G, T]
                # disallow edges with weak/no motif support
                sim = sim.masked_fill(~allowed, self.big_neg.item())
            else:
                allowed = None

            # 2) Optional soft prior: add scaled motif scores on allowed edges
            if self.motif_prior_scale > 0.0:
                prior = motif_mask
                if allowed is not None:
                    prior = torch.where(allowed, prior, torch.zeros_like(prior))
                sim = sim + self.motif_prior_scale * prior

        # numerics
        sim = sim.clamp_(-50, 50)

        # attention over TFs for each TG
        attn = torch.softmax(sim, dim=-1)          # [G, T]
        attn = torch.nan_to_num(attn, nan=0.0)

        # handle all-masked rows (all big_neg -> all zeros after softmax)
        row_sums = attn.sum(dim=-1, keepdim=True)
        attn = torch.where(row_sums > 0, attn / row_sums, attn)

        # optional top-k sparsification
        if self.topk is not None:
            topk_vals, topk_idx = torch.topk(attn, self.topk, dim=-1)
            mask = torch.zeros_like(attn)
            mask.scatter_(1, topk_idx, 1.0)
            attn = attn * mask
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

        # [B, G]: mix TF expression for each TG
        tf_scalar = tf_expr @ attn.T
        tf_scalar = F.dropout(tf_scalar, p=self.dropout_p, training=self.training)

        self.attn = attn  # for regularization / inspection

        output = self.scale * tf_scalar
        return output, attn

    def regularization(self):
        if not hasattr(self, "attn"):
            return 0.0
        l1 = self.lambda_l1 * self.attn.abs().sum()
        l2 = self.lambda_l2 * (self.attn ** 2).sum()
        return l1 + l2

class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
        self.query = nn.Parameter(torch.randn(d_model))  # learnable query vector

    def forward(self, x):
        # x: [batch_size, N, d_model]
        scores = torch.matmul(x, self.query) / (self.d_model ** 0.5)  # [batch_size, N]
        weights = F.softmax(scores, dim=1).unsqueeze(-1)          # [batch_size, N, 1]
        pooled = torch.sum(weights * x, dim=1)                   # [batch_size, d_model]
        return pooled, weights

class MultiomicTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout,
                 tf_vocab_size, tg_vocab_size, 
                 bias_scale=2.0,
                 use_bias=True,
                 use_shortcut=True, 
                 use_motif_mask=False, 
                 motif_mask_threshold=0.0,
                 motif_prior_scale=0.0,
                 lambda_l1=1e-4, 
                 lambda_l2=0.0,
                 topk=None,
                 shortcut_dropout=0.2,
                 use_gradient_checkpointing: bool = False,
                 ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.use_bias = use_bias
        self.use_motif_mask = use_motif_mask
        self.use_shortcut = use_shortcut
        self.bias_scale = bias_scale
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # TF and TG embedding tables - generates identities for TFs and TGs
        self.tf_identity_emb = nn.Embedding(tf_vocab_size, d_model)
        self.tg_query_emb = nn.Embedding(tg_vocab_size, d_model)        # 
        self.tg_identity_emb = nn.Embedding(tg_vocab_size, d_model)
        
        # ATAC windows do not have embeddings, their position and accessibility
        # are used to inform TFs and TGs
        
        # TF dense layer - Projects the TF RNA-seq expression data to [n_tfs x d_model]
        self.tf_expr_dense_input_layer = nn.Sequential(
            nn.Linear(1, d_ff, bias=False),        # Projects each TF independently
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
            nn.LayerNorm(d_model)
        )
        
        # ATAC dense layer - Projects the ATAC window accessibility to [n_windows x d_model]
        self.atac_acc_dense_input_layer = nn.Sequential(
            nn.Linear(1, d_ff, bias=False), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
            nn.LayerNorm(d_model)
        )
        
        # ATAC positional embedding - Adds a positional embedding to the windows so the model can learn how the relative 
        # position of each window in the sequence affects a TG
        self.posenc = PositionalEmbedding(d_model)
        
        # ATAC encoder - Model learns to pick up on patterns of ATAC accessibility
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_layers,
            enable_nested_tensor=False
        )
        # ----- TF-ATAC and ATAC-TF Cross Attention
        # TF -> ATAC Cross Attention "Which windows matter for this TF?"
        #     Each TF embedding (identity + expression) attends over window embeddings
        #     to learn TF-specific regulatory information. TF-specific information
        #     Creates one vector of length d_model per TF
        self.cross_tf_to_atac = CrossAttention(d_model, num_heads, dropout) # [batch_size, num_tfs_evaluated, d_model]
        
        # ATAC -> TF Cross Attention "Which TFs explain this accessibility?"
        #    Windows attends to the TF embeddings to create a window aware view of the
        #    TFs available and their expression. Global TF context based on accessibility.
        #    Creates one vector of length d_model per window
        self.cross_atac_to_tf = CrossAttention(d_model, num_heads, dropout) # [batch_size, num_windows, d_model]
        
        # ----- Attention Pooling -----
        #    Reduces shape from [sequence, d_model] -> [d_model]
        #    Weighted average of softmax-norm dot product between learnable query vector and each element in the sequence.
        #    Creates a global TF or ATAC sample-level summary of shape d_model
        
        # TF -> ATAC Cross Attention Pooling
        self.tf_to_atac_cross_attn_pool = AttentionPooling(d_model)     # [batch_size, d_model]
        
        # ATAC -> TF Cross Attention Pooling
        self.atac_to_tf_cross_attn_pool = AttentionPooling(d_model)     # [batch_size, d_model]
        
        # TF-ATAC and ATAC-TF attention pooling output is concatenated  # [batch_size, 2*d_model]
        
        # Pooled Cross-Attention Output Dense Layer
        #     Pass the output of the TF -> ATAC and ATAC -> TF cross-attention through a dense layer
        #     to reduce the dimensionality back down from [2*d_model] -> [d_model]
        self.pooled_cross_attn_dense_layer = nn.Sequential(
            nn.Linear(2*d_model, d_ff, bias=False),
            # GELU helps to keep small negatives rather than zeroing them out like RELU
            nn.GELU(),  
            nn.Dropout(self.dropout),
            nn.Linear(d_ff, d_model, bias=False),
            nn.LayerNorm(d_model)
        )
        
        # TG -> ATAC Cross Attention "Which windows matter for this TG?"
        #    TG query vector attends to the ATAC windows.
        #    Distance bias is added 
        self.cross_tg_to_atac = CrossAttention(d_model, num_heads, dropout)
        
        # The output from pooled_cross_attn_dense_layer is added to each TG in cross_tg_to_atac's output
        
        # Cast the TG prediction from [n_tgs, d_model] to [n_tgs]
        self.gene_pred_dense = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1, bias=False),
            )
        
        # Adds TF->TG weights at the end to directly modify the final TG expression predictions
        # Optional TF→TG shortcut that adapts to any TF/TG set
        if self.use_shortcut:
            self.shortcut_layer = TFtoTGShortcut(
                d_model, 
                use_motif_mask, 
                lambda_l1, 
                lambda_l2, 
                topk, 
                shortcut_dropout,
                motif_mask_threshold,
                motif_prior_scale
                )
                    
    def forward(self, atac_windows, tf_expr, tf_ids, tg_ids, bias=None, motif_mask=None):
        """
        atac_windows : [batch_size, n_window, 1]
        tf_expr      : [batch_size, n_tfs]
        tf_ids       : LongTensor [n_tfs]
        tg_ids       : LongTensor [n_tgs]
        bias         : [batch_size, n_tgs, n_window] or None   (distance bias)
        returns      : [batch_size, n_tgs]
        """
        batch_size, n_window, _ = atac_windows.shape
        device = atac_windows.device
        
        atac_windows = torch.nan_to_num(atac_windows, nan=0.0, posinf=1e6, neginf=-1e6).clamp_(-10.0, 10.0)
        tf_expr      = torch.nan_to_num(tf_expr,      nan=0.0, posinf=1e6, neginf=-1e6).clamp_(-10.0, 10.0)

        # ----- ATAC encoding -----
        win_emb = self.atac_acc_dense_input_layer(atac_windows)       # [batch_size,n_window,d_model]
        pos = torch.arange(n_window, device=device, dtype=torch.float32)
        win_emb = win_emb + self.posenc(pos, bsz=batch_size).transpose(0, 1)  # [batch_size,n_window,d_model]
        if self.use_gradient_checkpointing and self.training:
            win_emb = self._encode_with_checkpointing(win_emb)
        else:
            win_emb = self.encoder(win_emb)                               # [batch_size,n_window,d_model]

        # Guard against indexing errors with TF and TG vocab
        assert tf_ids.dtype == torch.long and tg_ids.dtype == torch.long
        assert tf_ids.min().item() >= 0 and tf_ids.max().item() < self.tf_identity_emb.num_embeddings, \
            f"tf_ids out of range: max={tf_ids.max().item()}, vocab={self.tf_identity_emb.num_embeddings}"
        assert tg_ids.min().item() >= 0 and tg_ids.max().item() < self.tg_query_emb.num_embeddings, \
            f"tg_ids out of range: max={tg_ids.max().item()}, vocab={self.tg_query_emb.num_embeddings}"

        # ----- TF embeddings -----
        # Embed the TFs
        tf_id_emb = self.tf_identity_emb(tf_ids)                        # [n_tfs,d_model]
        
        # Pass the TF expression through the dense layer to project to d_model
        tf_expr_emb = self.tf_expr_dense_input_layer(tf_expr.unsqueeze(-1))        # [batch_size,n_tfs,d_model]
        
        # Add TF expression projection to TF identity embedding
        tf_emb = tf_expr_emb + tf_id_emb.unsqueeze(0)                     

        # ----- TF-ATAC and ATAC-TF Cross Attention -----
        tf_cross   = self._maybe_checkpoint(self.cross_tf_to_atac, tf_emb, win_emb)        # [batch_size,n_tfs,d_model]
        atac_cross = self._maybe_checkpoint(self.cross_atac_to_tf, win_emb, tf_emb)        # [batch_size,n_window,d_model]

        # Attention pooling
        tf_repr, _   = self.tf_to_atac_cross_attn_pool(tf_cross)                    # [batch_size,d_model]
        atac_repr, _ = self.atac_to_tf_cross_attn_pool(atac_cross)                # [batch_size,d_model]
        
        # Dense layer projection back to d_model
        tf_atac_cross_attn_output = self.pooled_cross_attn_dense_layer(torch.cat([tf_repr, atac_repr], dim=-1))  # [batch_size,d_model]
        
        # ----- TG - ATAC Cross Attention -----
        tg_base = self.tg_query_emb(tg_ids).unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size,n_tgs,d_model]

        # Check for distance bias
        attn_bias = None
        if self.use_bias and (bias is not None):
            attn_bias = bias
            if attn_bias.dim() == 3:
                attn_bias = attn_bias.unsqueeze(1)  # [batch_size,1,n_tgs,n_window]
            assert attn_bias.shape[0] == batch_size and attn_bias.shape[-2:] == (tg_base.size(1), win_emb.size(1)), \
                f"Bias shape {attn_bias.shape} not compatible with (n_tgs={tg_base.size(1)}, n_window={win_emb.size(1)})"
            if attn_bias.shape[1] == 1:
                attn_bias = attn_bias.expand(batch_size, self.num_heads, tg_base.size(1), win_emb.size(1))
            attn_bias = torch.nan_to_num(attn_bias, nan=0.0, posinf=1e4, neginf=-1e4)
            attn_bias = (self.bias_scale * attn_bias).clamp_(-20.0, 20.0)

        # TG-ATAC cross attention with distance bias
        tg_cross = self.cross_tg_to_atac(tg_base, win_emb, attn_bias=attn_bias)                             # [batch_size,n_tgs,d_model]
        if self.use_gradient_checkpointing and self.training and attn_bias is not None:
            def tg_attn_fn(tg_x, win_x, bias_x):
                return self.cross_tg_to_atac(tg_x, win_x, attn_bias=bias_x)
            tg_cross = cp.checkpoint(tg_attn_fn, tg_base, win_emb, attn_bias, use_reentrant=False)
        elif self.use_gradient_checkpointing and self.training:
            # no bias branch
            tg_cross = self._maybe_checkpoint(self.cross_tg_to_atac, tg_base, win_emb)
        else:
            tg_cross = self.cross_tg_to_atac(tg_base, win_emb, attn_bias=attn_bias)
        
        # Expand the TF-ATAC cross attention output for each TG
        tf_atac_cross_attn_output = tf_atac_cross_attn_output.unsqueeze(1).expand(-1, tg_cross.size(1), -1) # [batch_size,n_tgs,d_model]
        
        # For each TG, sum the outputs from the TG-ATAC cross attention 
        # with the fused TF-ATAC and ATAC-TF cross attn output
        tg_cross_attn_repr = tg_cross + tf_atac_cross_attn_output                                                      # [batch_size,n_tgs,d_model]
        

        # ----- TG identity to Cross-attention output dot product -----
        # Create TG identity embeddings
        tg_emb = self.tg_identity_emb(tg_ids)                   # [n_tgs,d_model]
        
        # Dot product between teach TG's global context from attention and the TG identity embedding
        #     If the TG ID embedding and global context agree, supports TG expression
        tg_similarity_to_attn_output = (tg_cross_attn_repr * tg_emb).sum(dim=-1)     # [batch_size,n_tgs]
        
        # Sum the TG similarity score and the per-TG cross attention output
        # If both similarit and weighted context features are high, predicts higher expression
        tg_pred = tg_similarity_to_attn_output + self.gene_pred_dense(tg_cross_attn_repr).squeeze(-1)
        
        if motif_mask is not None and self.use_motif_mask:
            assert motif_mask.shape == (tg_emb.size(0), tf_id_emb.size(0)), \
                f"Motif mask shape {motif_mask.shape} does not match (n_tgs, n_tfs)"
        
        # ----- Optional TF→TG shortcut without fixed matrix -----
        attn = None
        if self.use_shortcut:
            # Calculate the direct TF-TG attention (similarity of TF and TG embeddings * )
            tf_tg_shortcut_output, attn = self.shortcut_layer(
                tg_emb, tf_id_emb, tf_expr, motif_mask=motif_mask
                )
            
            # Add the output from the direct TF-TG expression layer to the predicted TG expression from attention
            tg_pred = tg_pred + tf_tg_shortcut_output
            
            # Store the last attention output
            self.last_attn = attn
        

        return tg_pred, attn
    
    def _encode_with_checkpointing(self, win_emb):
        """
        Run self.encoder with gradient checkpointing over its layers.
        Assumes self.encoder is an nn.TransformerEncoder.
        """
        encoder = self.encoder

        for layer in encoder.layers:
            # each layer: (x) -> (x_out)
            win_emb = cp.checkpoint(layer, win_emb, use_reentrant=False)

        if encoder.norm is not None:
            win_emb = encoder.norm(win_emb)

        return win_emb

    def _maybe_checkpoint(self, module, *args):
        """
        Conditionally run `module(*args)` under torch.utils.checkpoint.
        Assumes all args are Tensors (no kwargs).
        """
        if self.use_gradient_checkpointing and self.training:
            def fn(*inputs):
                return module(*inputs)
            return cp.checkpoint(fn, *args, use_reentrant=False)
        else:
            return module(*args)