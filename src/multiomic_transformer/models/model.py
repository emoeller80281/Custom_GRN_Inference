import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINEConv
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
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
        # Expected: Q,K,V = [B,H,L, d_k]
        assert Q.dim() == 4 and K.dim() == 4 and V.dim() == 4, \
            f"Q,K,V must be 4D: got {Q.shape}, {K.shape}, {V.shape}"
        B,H,Lq,Dq = Q.shape
        Bk,Hk,Lk,Dk = K.shape
        Bv,Hv,Lv,Dv = V.shape
        assert (B,H) == (Bk,Hk) == (Bv,Hv), f"Batch/heads mismatch: Q{(B,H)} K{(Bk,Hk)} V{(Bv,Hv)}"
        assert Dq == Dk == Dv, f"Head dim mismatch: {Dq} vs {Dk} vs {Dv}"
        assert Lk == Lv, f"K/V length mismatch: Lk={Lk}, Lv={Lv}"

        if attn_bias is not None:
            attn_bias = attn_bias.to(Q.device, Q.dtype)
            # Accept [B,1,Lq,Lk] or [B,H,Lq,Lk] (already expanded upstream)
            assert attn_bias.shape[0] in (1, B), f"attn_bias batch {attn_bias.shape[0]} vs B={B}"
            assert attn_bias.shape[1] in (1, H), f"attn_bias heads {attn_bias.shape[1]} vs H={H}"
            assert attn_bias.shape[-2:] == (Lq, Lk), f"attn_bias last dims {attn_bias.shape[-2:]} vs {(Lq,Lk)}"

        if mask is not None:
            # Expect broadcastable [B,1,Lq,Lk] or [B,H,Lq,Lk]
            assert mask.shape[-2:] == (Lq, Lk), f"mask last dims {mask.shape[-2:]} vs {(Lq,Lk)}"

        # Finite checks catch NaN/Inf early
        for name, t in (("Q",Q), ("K",K), ("V",V)):
            if not torch.isfinite(t).all():
                raise RuntimeError(f"{name} contains NaN/Inf")

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, torch.finfo(attn_scores.dtype).min)
        if attn_bias is not None:
            attn_scores = attn_scores + attn_bias
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
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
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value, mask=None, attn_bias=None):
        # query: [B, len_query, d_model]
        # key_value: [B, len_key_val, d_model]
        attn_out = self.attn(query, key_value, key_value, mask, attn_bias=attn_bias)
        out = self.norm(query + self.dropout(attn_out))  # residual
        return out

class TFtoTGShortcut(nn.Module):
    def __init__(self, d_model, use_motif_mask=False, lambda_l1=1e-4, lambda_l2=0.0,
                 topk=None, dropout_p=0.2):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))  # learnable scaling
        self.use_motif_mask = use_motif_mask
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.topk = topk  # keep only top-k TFs per TG if not None
        self.dropout_p = dropout_p

    def forward(self, tg_dec, tf_base, tf_expr, motif_mask=None):
        """
        tg_dec     : [G_eval, D]   TG embeddings
        tf_base    : [T_eval, D]   TF embeddings
        tf_expr    : [B, T_eval]   TF expression values
        motif_mask : [G_eval, T_eval] (0/1) prior: TF motif present in TG-linked peaks
        """
        # Compute similarity TG↔TF
        sim = torch.matmul(tg_dec, tf_base.T) / math.sqrt(tf_base.size(1))  # [G,T]

        # Apply motif mask if provided
        if self.use_motif_mask and motif_mask is not None:
            sim = sim.masked_fill(motif_mask == 0, float("-inf"))

        # Softmax attention
        attn = torch.softmax(sim, dim=-1)  # [G,T]
        
        # Protect against NaNs (all -inf rows after masking -> NaN after softmax)
        attn = torch.nan_to_num(attn, nan=0.0)

        # Optional sparsity: keep only top-k TFs per TG
        if self.topk is not None:
            topk_vals, topk_idx = torch.topk(attn, self.topk, dim=-1)
            mask = torch.zeros_like(attn)
            mask.scatter_(1, topk_idx, 1.0)
            attn = attn * mask
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

        # Weighted TF → TG contribution
        tf_scalar = tf_expr @ attn.T

        # Apply dropout (on shortcut contribution)
        tf_scalar = F.dropout(tf_scalar, p=self.dropout_p, training=self.training)

        # Save attention for regularization and interpretability
        self.attn = attn

        return self.scale * tf_scalar, attn  # return both logits contribution and edge weights

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
        # x: [B, N, d_model]
        scores = torch.matmul(x, self.query) / (self.d_model ** 0.5)  # [B, N]
        weights = F.softmax(scores, dim=1).unsqueeze(-1)          # [B, N, 1]
        pooled = torch.sum(weights * x, dim=1)                   # [B, d_model]
        return pooled, weights

class MultiomicTransformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout,
                 tf_vocab_size, tg_vocab_size, 
                 bias_scale=2.0,
                 use_bias=True,
                 use_shortcut=True, 
                 use_motif_mask=False, 
                 lambda_l1=1e-4, 
                 lambda_l2=0.0,
                 topk=None,
                 shortcut_dropout=0.2
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

        # Embedding tables (fixed vocab; you index subsets by ids at runtime)
        self.tf_emb_table = nn.Embedding(tf_vocab_size, d_model)
        self.tg_emb_table = nn.Embedding(tg_vocab_size, d_model)
        self.tg_decoder_table = nn.Embedding(tg_vocab_size, d_model)
        
        # Dense layer to pass the ATAC-seq windows into
        self.atac_window_dense_layer = nn.Sequential(
            nn.Linear(1, d_ff, bias=False), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
            nn.LayerNorm(d_model)
        )
        
        # Dense layer to pass the TF RNA-seq data into
        self.tf_dense_layer = nn.Sequential(
            nn.Linear(1, d_ff, bias=False),        # Projects each TF independently
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
            nn.LayerNorm(d_model)
        )
        
        # Encode ATAC Windows
        self.posenc = PositionalEmbedding(d_model)
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
        
        # Cross Attention between TF->peak, peak->TF, and TG->peak
        self.cross_tf_to_atac = CrossAttention(d_model, num_heads, dropout)
        self.cross_atac_to_tf = CrossAttention(d_model, num_heads, dropout)
        self.cross_tg_to_atac = CrossAttention(d_model, num_heads, dropout)
        
        # Add attention pooling to summarize the most important windows and TFs
        self.tf_pool = AttentionPooling(d_model)
        self.atac_pool = AttentionPooling(d_model)
        
        # Pass the output of the transformer through a dense network
        self.out_dense = nn.Sequential(
            nn.Linear(2*d_model, d_ff, bias=False),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(d_ff, d_model, bias=False),
            nn.LayerNorm(d_model)
        )
        
        self.gene_pred_dense = nn.Linear(d_model, 1)
        
        # Adds TF->TG weights at the end to directly modify the final TG expression predictions
        # Optional TF→TG shortcut that adapts to any TF/TG set
        if self.use_shortcut:
            self.shortcut_layer = TFtoTGShortcut(d_model, use_motif_mask, lambda_l1, lambda_l2, topk, shortcut_dropout)
                    
    def forward(self, atac_windows, tf_expr, tf_ids, tg_ids, bias=None, motif_mask=None):
        """
        atac_windows : [B, W, 1]
        tf_expr      : [B, T_eval]       (values must correspond to tf_ids order)
        tf_ids       : LongTensor [T_eval]
        tg_ids       : LongTensor [G_eval]
        bias         : [B, G_eval, W] or None   (distance bias)
        returns      : [B, G_eval]
        """
        B, W, _ = atac_windows.shape
        device = atac_windows.device

        # ----- ATAC encoding -----
        win_emb = self.atac_window_dense_layer(atac_windows)       # [B,W,D]
        pos = torch.arange(W, device=device, dtype=torch.float32)
        win_emb = win_emb + self.posenc(pos, bsz=B).transpose(0, 1)  # [B,W,D]
        win_emb = self.encoder(win_emb)                               # [B,W,D]

        # Guard against indexing errors with TF and TG vocab
        assert tf_ids.dtype == torch.long and tg_ids.dtype == torch.long
        assert tf_ids.min().item() >= 0 and tf_ids.max().item() < self.tf_emb_table.num_embeddings, \
            f"tf_ids out of range: max={tf_ids.max().item()}, vocab={self.tf_emb_table.num_embeddings}"
        assert tg_ids.min().item() >= 0 and tg_ids.max().item() < self.tg_emb_table.num_embeddings, \
            f"tg_ids out of range: max={tg_ids.max().item()}, vocab={self.tg_emb_table.num_embeddings}"

        # ----- TF embeddings -----
        tf_base = self.tf_emb_table(tf_ids)                        # [T_eval,D]
        tf_emb = self.tf_dense_layer(tf_expr.unsqueeze(-1))        # [B,T_eval,D]
        tf_emb = tf_emb + tf_base.unsqueeze(0)                     # inject identity

        # cross TF <-> ATAC
        tf_cross   = self.cross_tf_to_atac(tf_emb, win_emb)        # [B,T_eval,D]
        atac_cross = self.cross_atac_to_tf(win_emb, tf_emb)        # [B,W,D]

        # ----- TG queries ATAC -----
        tg_base = self.tg_emb_table(tg_ids).unsqueeze(0).expand(B, -1, -1)  # [B,G_eval,D]

        attn_bias = None
        if self.use_bias and (bias is not None):
            attn_bias = bias
            if attn_bias.dim() == 3:
                attn_bias = attn_bias.unsqueeze(1)  # [B,1,G_eval,W]
            assert attn_bias.shape[0] == B and attn_bias.shape[-2:] == (tg_base.size(1), win_emb.size(1)), \
                f"Bias shape {attn_bias.shape} not compatible with (G_eval={tg_base.size(1)}, W={win_emb.size(1)})"
            # Broadcast over heads
            if attn_bias.shape[1] == 1:
                attn_bias = attn_bias.expand(B, self.num_heads, tg_base.size(1), win_emb.size(1))
            attn_bias = self.bias_scale * attn_bias

        tg_cross = self.cross_tg_to_atac(tg_base, win_emb, attn_bias=attn_bias)   # [B,G_eval,D]

        # ----- Fuse global context -----
        tf_repr, _   = self.tf_pool(tf_cross)                    # [B,D]
        atac_repr, _ = self.atac_pool(atac_cross)                # [B,D]
        fused = self.out_dense(torch.cat([tf_repr, atac_repr], dim=-1))  # [B,D]
        fused = fused.unsqueeze(1).expand(-1, tg_cross.size(1), -1)      # [B,G_eval,D]
        tg_repr = tg_cross + fused                                        # [B,G_eval,D]

        # ----- TG-aware prediction head (dot product) -----
        tg_dec = self.tg_decoder_table(tg_ids)                   # [G_eval,D]
        logits = torch.einsum("bgd,gd->bg", tg_repr, tg_dec)     # [B,G_eval]
        
        logits = logits + self.gene_pred_dense(tg_repr).squeeze(-1)
                
        if motif_mask is not None and self.use_motif_mask:
            assert motif_mask.shape == (tg_dec.size(0), tf_base.size(0)), \
                f"Motif mask shape {motif_mask.shape} does not match (G_eval, T_eval)"
        
        # ----- Optional TF→TG shortcut without fixed matrix -----
        attn = None
        if self.use_shortcut:
            tf_scalar, attn = self.shortcut_layer(tg_dec, tf_base, tf_expr, motif_mask=motif_mask)
            logits = logits + tf_scalar
            self.last_attn = attn
        

        return logits, attn
