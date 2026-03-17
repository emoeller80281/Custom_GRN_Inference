import torch
import torch.nn as nn
import torch.nn.functional as F
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
        """Generates sinusoidal positional embeddings for a given sequence length and model dimension."""
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
    def __init__(self, 
                d_model, 
                num_heads, 
                num_layers, 
                d_ff, 
                dropout,
                tf_vocab_size, 
                tg_vocab_size, 
                bias_scale=2.0,
                use_bias=True,
                 ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias_scale = bias_scale

        # TF and TG embedding tables - generates identities for TFs and TGs
        self.tf_identity_emb = nn.Embedding(tf_vocab_size, d_model)
        self.tg_query_emb = nn.Embedding(tg_vocab_size, d_model)        # 
        
        # ATAC windows do not have embeddings, their position and accessibility
        # are used to inform TFs and TGs
        
        # TF dense layer - Projects the TF RNA-seq expression data to [n_tfs x d_model]
        self.tf_expr_dense_input_layer = nn.Sequential(
            nn.Linear(1, d_ff),        # Projects each TF independently
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
            nn.LayerNorm(d_model)
        )
        
        # ATAC dense layer - Projects the ATAC window accessibility to [n_windows x d_model]
        self.atac_acc_dense_input_layer = nn.Sequential(
            nn.Linear(1, d_ff), 
            nn.SiLU(),
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
        
        # TF Q -> ATAC K Cross Attention Pooling
        self.tf_to_atac_cross_attn_pool = AttentionPooling(d_model)     # [batch_size, d_model]
        
        # ATAC Q -> TF KCross Attention Pooling
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
        
                    
    def forward(
        self,
        atac_windows,
        tf_expr,
        tf_ids,
        tg_ids,
        bias=None,
    ):
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
        
        # Add positional encoding
        pos = torch.arange(n_window, device=device, dtype=torch.float32)
        win_emb = win_emb + self.posenc(pos, bsz=batch_size).transpose(0, 1)  # [batch_size,n_window,d_model]
        win_emb = self.encoder(win_emb)                               # [batch_size,n_window,d_model]

        # ----- TF embeddings -----
        # Embed the TFs
        tf_id_emb = self.tf_identity_emb(tf_ids)                        # [n_tfs,d_model]
        
        # Pass the TF expression through the dense layer to project to d_model
        tf_expr_emb = self.tf_expr_dense_input_layer(tf_expr.unsqueeze(-1))        # [batch_size,n_tfs,d_model]
        
        # Add TF expression projection to TF identity embedding
        tf_emb = tf_expr_emb + tf_id_emb.unsqueeze(0)                     

        # ----- TF-ATAC and ATAC-TF Cross Attention -----
        tf_cross   = self.cross_tf_to_atac(tf_emb, win_emb)        # [batch_size,n_tfs,d_model]
        atac_cross = self.cross_atac_to_tf(win_emb, tf_emb)        # [batch_size,n_window,d_model]

        # TF-ATAC and ATAC-TF Attention pooling
        tf_repr, _   = self.tf_to_atac_cross_attn_pool(tf_cross)                  # [batch_size,d_model]
        atac_repr, _ = self.atac_to_tf_cross_attn_pool(atac_cross)                # [batch_size,d_model]
        
        # Concatenate the sample-level TF and ATAC cross attention representations to create a fused 
        # representation of the global TF and ATAC context for the sample
        combined_attention_pooling = torch.cat([tf_repr, atac_repr], dim=-1)  # [batch_size, 2*d_model]
        
        # Dense layer projection back down from [2*d_model] to [d_model]
        tf_atac_cross_attn_output = self.pooled_cross_attn_dense_layer(combined_attention_pooling)  # [batch_size,d_model]
        
        # ----- TG - ATAC Cross Attention -----
        tg_base = self.tg_query_emb(tg_ids).unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size,n_tgs,d_model]

        # Check for distance bias
        attn_bias = None
        if self.use_bias and (bias is not None):
            attn_bias = bias
            if attn_bias.dim() == 3:
                attn_bias = attn_bias.unsqueeze(1)  # [batch_size,1,n_tgs,n_window]

            if attn_bias.shape[1] == 1:
                attn_bias = attn_bias.expand(batch_size, self.num_heads, tg_base.size(1), win_emb.size(1))
                
            attn_bias = torch.nan_to_num(attn_bias, nan=0.0, posinf=1e4, neginf=-1e4)
            attn_bias = (self.bias_scale * attn_bias).clamp_(-20.0, 20.0)

        # TG-ATAC cross attention with distance bias
        tg_cross = self.cross_tg_to_atac(tg_base, win_emb, attn_bias=attn_bias)        
        
        # Expand the TF-ATAC cross attention output for each TG
        n_tgs = tg_cross.size(1)
        tf_atac_cross_attn_output = tf_atac_cross_attn_output.unsqueeze(1).expand(-1, n_tgs, -1)
               
        # For each TG, sum the outputs from the TG-ATAC cross attention 
        # with the fused TF-ATAC and ATAC-TF cross attn output
        tg_cross_attn_repr = tg_cross + tf_atac_cross_attn_output
        
        # Pass the TG cross attention representation through the final dense layer to get a prediction for each TG
        tg_pred = self.gene_pred_dense(tg_cross_attn_repr).squeeze(-1)

        return tg_pred