import torch
import torch.nn as nn
import torch.nn.functional as F
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
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Add the peak-gene distance to the attention scores if its passed
        if attn_bias is not None:
            attn_scores = attn_scores + attn_bias
        
        # Softmax is applied to obtain attention probabilities
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Multiply by values to obtain the final output
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
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout, num_tf, num_tg):
        super(MultiomicTransformer, self).__init__()
        
        self.d_model = d_model
        self.dropout = dropout
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_tg = num_tg
        
        # Create a TG embedding (doesn't use TG expression)
        self.tg_emb = nn.Parameter(torch.empty(num_tg, d_model))
        nn.init.xavier_uniform_(self.tg_emb)
        
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
            nn.Linear(1, self.d_ff, bias=False),        # Projects each TF independently
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=False),
            nn.LayerNorm(d_model)
        )
        
        # Positional encoding of the windows
        self.posenc = PositionalEmbedding(d_model)
        
        # Encode ATAC Windows
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=self.d_ff,
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
        # with a Tanh activation function (from UNADON Yang and Ma 2023)
        self.out_dense = nn.Sequential(
            nn.Linear(2*d_model, d_ff, bias=False),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(d_ff, d_model, bias=False),
            nn.LayerNorm(d_model)
        )
        
        # Adds TF->TG weights at the end to directly modify the final TG expression predictions
        self.tf_tg_weights = nn.Parameter(torch.empty(num_tf, num_tg))
        nn.init.xavier_uniform_(self.tf_tg_weights)
                
        self.gene_pred_dense = nn.Linear(d_model, 1)
    
    def forward(self, atac_windows, tf_expr, bias=None):
        # ----- Embed ATAC windows -----
        win_emb = self.atac_window_dense_layer(atac_windows)      # [B, num_windows, d_model]

        # ----- Embed TF expression -----
        tf_raw = tf_expr                                          # [B, num_tf]
        tf_emb = self.tf_dense_layer(tf_expr.unsqueeze(-1))       # [B, num_tf, d_model]

        # ----- Positional encoding -----
        positions = torch.arange(win_emb.size(1), device=win_emb.device).float()
        pos_emb = self.posenc(positions, bsz=win_emb.size(0))     # [seq_len, B, d_model]
        pos_emb = pos_emb.transpose(0, 1)                         # [B, seq_len, d_model]
        win_emb = win_emb + pos_emb
        win_emb = self.encoder(win_emb)                           # [B, num_windows, d_model]

        # ----- Cross attention TF <-> ATAC -----
        tf_cross   = self.cross_tf_to_atac(tf_emb, win_emb)       # [B, num_tf, d_model]
        atac_cross = self.cross_atac_to_tf(win_emb, tf_emb)       # [B, num_windows, d_model]

        # ----- TG queries ATAC (with distance bias) -----
        B = atac_windows.size(0)
        tg_emb = self.tg_emb.unsqueeze(0).expand(B, -1, -1)       # [B, num_tg, d_model]
        if bias is not None:
            # bias: [B, num_tg, num_windows] -> [B, 1, num_tg, num_windows]
            bias = bias.unsqueeze(1)
            # expand across heads -> [B, num_heads, num_tg, num_windows]
            bias = bias.expand(-1, self.num_heads, -1, -1)         # [B, num_heads, num_tg, num_windows]
        else:
            bias = None
        tg_cross = self.cross_tg_to_atac(tg_emb, win_emb, attn_bias=bias)  # [B, num_tg, d_model]

        # ----- Global context pooling -----
        tf_repr, _   = self.tf_pool(tf_cross)                     # [B, d_model]
        atac_repr, _ = self.atac_pool(atac_cross)                 # [B, d_model]
        fused_repr   = self.out_dense(torch.cat([tf_repr, atac_repr], dim=-1))  # [B, d_model]

        # ----- Combine global + gene-specific -----
        fused_repr = fused_repr.unsqueeze(1).expand(-1, self.num_tg, -1)   # [B, num_tg, d_model]
        tg_repr    = tg_cross + fused_repr                                # [B, num_tg, d_model]

        # ----- Predict TG expression -----
        gene_logits = self.gene_pred_dense(tg_repr).view(B, self.num_tg)   # [B, num_tg]

        # Direct TF->TG shortcut
        tf_expr_logits = tf_raw @ self.tf_tg_weights              # [B, num_tg]
        gene_logits = gene_logits + tf_expr_logits

        return gene_logits
            
        
        
        
