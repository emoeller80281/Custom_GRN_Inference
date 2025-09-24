import torch
import torch.nn as nn
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
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (useful for preventing attention to certain parts like padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
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
        
    def forward(self, Q, K, V, mask=None):
        # Apply linear transformations and split heads
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        # Perform scaled dot-product attention
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
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

    def forward(self, query, key_value, mask=None):
        # query: [B, len_query, d_model]
        # key_value: [B, len_key_val, d_model]
        attn_out = self.attn(query, key_value, key_value, mask)
        out = self.norm(query + self.dropout(attn_out))  # residual
        return out
    
class MultiomicTransformer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, num_tf, num_windows, num_tg):
        super(MultiomicTransformer, self).__init__()
        
        self.d_model = d_model
        self.dropout = dropout
        self.d_ff = d_ff
        self.num_heads = num_heads
        
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
            num_layers=2
        )
        
        # Cross Attention in both directions (TF Q -> ATAC KV & ATAC Q -> TF KV)
        self.cross_tf_to_atac = CrossAttention(d_model, num_heads, dropout)
        self.cross_atac_to_tf = CrossAttention(d_model, num_heads, dropout)
        
        # Pass the output of the transformer through a dense network
        # with a Tanh activation function (from UNADON Yang and Ma 2023)
        self.out_dense = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(d_ff, d_model, bias=False),
            nn.LayerNorm(d_model)
        )
        
        self.gene_pred_dense = nn.Linear((num_tf + num_windows) * d_model, num_tg)
    
    def forward(self, atac_windows, tf_expr):
        # Use a dense layer to embed the ATAC windows
        win_emb = self.atac_window_dense_layer(atac_windows)                        # [B, num_windows, d_model]
        
        # Use a dense layer to embed each TF separately
        # tf_expr: [B, num_tf]
        tf_expr = tf_expr.unsqueeze(-1)                                             # [B, num_tf, 1]
        tf_emb = self.tf_dense_layer(tf_expr)                                       # [B, num_tf, d_model]
        
        # Add positional encodings to the windows based on the number of windows
        # atac_windows: [B, num_windows, num_features]
        positions = torch.arange(win_emb.size(1), device=win_emb.device).float()    
        pos_emb = self.posenc(positions, bsz=win_emb.size(0))                       # [seq_len, B, d_model]
        pos_emb = pos_emb.transpose(0, 1)                                           # [B, seq_len, d_model]
        win_emb = win_emb + pos_emb
        win_emb = self.encoder(win_emb)                                             # [B, num_windows, d_model]
        
        # Run cross-attention both ways
        tf_cross = self.cross_tf_to_atac(tf_emb, win_emb)                           # [B, num_tf, d_model]
        atac_cross = self.cross_atac_to_tf(win_emb, tf_emb)                         # [B, num_windows, d_model]
        
        # Concatenate the results from cross-attention
        fused_cross = torch.cat([tf_cross, atac_cross], dim=1)                      # [B, num_tf+num_windows, d_model]
        
        # Final dense layer for the bi-directional cross-attention
        fused_cross = self.out_dense(fused_cross)                                   

        # Project and flatten                           
        fused_cross = fused_cross.flatten(start_dim=1)                                      # [B, num_tf+num_windows * d_model]
        
        # Final linear projection to the dimensionality of the target genes
        gene_logits = self.gene_pred_dense(fused_cross)
        
        return gene_logits
        
        
        
        
