import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import dropout_edge

class GRN_GAT_Encoder(nn.Module):
    """
    Self-supervised Graph Attention Encoder (no labels).
    Learns node embeddings by maximizing mutual information
    between local (node) and global (graph) representations.
    """

    def __init__(self, in_node_feats, in_edge_feats, hidden_dim=128, heads=4, dropout=0.3, edge_dropout_p=0.2):
        super().__init__()
        self.edge_dropout_p = edge_dropout_p
        self.dropout = dropout

        self.gat1 = GATConv(
            in_channels=in_node_feats,
            out_channels=hidden_dim,
            heads=heads,
            edge_dim=in_edge_feats,
            dropout=dropout
        )
        self.gat2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=1,
            edge_dim=in_edge_feats,
            dropout=dropout
        )

        # Readout: global graph embedding
        self.readout = lambda h: torch.sigmoid(torch.mean(h, dim=0))

        # Projection head for contrastive stability
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        if self.training and self.edge_dropout_p > 0:
            edge_index, edge_mask = dropout_edge(edge_index, p=self.edge_dropout_p)
            edge_attr = edge_attr[edge_mask]

        rev_edge_index = edge_index.flip(0)
        edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

        h = F.elu(self.gat1(x, edge_index, edge_attr))
        h = F.elu(self.gat2(h, edge_index, edge_attr))
        h_proj = self.proj(h)
        g = self.readout(h_proj)
        return h_proj, g

# Simple classifier head on top of frozen encoder
class EdgeClassifier(nn.Module):
    def __init__(self, base_model, embed_dim):
        super().__init__()
        self.encoder = base_model
        in_dim = embed_dim * 4  # [tf, tg, tf*tg, |tf-tg|]

        self.bn = nn.BatchNorm1d(in_dim, affine=True)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 2*embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(2*embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 1)
        )
        # Init: last layer bias slightly negative
        nn.init.constant_(self.classifier[-1].bias, -0.2)

    def forward(self, x, edge_index, edge_attr, pairs):
        h, _ = self.encoder(x, edge_index, edge_attr)  # <-- no F.normalize
        tf = h[pairs[:,0]]; tg = h[pairs[:,1]]
        z = torch.cat([tf, tg, tf*tg, (tf - tg).abs()], dim=1)
        z = self.bn(z)  # BatchNorm instead of LayerNorm
        return self.classifier(z).squeeze(-1)
    