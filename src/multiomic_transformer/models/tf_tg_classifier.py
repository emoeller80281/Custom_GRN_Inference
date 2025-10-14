import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import dropout_edge

class GRN_GAT_Bidirectional(nn.Module):
    """
    Bidirectional Graph Attention Network for GRN inference
    - Supports edge dropout during training for regularization
    - Learns TF↔TG embeddings jointly
    - Predicts regulatory edge probabilities per TF-TG pair
    """

    def __init__(self, in_node_feats, in_edge_feats, hidden_dim=64, heads=4, dropout=0.3, edge_dropout_p=0.2):
        super().__init__()

        self.edge_dropout_p = edge_dropout_p
        self.dropout = dropout

        # Bidirectional GAT with edge features
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

        # Classifier: combines TF and TG embeddings
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr, pairs):
        """
        x: [N, F_node]          node features
        edge_index: [2, E]      directed edge list (TF->TG)
        edge_attr: [E, F_edge]  edge features (e.g., reg_potential)
        pairs: [E, 2]           TF–TG pairs for prediction
        """

        # Edge dropout for regularization
        if self.training and self.edge_dropout_p > 0:
            edge_index, edge_mask = dropout_edge(edge_index, p=self.edge_dropout_p)
            edge_attr = edge_attr[edge_mask]

        # Make edges bidirectional
        rev_edge_index = edge_index.flip(0)
        edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

        # GAT message passing
        h = F.elu(self.gat1(x, edge_index, edge_attr))
        h = F.elu(self.gat2(h, edge_index, edge_attr))

        # Gather embeddings for labeled TF–TG pairs ---
        tf_emb = h[pairs[:, 0]]
        tg_emb = h[pairs[:, 1]]

        # Edge classification
        logits = self.classifier(torch.cat([tf_emb, tg_emb], dim=1))
        return logits.squeeze(-1)
