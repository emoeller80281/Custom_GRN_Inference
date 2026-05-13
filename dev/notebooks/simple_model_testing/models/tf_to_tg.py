import torch
import torch.nn as nn


class TFTGRegulationModel(nn.Module):
    def __init__(
        self,
        pretrained_tf_peak_model,
        d_model,
        num_heads=4,
        dropout=0.1,
        tf_peak_chunk_size=256,
        use_expression_conditioned_query=True,
    ):
        super().__init__()

        self.tf_peak_model = pretrained_tf_peak_model
        self.tf_peak_chunk_size = tf_peak_chunk_size
        self.use_expression_conditioned_query = use_expression_conditioned_query

        # Frozen TF-peak feature extractor
        for p in self.tf_peak_model.parameters():
            p.requires_grad = False

        self.peak_feature_proj = nn.Sequential(
            nn.Linear(4, d_model),  # binding, accessibility, distance_scaled, distance_weight
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.tf_expr_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.tg_expr_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.tg_query_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        self.peak_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(d_model)

        # peak_context + tf_expr + tg_expr + tg_identity
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 4, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def compute_binding_logits(
        self,
        tf_embedding_flat,
        tf_mask_flat,
        peak_seq_flat,
    ):
        """
        Frozen TF-DNA model forward pass, chunked over TF-peak pairs.

        Inputs
        ------
        tf_embedding_flat : [N, T, D]
        tf_mask_flat : [N, T]
        peak_seq_flat : [N, L, 4]

        Returns
        -------
        logits : [N]
        """

        self.tf_peak_model.eval()
        logits_chunks = []

        with torch.no_grad():
            for start in range(0, tf_embedding_flat.shape[0], self.tf_peak_chunk_size):
                end = start + self.tf_peak_chunk_size

                logits_chunk = self.tf_peak_model(
                    tf_embedding=tf_embedding_flat[start:end],
                    tf_mask=tf_mask_flat[start:end],
                    peak_embedding=peak_seq_flat[start:end],
                )

                logits_chunks.append(logits_chunk)

        return torch.cat(logits_chunks, dim=0)

    @staticmethod
    def pool_cell_logits(
        cell_logits,
        cell_mask=None,
        mode="lse",
        temperature=1.0,
    ):
        """
        cell_logits: [E, C]
        cell_mask: [E, C], True for real cells, False for padded cells
        """

        if cell_mask is None:
            cell_mask = torch.ones_like(cell_logits, dtype=torch.bool)

        if mode == "mean":
            masked_logits = cell_logits.masked_fill(~cell_mask, 0.0)
            denom = cell_mask.sum(dim=1).clamp_min(1)
            return masked_logits.sum(dim=1) / denom

        elif mode == "max":
            masked_logits = cell_logits.masked_fill(~cell_mask, float("-inf"))
            return masked_logits.max(dim=1).values

        elif mode == "lse":
            masked_logits = cell_logits.masked_fill(~cell_mask, float("-inf"))
            n_cells = cell_mask.sum(dim=1).clamp_min(1)

            return temperature * (
                torch.logsumexp(masked_logits / temperature, dim=1)
                - torch.log(n_cells.float())
            )

        else:
            raise ValueError(f"Unknown pooling mode: {mode}")

    def forward(
        self,
        tf_embedding,
        tf_mask,
        peak_sequences,
        peak_accessibility,
        peak_distance,
        tf_expression,
        tg_expression,
        tg_embedding,
        cell_mask,
        peak_mask=None,
        pooling_mode: str = "lse",
        pooling_temperature: float = 1.0,
    ):
        """
        Bag-level forward pass.

        This computes TF-DNA binding once per TF-TG edge and peak,
        then reuses those binding scores across sampled cells.

        Parameters
        ----------
        tf_embedding : [E, C, T, D]
        tf_mask : [E, C, T]
        peak_sequences : [E, C, P, L, 4]
        peak_accessibility : [E, C, P]
        peak_distance : [E, C, P]
        tf_expression : [E, C]
        tg_expression : [E, C]
        tg_embedding : [E, C, d_model]
        cell_mask : [E, C]
        peak_mask : [E, C, P], optional

        Returns
        -------
        edge_logits : [E]
        cell_logits : [E, C]
        """

        E, C = cell_mask.shape
        _, _, P, L, nuc_dim = peak_sequences.shape
        EC = E * C

        # ------------------------------------------------------------
        # 1. Cell-invariant edge-level tensors
        # ------------------------------------------------------------
        # These are repeated across cells in your current dataloader.
        # Use only the first cell to avoid C-fold redundant TF-DNA inference.
        tf_embedding_edge = tf_embedding[:, 0]          # [E, T, D]
        tf_mask_edge = tf_mask[:, 0]                    # [E, T]
        peak_sequences_edge = peak_sequences[:, 0]      # [E, P, L, 4]
        peak_distance_edge = peak_distance[:, 0]        # [E, P]
        tg_embedding_edge = tg_embedding[:, 0]          # [E, d_model]

        if peak_mask is not None:
            peak_mask_edge = peak_mask[:, 0]            # [E, P]
        else:
            peak_mask_edge = None

        # ------------------------------------------------------------
        # 2. Frozen TF-DNA binding model: [E, P] only
        # ------------------------------------------------------------
        tf_embedding_rep = tf_embedding_edge[:, None].expand(
            E, P, *tf_embedding_edge.shape[1:]
        )
        tf_embedding_flat = tf_embedding_rep.reshape(
            E * P, *tf_embedding_edge.shape[1:]
        )

        tf_mask_rep = tf_mask_edge[:, None].expand(E, P, tf_mask_edge.shape[1])
        tf_mask_flat = tf_mask_rep.reshape(E * P, tf_mask_edge.shape[1])

        peak_seq_flat = peak_sequences_edge.reshape(E * P, L, nuc_dim)

        binding_logits = self.compute_binding_logits(
            tf_embedding_flat=tf_embedding_flat,
            tf_mask_flat=tf_mask_flat,
            peak_seq_flat=peak_seq_flat,
        ).reshape(E, P)

        binding_score = torch.sigmoid(binding_logits)  # [E, P]

        if peak_mask_edge is not None:
            binding_score = binding_score.masked_fill(~peak_mask_edge, 0.0)

        # Reuse TF-peak binding score across cells
        binding_score = binding_score[:, None, :].expand(E, C, P)  # [E, C, P]

        # ------------------------------------------------------------
        # 3. Distance features
        # ------------------------------------------------------------
        abs_distance = peak_distance_edge.abs()
        distance_scaled = torch.clamp(abs_distance / 250_000.0, 0.0, 1.0)
        distance_weight = torch.exp(-abs_distance / 50_000.0)

        if peak_mask_edge is not None:
            distance_scaled = distance_scaled.masked_fill(~peak_mask_edge, 0.0)
            distance_weight = distance_weight.masked_fill(~peak_mask_edge, 0.0)

        distance_scaled = distance_scaled[:, None, :].expand(E, C, P)
        distance_weight = distance_weight[:, None, :].expand(E, C, P)

        # ------------------------------------------------------------
        # 4. Cell-specific peak features
        # ------------------------------------------------------------
        if peak_mask is not None:
            peak_accessibility = peak_accessibility.masked_fill(~peak_mask, 0.0)

        peak_features = torch.stack(
            [
                binding_score,
                peak_accessibility,
                distance_scaled,
                distance_weight,
            ],
            dim=-1,
        )  # [E, C, P, 4]

        peak_features = peak_features.reshape(EC, P, 4)
        peak_tokens = self.peak_feature_proj(peak_features)  # [E*C, P, d_model]

        # ------------------------------------------------------------
        # 5. Expression and TG identity tokens
        # ------------------------------------------------------------
        tf_expr_token = self.tf_expr_proj(
            tf_expression.reshape(EC, 1)
        )  # [E*C, d_model]

        tg_expr_token = self.tg_expr_proj(
            tg_expression.reshape(EC, 1)
        )  # [E*C, d_model]

        tg_embedding_cell = tg_embedding_edge[:, None, :].expand(
            E, C, tg_embedding_edge.shape[-1]
        ).reshape(EC, tg_embedding_edge.shape[-1])  # [E*C, d_model]

        if self.use_expression_conditioned_query:
            tg_query_input = tg_embedding_cell + tf_expr_token + tg_expr_token
        else:
            tg_query_input = tg_embedding_cell

        tg_query = self.tg_query_proj(tg_query_input).unsqueeze(1)  # [E*C, 1, d_model]

        # ------------------------------------------------------------
        # 6. TG query attends to linked peak tokens
        # ------------------------------------------------------------
        key_padding_mask = None

        if peak_mask_edge is not None:
            key_padding_mask = peak_mask_edge[:, None, :].expand(E, C, P)
            key_padding_mask = ~key_padding_mask.reshape(EC, P)  # True = ignore

        peak_context, _ = self.peak_attention(
            query=tg_query,
            key=peak_tokens,
            value=peak_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        peak_context = self.norm(peak_context.squeeze(1))  # [E*C, d_model]

        # ------------------------------------------------------------
        # 7. Cell-level logits
        # ------------------------------------------------------------
        final = torch.cat(
            [
                peak_context,
                tf_expr_token,
                tg_expr_token,
                tg_embedding_cell,
            ],
            dim=-1,
        )  # [E*C, d_model * 4]

        cell_logits = self.classifier(final).squeeze(-1)  # [E*C]
        cell_logits = cell_logits.reshape(E, C)           # [E, C]

        # ------------------------------------------------------------
        # 8. Pool cell logits into edge logits
        # ------------------------------------------------------------
        edge_logits = self.pool_cell_logits(
            cell_logits,
            cell_mask=cell_mask,
            mode=pooling_mode,
            temperature=pooling_temperature,
        )  # [E]

        return edge_logits, cell_logits