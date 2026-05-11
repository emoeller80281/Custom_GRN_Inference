import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt
import wandb
import time

from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision

from pathlib import Path
import sys
PROJECT_DIR = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER")
SRC_DIR = str(PROJECT_DIR / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from multiomic_transformer.models.model_classifier import MultiomicTransformer
from multiomic_transformer.utils.experiment_classifier_handler import ExperimentHandler

# ================================================================
# STEP-BASED WARMUP + COSINE LR
# ================================================================
class LitMultiomicTransformer(pl.LightningModule):
    def __init__(
        self, 
        exp: ExperimentHandler,
        model: MultiomicTransformer = None,
        ground_truth_edges: pd.DataFrame | None = None,
        ground_truth_name: str = "chip_atlas",
        ground_truth_negative_ratio: int = 10,
        max_ground_truth_pairs: int | None = 20000,
        ground_truth_seed: int = 1337,
        exclude_ground_truth_from_supervised_loss: bool = True,
        validation_metric_sample_size: int = 200_000,
        validation_metric_samples_per_batch: int = 4096,
        metric_thresholds: int = 256,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["exp", "model", "ground_truth_edges"])

        self.model = model
        self.exp = exp
        self.learning_rate = exp.initial_lr
        self.scheduler_reduction_factor = exp.scheduler_reduction_factor
        self.scheduler_patience_epochs = exp.scheduler_patience_epochs
        self.scheduler_cooldown_epochs = exp.scheduler_cooldown_epochs

        self.tf_scaler = exp.tf_scaler
        self.tg_scaler = exp.tg_scaler
        self.criterion = nn.BCEWithLogitsLoss()

        self.val_auroc = BinaryAUROC(thresholds=metric_thresholds, sync_on_compute=True)
        self.val_auprc = BinaryAveragePrecision(thresholds=metric_thresholds, sync_on_compute=True)
        self.validation_metric_sample_size = int(validation_metric_sample_size)
        self.validation_metric_samples_per_batch = int(validation_metric_samples_per_batch)
        self.val_probs = []
        self.val_targets = []
        self.val_metric_sample_count = 0

        self.ground_truth_name = ground_truth_name
        self.ground_truth_negative_ratio = int(ground_truth_negative_ratio)
        self.max_ground_truth_pairs = max_ground_truth_pairs
        self.ground_truth_seed = int(ground_truth_seed)
        self.exclude_ground_truth_from_supervised_loss = bool(exclude_ground_truth_from_supervised_loss)
        self.gt_pair_df = self._prepare_ground_truth_pairs(ground_truth_edges)
        self.gt_probs = []
        self.gt_targets = []
        self.gt_eval_pair_count = 0
        self.gt_eval_pos_count = 0
        self.train_supervised_gt_excluded_count = 0
        self.val_supervised_gt_excluded_count = 0
        
        self.epoch_step_times = []

    def forward(self, atac_wins, tf_tensor, tg_tensor, tf_ids=None, tg_ids=None, bias=None, return_logits=False):
        return self.model(
            atac_wins,
            tf_tensor,
            tg_tensor,
            tf_ids=tf_ids,
            tg_ids=tg_ids,
            bias=bias,
            return_logits=return_logits,
        )

    @staticmethod
    def _match_pred_target_shape(preds, targets):
        preds = preds.reshape(-1)
        targets = targets.reshape(-1)
        return preds, targets

    def _append_validation_metric_sample(self, probs, targets):
        max_samples = max(0, int(self.validation_metric_sample_size))
        per_batch = max(0, int(self.validation_metric_samples_per_batch))
        if max_samples == 0 or per_batch == 0:
            return

        probs = probs.detach().reshape(-1).float().cpu()
        targets = targets.detach().reshape(-1).int().cpu()
        n = probs.numel()
        if n == 0:
            return

        k = min(n, per_batch)
        if k < n:
            idx = torch.randint(n, (k,), device=probs.device)
            probs = probs.index_select(0, idx)
            targets = targets.index_select(0, idx)

        self.val_probs.append(probs)
        self.val_targets.append(targets)
        self.val_metric_sample_count += int(probs.numel())

        if self.val_metric_sample_count > max_samples * 2:
            self._compact_validation_metric_sample()

    def _compact_validation_metric_sample(self):
        max_samples = max(0, int(self.validation_metric_sample_size))
        if max_samples == 0 or not self.val_probs:
            self.val_probs.clear()
            self.val_targets.clear()
            self.val_metric_sample_count = 0
            return

        probs = torch.cat(self.val_probs, dim=0).view(-1)
        targets = torch.cat(self.val_targets, dim=0).view(-1).int()
        n = probs.numel()
        if n > max_samples:
            idx = torch.randperm(n, device=probs.device)[:max_samples]
            probs = probs.index_select(0, idx)
            targets = targets.index_select(0, idx)

        self.val_probs = [probs.contiguous()]
        self.val_targets = [targets.contiguous()]
        self.val_metric_sample_count = int(probs.numel())

    @staticmethod
    def _sample_curve_points(x, y, n_points: int = 10):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.size == 0 or y.size == 0:
            return x, y

        order = np.argsort(x, kind="stable")
        x = x[order]
        y = y[order]

        unique_x, inverse = np.unique(x, return_inverse=True)
        if unique_x.size != x.size:
            y_reduced = np.zeros_like(unique_x, dtype=np.float64)
            for i in range(unique_x.size):
                y_reduced[i] = y[inverse == i].max()
            x, y = unique_x, y_reduced

        if x.size <= n_points:
            return x, y

        sample_x = np.linspace(float(x.min()), float(x.max()), int(n_points))
        sample_y = np.interp(sample_x, x, y)
        return sample_x, sample_y

    @staticmethod
    def _resolve_batch(batch):
        if len(batch) == 8:
            atac_wins, tf_tensor, tg_tensor, labels, bias, tf_ids, tg_ids, motif_mask = batch
            return atac_wins, tf_tensor, tg_tensor, labels, bias, tf_ids, tg_ids, motif_mask

        if len(batch) in (10, 12):
            (
                atac_wins,
                tf_tensor,
                tg_tensor,
                labels,
                pair_tf_idx,
                pair_tg_idx,
                shared_bias,
                shared_tf_ids,
                shared_tg_ids,
                motif_mask,
                *_
            ) = batch

            tf_ids = shared_tf_ids.index_select(0, pair_tf_idx)
            tg_ids = shared_tg_ids.index_select(0, pair_tg_idx)
            bias = shared_bias.index_select(0, pair_tg_idx)
            return atac_wins, tf_tensor, tg_tensor, labels, bias, tf_ids, tg_ids, motif_mask

        raise ValueError(f"Unexpected batch structure with {len(batch)} fields")

    @staticmethod
    def _standardize_name(name):
        if not isinstance(name, str):
            return name
        return name.upper()

    def _prepare_ground_truth_pairs(self, edge_df: pd.DataFrame | None) -> pd.DataFrame | None:
        if edge_df is None:
            return None
        if not {"Source", "Target"}.issubset(edge_df.columns):
            raise ValueError("ground_truth_edges must contain Source and Target columns")

        dataset = getattr(self.exp, "dataset", None)
        if dataset is None:
            raise ValueError("exp.dataset must be created before passing ground_truth_edges")

        tf_name2id = {
            self._standardize_name(name): int(i)
            for i, name in enumerate(getattr(dataset, "tf_names", []))
        }
        tg_name2id = {
            self._standardize_name(name): int(i)
            for i, name in enumerate(getattr(dataset, "tg_names", []))
        }

        positives = (
            edge_df[["Source", "Target"]]
            .dropna()
            .assign(
                Source=lambda df: df["Source"].astype(str).str.upper(),
                Target=lambda df: df["Target"].astype(str).str.upper(),
            )
            .drop_duplicates()
        )
        positives = positives[
            positives["Source"].isin(tf_name2id)
            & positives["Target"].isin(tg_name2id)
        ].copy()

        if positives.empty:
            raise ValueError("No ChIP-Atlas edges overlap the model TF/TG vocabulary")

        pos_keys = set(zip(positives["Source"], positives["Target"]))
        pos_tfs = sorted(positives["Source"].unique())
        pos_tgs = sorted(positives["Target"].unique())

        candidate_negatives = [
            (tf, tg)
            for tf in pos_tfs
            for tg in pos_tgs
            if (tf, tg) not in pos_keys
        ]
        rng = random.Random(self.ground_truth_seed)
        rng.shuffle(candidate_negatives)
        n_neg = min(
            len(candidate_negatives),
            max(len(positives) * self.ground_truth_negative_ratio, len(positives)),
        )

        positives["label"] = 1
        negatives = pd.DataFrame(candidate_negatives[:n_neg], columns=["Source", "Target"])
        negatives["label"] = 0

        pairs = pd.concat([positives, negatives], ignore_index=True)
        if self.max_ground_truth_pairs is not None and len(pairs) > self.max_ground_truth_pairs:
            pos = pairs[pairs["label"] == 1]
            neg = pairs[pairs["label"] == 0]
            neg_keep = max(0, int(self.max_ground_truth_pairs) - len(pos))
            neg = neg.sample(n=min(neg_keep, len(neg)), random_state=self.ground_truth_seed)
            pairs = pd.concat([pos, neg], ignore_index=True)

        pairs["tf_id"] = pairs["Source"].map(tf_name2id).astype(int)
        pairs["tg_id"] = pairs["Target"].map(tg_name2id).astype(int)
        self.gt_positive_pair_ids = set(
            zip(
                pairs.loc[pairs["label"] == 1, "tf_id"].astype(int),
                pairs.loc[pairs["label"] == 1, "tg_id"].astype(int),
            )
        )
        return pairs.sample(frac=1.0, random_state=self.ground_truth_seed).reset_index(drop=True)

    def _build_batch_ground_truth_pairs(self, shared_tf_ids, shared_tg_ids):
        positive_pair_ids = getattr(self, "gt_positive_pair_ids", set())
        if not positive_pair_ids:
            return None

        tf_ids_cpu = [int(i) for i in shared_tf_ids.detach().cpu().long().tolist()]
        tg_ids_cpu = [int(i) for i in shared_tg_ids.detach().cpu().long().tolist()]
        tf_id_set = set(tf_ids_cpu)
        tg_id_set = set(tg_ids_cpu)

        batch_pos = [
            (tf_id, tg_id)
            for tf_id, tg_id in positive_pair_ids
            if tf_id in tf_id_set and tg_id in tg_id_set
        ]
        if not batch_pos:
            return None

        batch_pos_set = set(batch_pos)
        n_possible_neg = len(tf_ids_cpu) * len(tg_ids_cpu) - len(batch_pos_set)
        if n_possible_neg <= 0:
            return None

        max_neg = min(
            n_possible_neg,
            max(len(batch_pos), len(batch_pos) * self.ground_truth_negative_ratio),
        )
        rng = random.Random(self.ground_truth_seed + int(self.current_epoch))
        batch_neg_set = set()
        max_attempts = max(1000, max_neg * 20)
        attempts = 0
        while len(batch_neg_set) < max_neg and attempts < max_attempts:
            attempts += 1
            candidate = (rng.choice(tf_ids_cpu), rng.choice(tg_ids_cpu))
            if candidate not in batch_pos_set:
                batch_neg_set.add(candidate)

        if len(batch_neg_set) < max_neg:
            for tf_id in tf_ids_cpu:
                for tg_id in tg_ids_cpu:
                    candidate = (tf_id, tg_id)
                    if candidate not in batch_pos_set:
                        batch_neg_set.add(candidate)
                        if len(batch_neg_set) >= max_neg:
                            break
                if len(batch_neg_set) >= max_neg:
                    break

        batch_neg = list(batch_neg_set)

        pairs = pd.DataFrame(batch_pos + batch_neg, columns=["tf_id", "tg_id"])
        pairs["label"] = [1] * len(batch_pos) + [0] * len(batch_neg)
        if self.max_ground_truth_pairs is not None and len(pairs) > self.max_ground_truth_pairs:
            pos = pairs[pairs["label"] == 1]
            neg = pairs[pairs["label"] == 0]
            neg_keep = max(0, int(self.max_ground_truth_pairs) - len(pos))
            neg = neg.sample(
                n=min(neg_keep, len(neg)),
                random_state=self.ground_truth_seed + int(self.current_epoch),
            )
            pairs = pd.concat([pos, neg], ignore_index=True)

        return pairs.sample(
            frac=1.0,
            random_state=self.ground_truth_seed + int(self.current_epoch),
        ).reset_index(drop=True)

    def _supervised_pair_keep_mask(self, tf_ids, tg_ids, stage: str):
        if not self.exclude_ground_truth_from_supervised_loss:
            return None

        positive_pair_ids = getattr(self, "gt_positive_pair_ids", set())
        if not positive_pair_ids:
            return None

        tf_ids_cpu = tf_ids.detach().cpu().long().view(-1).tolist()
        tg_ids_cpu = tg_ids.detach().cpu().long().view(-1).tolist()

        keep = [
            [
                (int(tf_id), int(tg_id)) not in positive_pair_ids
                for tg_id in tg_ids_cpu
            ]
            for tf_id in tf_ids_cpu
        ]

        keep_mask = torch.tensor(keep, dtype=torch.bool, device=tf_ids.device)
        n_excluded = int((~keep_mask).sum().item())
        if stage == "train":
            self.train_supervised_gt_excluded_count += n_excluded
        elif stage == "val":
            self.val_supervised_gt_excluded_count += n_excluded
        return keep_mask

    def _update_ground_truth_metrics(self, batch):
        if self.gt_pair_df is None:
            return

        if len(batch) == 8:
            atac_wins, tf_tensor, tg_tensor, _, shared_bias, shared_tf_ids, shared_tg_ids, _ = batch
        elif len(batch) in (10, 12):
            (
                atac_wins,
                tf_tensor,
                tg_tensor,
                _,
                _,
                _,
                shared_bias,
                shared_tf_ids,
                shared_tg_ids,
                _,
                *extra,
            ) = batch
            if len(extra) >= 2:
                tf_tensor, tg_tensor = extra[:2]
                tf_tensor = tf_tensor.unsqueeze(0)
                tg_tensor = tg_tensor.unsqueeze(0)
                shared_bias = shared_bias.unsqueeze(0)
        else:
            return

        shared_tf_ids_cpu = shared_tf_ids.detach().cpu().long()
        shared_tg_ids_cpu = shared_tg_ids.detach().cpu().long()
        tf_local = {int(gid): i for i, gid in enumerate(shared_tf_ids_cpu.tolist())}
        tg_local = {int(gid): i for i, gid in enumerate(shared_tg_ids_cpu.tolist())}

        eval_df = self._build_batch_ground_truth_pairs(shared_tf_ids, shared_tg_ids)
        if eval_df is None or eval_df["label"].nunique() < 2:
            return
        self.gt_eval_pair_count += int(len(eval_df))
        self.gt_eval_pos_count += int(eval_df["label"].sum())

        device = atac_wins.device
        pair_tf_idx = torch.tensor(
            [tf_local[int(i)] for i in eval_df["tf_id"].to_numpy()],
            dtype=torch.long,
            device=device,
        )
        pair_tg_idx = torch.tensor(
            [tg_local[int(i)] for i in eval_df["tg_id"].to_numpy()],
            dtype=torch.long,
            device=device,
        )
        labels = torch.tensor(eval_df["label"].to_numpy(), dtype=torch.float32, device=device)

        tf_eval = self.tf_scaler.transform(tf_tensor[:1], shared_tf_ids)
        tg_eval = self.tg_scaler.transform(tg_tensor[:1], shared_tg_ids)

        logits = self.model(
            atac_wins[:1],
            tf_eval,
            tg_eval,
            tf_ids=shared_tf_ids,
            tg_ids=shared_tg_ids,
            bias=shared_bias[:1],
            return_logits=True,
        )

        pair_logits = logits[0, pair_tf_idx, pair_tg_idx]
        probs = torch.sigmoid(pair_logits.reshape(-1))
        self.gt_probs.append(probs.detach().cpu())
        self.gt_targets.append(labels.detach().cpu())

    def training_step(self, batch, batch_idx):
        atac_wins, tf_tensor, tg_tensor, labels, bias, tf_ids, tg_ids, _ = self._resolve_batch(batch)
        
        tf_tensor = self.tf_scaler.transform(tf_tensor, tf_ids)
        tg_tensor = self.tg_scaler.transform(tg_tensor, tg_ids)
        
        logits = self.model(
            atac_wins,
            tf_tensor,
            tg_tensor,
            tf_ids=tf_ids,
            tg_ids=tg_ids,
            bias=bias,
            return_logits=True,
        )

        labels = labels.float()
        keep_mask = self._supervised_pair_keep_mask(tf_ids, tg_ids, stage="train")
        if keep_mask is not None:
            if keep_mask.dim() == 2 and logits.dim() == 3:
                keep_mask = keep_mask.unsqueeze(0).expand(logits.shape[0], -1, -1)
            logits = logits[keep_mask]
            labels = labels[keep_mask]
            if labels.numel() == 0:
                return logits.sum() * 0.0
        else:
            logits, labels = self._match_pred_target_shape(logits, labels)

        train_loss = self.criterion(logits, labels)
        probs = torch.sigmoid(logits)
        train_accuracy = ((probs >= 0.5).float() == labels).float().mean()

        self.log(
            "train/loss",
            train_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "train/accuracy",
            train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        return train_loss

    def validation_step(self, batch, batch_idx):
        atac_wins, tf_tensor, tg_tensor, labels, bias, tf_ids, tg_ids, _ = self._resolve_batch(batch)
        
        tf_tensor = self.tf_scaler.transform(tf_tensor, tf_ids)
        tg_tensor = self.tg_scaler.transform(tg_tensor, tg_ids)

        logits = self.model(
            atac_wins,
            tf_tensor,
            tg_tensor,
            tf_ids=tf_ids,
            tg_ids=tg_ids,
            bias=bias,
            return_logits=True,
        )

        labels = labels.float()
        keep_mask = self._supervised_pair_keep_mask(tf_ids, tg_ids, stage="val")
        if keep_mask is not None:
            if keep_mask.dim() == 2 and logits.dim() == 3:
                keep_mask = keep_mask.unsqueeze(0).expand(logits.shape[0], -1, -1)
            logits = logits[keep_mask]
            labels = labels[keep_mask]
            if labels.numel() == 0:
                self._update_ground_truth_metrics(batch)
                return
        else:
            logits, labels = self._match_pred_target_shape(logits, labels)

        val_loss = self.criterion(logits, labels)
        probs = torch.sigmoid(logits)
        val_accuracy = ((probs >= 0.5).float() == labels).float().mean()

        self.val_auroc.update(probs.detach(), labels.int())
        self.val_auprc.update(probs.detach(), labels.int())
        self._append_validation_metric_sample(probs, labels)

        self.log(
            "val/loss",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log(
            "val/accuracy",
            val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self._update_ground_truth_metrics(batch)

    def on_validation_epoch_start(self):
        self.val_probs.clear()
        self.val_targets.clear()
        self.gt_probs.clear()
        self.gt_targets.clear()
        self.gt_eval_pair_count = 0
        self.gt_eval_pos_count = 0
        self.val_supervised_gt_excluded_count = 0
        self.val_metric_sample_count = 0

    def on_validation_epoch_end(self):
        probs = None
        targets = None
        if self.val_probs:
            self._compact_validation_metric_sample()
            probs = torch.cat(self.val_probs, dim=0).view(-1)
            targets = torch.cat(self.val_targets, dim=0).view(-1).int()

            self.val_probs.clear()
            self.val_targets.clear()
            self.val_metric_sample_count = 0

            auroc = self.val_auroc.compute()
            auprc = self.val_auprc.compute()

            self.log("val/roc_auc", auroc, prog_bar=True, sync_dist=False)
            self.log("val/pr_auc", auprc, prog_bar=True, sync_dist=False)
            self.log("val/metric_curve_sample_size", float(probs.numel()), prog_bar=False, sync_dist=False)

            self.val_auroc.reset()
            self.val_auprc.reset()

        if self.gt_pair_df is not None:
            prefix = self.ground_truth_name
            self.log(f"{prefix}/n_pairs", float(self.gt_eval_pair_count), prog_bar=False, sync_dist=False)
            self.log(f"{prefix}/n_pos", float(self.gt_eval_pos_count), prog_bar=False, sync_dist=False)
            self.log(
                f"{prefix}/val_supervised_labels_excluded",
                float(self.val_supervised_gt_excluded_count),
                prog_bar=False,
                sync_dist=False,
            )

        if self.gt_probs:
            gt_probs = torch.cat(self.gt_probs, dim=0).view(-1)
            gt_targets = torch.cat(self.gt_targets, dim=0).view(-1).int()
            self.gt_probs.clear()
            self.gt_targets.clear()
            prefix = self.ground_truth_name

            if gt_targets.unique().numel() >= 2:
                gt_auroc = BinaryAUROC()(gt_probs, gt_targets)
                gt_auprc = BinaryAveragePrecision()(gt_probs, gt_targets)
                self.log(f"{prefix}/auroc", gt_auroc, prog_bar=True, sync_dist=False)
                self.log(f"{prefix}/auprc", gt_auprc, prog_bar=True, sync_dist=False)

                if self.trainer.is_global_zero:
                    try:
                        from sklearn.metrics import precision_recall_curve, roc_curve

                        gt_p_np = gt_probs.detach().cpu().numpy()
                        gt_y_np = gt_targets.detach().cpu().numpy()
                        gt_pre, gt_rec, _ = precision_recall_curve(gt_y_np, gt_p_np)
                        gt_fpr, gt_tpr, _ = roc_curve(gt_y_np, gt_p_np)
                        gt_rec, gt_pre = self._sample_curve_points(gt_rec, gt_pre, n_points=10)
                        gt_fpr, gt_tpr = self._sample_curve_points(gt_fpr, gt_tpr, n_points=10)

                        self.logger.experiment.log({
                            f"{prefix} Curves/AUPRC Precision vs Recall": wandb.plot.line_series(
                                [gt_rec],
                                [gt_pre],
                                keys=[f"{prefix} Precision"],
                                xname="Recall",
                            ),
                            f"{prefix} Curves/AUROC TPR vs FPR": wandb.plot.line_series(
                                [gt_fpr],
                                [gt_tpr],
                                keys=[f"{prefix} TPR"],
                                xname="FPR",
                            ),
                        })
                    except Exception as e:
                        print(f"[WARN] {prefix} PR/ROC curve error:", e)

        if probs is not None and self.trainer.is_global_zero:
            try:
                p_np = probs.detach().cpu().numpy()
                y_np = targets.detach().cpu().numpy()

                # Keep the same lightweight curve logging pattern used elsewhere in the repo.
                from sklearn.metrics import precision_recall_curve, roc_curve

                pre, rec, _ = precision_recall_curve(y_np, p_np)
                fpr, tpr, _ = roc_curve(y_np, p_np)
                rec, pre = self._sample_curve_points(rec, pre, n_points=10)
                fpr, tpr = self._sample_curve_points(fpr, tpr, n_points=10)

                self.logger.experiment.log({
                    "Validation Curves/AUPRC Precision vs Recall": wandb.plot.line_series(
                        [rec],
                        [pre],
                        keys=["Precision"],
                        xname="Recall",
                    ),
                    "Validation Curves/AUROC TPR vs FPR": wandb.plot.line_series(
                        [fpr],
                        [tpr],
                        keys=["TPR"],
                        xname="FPR",
                    ),
                })
            except Exception as e:
                print("[WARN] PR/ROC curve error:", e)

    def on_train_epoch_start(self):
        self.train_supervised_gt_excluded_count = 0

    def on_train_batch_start(self, batch, batch_idx):
        self._batch_start_time = time.time()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if hasattr(self, "_batch_start_time"):
            step_time = time.time() - self._batch_start_time

            self.epoch_step_times.append(step_time)

            self.log(
                "train/step_time_sec",
                step_time,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
            )

    def on_train_epoch_end(self):
        if self.gt_pair_df is not None:
            self.log(
                f"{self.ground_truth_name}/train_supervised_labels_excluded",
                float(self.train_supervised_gt_excluded_count),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
            )

        if len(self.epoch_step_times) > 0:
            total_step_time = float(np.sum(self.epoch_step_times))

            self.log(
                "train/total_train_time_sec",
                total_step_time,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

            self.epoch_step_times = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_reduction_factor,
            patience=self.scheduler_patience_epochs,
            cooldown=self.scheduler_cooldown_epochs,
            threshold=1e-4,
            min_lr=1e-7,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
                "name": "lr_plateau",
            },
        }
