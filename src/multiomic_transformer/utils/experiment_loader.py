import json
import os
import random
from venv import create
from flask import g
from matplotlib.ticker import FuncFormatter, MultipleLocator
from numpy import gradient
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import sys
import logging
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from typing import Set, Tuple, Optional
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from multiomic_transformer.datasets.dataset_refactor import SimpleScaler
from multiomic_transformer.models.model import MultiomicTransformer

logging.basicConfig(level=logging.INFO, format='%(message)s')

class ExperimentLoader:
    def __init__(self, experiment_dir: str, experiment_name: str, model_num: int):
        
        assert os.path.exists(experiment_dir), f"Experiment directory {experiment_dir} does not exist."
        
        self.experiment_dir = Path(experiment_dir)
        self.experiment_name = experiment_name
        self.model_num = model_num
        
        if "chr19" in [p.name for p in (Path(experiment_dir) / experiment_name).iterdir()]:
            self.model_training_dir = Path(f"{experiment_dir}/{experiment_name}/chr19/model_training_00{model_num}")
        else:
            self.model_training_dir = Path(f"{experiment_dir}/{experiment_name}/model_training_00{model_num}")
                
        assert self.model_training_dir.exists(), f"Model training directory {self.model_training_dir} does not exist."
        
        # Load the run parameters saved during model training
        if not (self.model_training_dir / "run_parameters.json").exists():
            logging.warning(f"WARNING: Run parameters file {self.model_training_dir / 'run_parameters.json'} does not exist.")
            self.model_training_params = None
        else:
            self.model_training_params = self._load_json(self.model_training_dir / "run_parameters.json")
        
        # Open the full experiment settings file
        if not (self.experiment_dir / self.experiment_name / "run_parameters_long.csv").exists():
            logging.warning(f"WARNING: Experiment settings file {self.experiment_dir / self.experiment_name / 'run_parameters_long.csv'} does not exist.")
            self.experiment_settings_df = None
        else:
            self.experiment_settings_df = pd.read_csv(self.experiment_dir / self.experiment_name / "run_parameters_long.csv")
        
        # Load the GPU usage log and format it into a more usable format
        self.gpu_usage_df, self.gpu_usage_mean_per_sec_df, self.gpu_memory_limit_gib = self._format_gpu_usage_file()
        
        # Load the model training log data
        self.training_df = pd.read_csv(self.model_training_dir / "training_log.csv")
        
        # Loads the TF and TG names in order by their index
        self.tf_names = self._load_json(self.model_training_dir / "tf_names_ordered.json")
        self.tg_names = self._load_json(self.model_training_dir / "tg_names_ordered.json")
        
        # Model and training state will be loaded when load_trained_model is called
        self.model = None
        self.state = None
        self.test_loader = None
        self.tg_scaler = None
        self.tf_scaler = None
        
        # Gradient Attribution dataframe will be loaded when load_gradient_attribution is called
        self.gradient_attribution_df = None
        
        # Transcription Factor Knockout score dataframe will be loaded when load_tf_knockout is called
        self.tf_knockout_df = None
        
        # Model forward pass predictions vs true values
        self.tg_prediction_df = None
        self.tg_true_df = None
        
        # Model evaluation metric results
        self.raw_results_df = None
        self.results_df = None
        self.per_tf_all_df = None
        self.per_tf_summary_df = None
        
        # Model evaluation metric results with ground truth
        self.df_with_ground_truth = None
        self.gt_labeled_dfs = {}
        self.auroc_auprc_scores = None
        
    def load_trained_model(self, checkpoint_file):
        """
        Loads a trained model given a checkpoint file and loads the corresponding model parameters

        Parameters:
        checkpoint_file (str): The name of the checkpoint file to load

        Returns:
        None
        """
        if not os.path.exists(self.model_training_dir / checkpoint_file):
            logging.warning(f"Checkpoint file {checkpoint_file} does not exist in {self.model_training_dir}. Attempting to locate the last checkpoint.")
            checkpoint_file = self._locate_last_checkpoint()
            logging.info(f"Located checkpoint file: {checkpoint_file}")

        # Pull out architecture hyperparameters
        params = self.model_training_params
        d_model   = params.get("d_model")
        num_heads = params.get("num_heads")
        num_layers = params.get("num_layers")
        d_ff      = params.get("d_ff")
        dropout   = params.get("dropout", 0.0)
        use_shortcut   = params.get("use_shortcut", False)
        use_dist_bias  = params.get("use_dist_bias", False)
        use_motif_mask = params.get("use_motif_mask", False)

        # Load test loader
        self.test_loader = torch.load(self.model_training_dir / "test_loader.pt", weights_only=False)

        # Load the model checkpoint and state dictionary
        ckpt_path = os.path.join(self.model_training_dir, checkpoint_file)
        self.state = torch.load(ckpt_path, map_location="cpu")
        
        # Recreate the model from the training parameters
        self.model = MultiomicTransformer(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            tf_vocab_size=len(self.state["tf_scaler_mean"]),
            tg_vocab_size=len(self.state["tg_scaler_mean"]),
            use_bias=use_dist_bias,
            use_shortcut=use_shortcut,
            use_motif_mask=use_motif_mask,
        )

        if isinstance(self.state, dict) and "model_state_dict" in self.state:
            missing, unexpected = self.model.load_state_dict(
                self.state["model_state_dict"], strict=False
            )
            if len(missing) > 0:
                logging.info(f"Missing keys: {missing}")
            if len(unexpected) > 0:
                logging.info(f"Unexpected keys: {unexpected}")
        elif isinstance(self.state, dict) and "model_state_dict" not in self.state:
            missing, unexpected = self.model.load_state_dict(self.state, strict=False)
            if len(missing) > 0:
                logging.info(f"Missing keys: {missing}")
            if len(unexpected) > 0:
                logging.info(f"Unexpected keys: {unexpected}")
        else:
            missing, unexpected = self.model.load_state_dict(self.state, strict=False)
            if len(missing) > 0:
                logging.info(f"Missing keys: {missing}")
            if len(unexpected) > 0:
                logging.info(f"Unexpected keys: {unexpected}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

        # Rebuild the scalers from the training parameters
        self.tg_scaler = SimpleScaler(
            mean=torch.as_tensor(self.state["tg_scaler_mean"], device=self.device, dtype=torch.float32),
            std=torch.as_tensor(self.state["tg_scaler_std"],  device=self.device, dtype=torch.float32),
        )
        self.tf_scaler = SimpleScaler(
            mean=torch.as_tensor(self.state["tf_scaler_mean"], device=self.device, dtype=torch.float32),
            std=torch.as_tensor(self.state["tf_scaler_std"],  device=self.device, dtype=torch.float32),
        )
    
    def load_grn(self, method="gradient attribution"):
        """
        Loads a GRN dataframe given a method. The dataframe contains the source transcription factor, target gene, and score.

        Parameters:
        method (str): The method to use. Must be 'Gradient Attribution' or 'TF Knockout'.

        Returns:
        pd.DataFrame: The GRN dataframe containing the source transcription factor, target gene, and score.
        """
        method = method.lower()
        
        assert method in ["gradient attribution", "tf knockout"], \
            f"Invalid GRN method {method}. Must be 'Gradient Attribution' or 'TF Knockout'."        
        
        if method == "gradient attribution":
            score_file = self.model_training_dir / "tf_tg_grad_attribution.npy"
            
        elif method == "tf knockout":
            score_file = self.model_training_dir / "tf_tg_fullmodel_knockout.npy"
            
        assert score_file.exists(), f"GRN file for method {method} {score_file} does not exist."
        
        score = np.load(score_file).astype(np.float32)
        assert score.shape == (len(self.tf_names), len(self.tg_names))

        score = np.nan_to_num(score, nan=0.0)
        score_abs = np.abs(score)

        # Calculate per-TF robust z-score
        median_val = np.median(score_abs, axis=1, keepdims=True)
        mad = np.median(np.abs(score_abs - median_val), axis=1, keepdims=True) + 1e-6
        score = (score_abs - median_val) / mad
        
        T, G = score_abs.shape
        tf_idx, tg_idx = np.meshgrid(np.arange(T), np.arange(G), indexing="ij")
        
        tf_idx = tf_idx.ravel()
        tg_idx = tg_idx.ravel()

        df = pd.DataFrame({
            "Source": np.asarray(self.tf_names, dtype=object)[tf_idx],
            "Target": np.asarray(self.tg_names, dtype=object)[tg_idx],
            "Score": score.ravel(),
        })
        
        df["Source"] = df["Source"].astype(str).str.upper()
        df["Target"] = df["Target"].astype(str).str.upper()
        
        return df
    
    def load_eval_results(self):      
        eval_files = [
            "pooled_auroc_auprc_raw_results.csv",
            "pooled_auroc_auprc_results.csv",
            "per_tf_auroc_auprc_results.csv",
            "per_tf_auroc_auprc_summary.csv",
        ]
        
        assert all([(self.model_training_dir / f).exists() for f in eval_files]), \
            f"Not all evaluation result files exist in {self.model_training_dir}. Expected files: {eval_files}"
        
        self.raw_results_df = pd.read_csv(self.model_training_dir / "pooled_auroc_auprc_raw_results.csv")
        self.results_df = pd.read_csv(self.model_training_dir / "pooled_auroc_auprc_results.csv")
        self.per_tf_all_df = pd.read_csv(self.model_training_dir / "per_tf_auroc_auprc_results.csv")
        self.per_tf_summary_df = pd.read_csv(self.model_training_dir / "per_tf_auroc_auprc_summary.csv")
    
    def run_forward_pass(self, num_batches: int = 1):
        if self.model is None:
            self.load_trained_model("trained_model.pt")

        device = self.device
        self.model.eval()

        global_tg_names = self.test_loader.dataset.tg_names

        pred_blocks = []
        true_blocks = []
        
        dataset = self.test_loader.dataset

        with torch.no_grad():
            for b, (batch_indices, batch) in tqdm(
                enumerate(
                    zip(self.test_loader.batch_sampler, self.test_loader)
                    ), 
                total=min(num_batches, len(self.test_loader.batch_sampler)), 
                desc="Running forward pass",
                ncols=80,
                ):
                if b >= num_batches:
                    break
                
                # Gets the cell indices for the batch, which is used to align the metacell names
                if hasattr(dataset, "_locate"):
                    local_indices = [dataset._locate(i)[1] for i in batch_indices]
                    if dataset._cell_idx is not None:
                        col_indices = [int(dataset._cell_idx[i]) for i in local_indices]
                    else:
                        col_indices = [int(i) for i in local_indices]
                else:
                    # Single-chrom dataset
                    if getattr(dataset, "_cell_idx", None) is not None:
                        col_indices = [int(dataset._cell_idx[i]) for i in batch_indices]
                    else:
                        col_indices = [int(i) for i in batch_indices]

                metacell_names = [dataset.metacell_names[i] for i in col_indices]
                
                atac_wins, tf_tensor, tg_expr_true, bias, tf_ids, tg_ids, motif_mask = batch
                atac_wins    = atac_wins.to(device, non_blocking=True)
                tf_tensor    = tf_tensor.to(device, non_blocking=True)
                tg_expr_true = tg_expr_true.to(device, non_blocking=True)
                bias         = bias.to(device, non_blocking=True)
                tf_ids       = tf_ids.to(device, non_blocking=True)
                tg_ids       = tg_ids.to(device, non_blocking=True)
                motif_mask   = motif_mask.to(device, non_blocking=True)            

                out, _, _ = self.model(
                    atac_wins, tf_tensor,
                    tf_ids=tf_ids, tg_ids=tg_ids,
                    bias=bias, motif_mask=motif_mask,
                    return_shortcut_contrib=False,
                )

                pred = self.tg_scaler.inverse_transform(out, ids=tg_ids).detach().cpu().numpy()
                true = self.tg_scaler.inverse_transform(tg_expr_true, ids=tg_ids).detach().cpu().numpy()

                tg_ids_cpu = tg_ids.detach().cpu().numpy().astype(int)
                tg_names_batch = [global_tg_names[i] for i in tg_ids_cpu]

                pred_df = pd.DataFrame(pred.T, index=tg_names_batch, columns=metacell_names)
                true_df = pd.DataFrame(true.T, index=tg_names_batch, columns=metacell_names)

                pred_df = pred_df.reindex(index=global_tg_names)
                true_df = true_df.reindex(index=global_tg_names)

                pred_blocks.append(pred_df)
                true_blocks.append(true_df)

        pred_df = pd.concat(pred_blocks, axis=1, copy=False) if pred_blocks else pd.DataFrame()
        true_df = pd.concat(true_blocks, axis=1, copy=False) if true_blocks else pd.DataFrame()

        pred_df = pred_df.dropna(axis=0, how="all")
        true_df = true_df.loc[pred_df.index]
        
        def _compare_pred_true(pred_df, true_df):
            pred_df["mean_expr"] = pred_df.mean(axis=1).values
            pred_df["std_expr"] = pred_df.std(axis=1).values
            predicted_expr = pred_df[["mean_expr", "std_expr"]]

            true_df["mean_expr"] = true_df.mean(axis=1).values
            true_df["std_expr"] = true_df.std(axis=1).values
            true_expr = true_df[["mean_expr", "std_expr"]]

            merged = predicted_expr.merge(
                true_expr,
                left_index=True,
                right_index=True,
                suffixes=("_pred", "_true")
            )

            merged["diff"] = merged["mean_expr_pred"] - merged["mean_expr_true"]
            
            return merged
        
        pred_vs_true_expr_comparison_df = _compare_pred_true(pred_df, true_df)
        
        self.tg_prediction_df = pred_df
        self.tg_true_df = true_df

        return pred_df, true_df, pred_vs_true_expr_comparison_df

    def visualize_model_structure(self):
        if self.model is None:
            self.load_trained_model("trained_model.pt")

        return self.model.module
    
    def load_ground_truth(self, ground_truth_file: Tuple[str, Path]):
        if type(ground_truth_file) == str:
            ground_truth_file = Path(ground_truth_file)
            
        if ground_truth_file.suffix == ".csv":
            sep = ","
        elif ground_truth_file.suffix == ".tsv":
            sep="\t"
            
        ground_truth_df = pd.read_csv(ground_truth_file, sep=sep, on_bad_lines="skip", engine="python")
        
        if "chip" in ground_truth_file.name and "atlas" in ground_truth_file.name:
            ground_truth_df = ground_truth_df[["source_id", "target_id"]]

        if ground_truth_df.columns[0] != "Source" or ground_truth_df.columns[1] != "Target":
            ground_truth_df = ground_truth_df.rename(columns={ground_truth_df.columns[0]: "Source", ground_truth_df.columns[1]: "Target"})
        ground_truth_df["Source"] = ground_truth_df["Source"].astype(str).str.upper()
        ground_truth_df["Target"] = ground_truth_df["Target"].astype(str).str.upper()
        
        # Build TF, TG, and edge sets for quick lookup later
        gt = ground_truth_df[["Source", "Target"]].dropna()

        gt_tfs = set(gt["Source"].unique())
        gt_tgs = set(gt["Target"].unique())
        
        gt_pairs = (gt["Source"] + "\t" + gt["Target"]).drop_duplicates()
        
        gt_lookup = (gt_tfs, gt_tgs, set(gt_pairs))
            
        return ground_truth_df, gt_lookup
    
    def create_ground_truth_comparison_df(self, score_df, ground_truth_lookup, ground_truth_name):
        # Normalize once
        gt_tfs, gt_tgs, gt_pairs_set = ground_truth_lookup

        src = score_df["Source"]
        tgt = score_df["Target"]

        mask = src.isin(gt_tfs) & tgt.isin(gt_tgs)

        df = score_df.loc[mask].copy()
        # re-use normalized versions so we don't upper twice
        df["Source"] = src.loc[mask].values
        df["Target"] = tgt.loc[mask].values

        key = df["Source"] + "\t" + df["Target"]
        df["_in_gt"] = key.isin(gt_pairs_set).astype("int8")
        df["ground_truth_name"] = ground_truth_name

        return df
    
    def create_grn_ground_truth_overlap_comparison_df(
        self, 
        unlabeled_df: pd.DataFrame, 
        labeled_df: pd.DataFrame,
        ground_truth_df: pd.DataFrame, 
        ground_truth_name: str,
        ):
        """
        Creates a DataFrame that compares the TF, TG, and edge sets of GRN and Ground Truth.
        
        Parameters
        ----------
        unlabeled_df : pd.DataFrame
            DataFrame with at least ['Source', 'Target'] columns that contains the GRN edges
        labeled_df : pd.DataFrame
            DataFrame with at least ['Source', 'Target'] columns that contains the overlap between GRN and Ground Truth
        ground_truth_df : pd.DataFrame
            DataFrame with at least ['Source', 'Target'] columns that contains the Ground Truth edges
        ground_truth_name : str
            The name of the Ground Truth dataset
        
        Returns
        -------
        overlap_info_df : pd.DataFrame
            DataFrame containing the comparison information
        """
        grn_unique_tfs = unlabeled_df["Source"].nunique()
        grn_unique_tgs = unlabeled_df["Target"].nunique()
        grn_unique_edges = len(unlabeled_df)

        gt_unique_tfs = ground_truth_df["Source"].nunique()
        gt_unique_tgs = ground_truth_df["Target"].nunique()
        gt_unique_edges = len(ground_truth_df)

        overlap_tfs = labeled_df["Source"].nunique()
        overlap_tgs = labeled_df["Target"].nunique()
        overlap_edges = len(labeled_df)
        
        comparison_dict = {
            "TFs": [grn_unique_tfs, gt_unique_tfs, overlap_tfs],
            "TGs": [grn_unique_tgs, gt_unique_tgs, overlap_tgs],
            "edges": [grn_unique_edges, gt_unique_edges, overlap_edges],
        }
        
        def pct(num, den):
            return np.where(den == 0, np.nan, (num / den) * 100)

        overlap_info_df = pd.DataFrame.from_dict(comparison_dict, orient="index", columns=["GRN", f"Ground Truth {ground_truth_name}", "Overlap (Score DF in GT)"])
        overlap_info_df["Pct of GRN in GT"] = pct(overlap_info_df["Overlap (Score DF in GT)"], overlap_info_df["GRN"]).round(2)
        overlap_info_df["Pct of GT in GRN"] = pct(overlap_info_df["Overlap (Score DF in GT)"], overlap_info_df[f"Ground Truth {ground_truth_name}"]).round(2)
        return overlap_info_df
    
    def plot_auroc_auprc(
        self, 
        score_df: pd.DataFrame, 
        ground_truth: Tuple[pd.DataFrame, Tuple[Set[str], Set[str], Set[str]]],
        ground_truth_name: str, 
        return_overlap_info: bool = True,
        balance: bool = True,
        no_fig: bool = False,
        save_fig: bool = False,
        ):
        
        ground_truth_df, gt_lookup = ground_truth
        
        # Uses a fast lookup to label GRN edges as 1 or 0 depending on whether they are in the ground truth or not
        # (only compares TFs and TGs that are in both the GRN and the ground truth)
        labeled_df = self.create_ground_truth_comparison_df(score_df, gt_lookup, ground_truth_name)
            
        if len(labeled_df) == 0 or labeled_df["_in_gt"].nunique() < 2:
            logging.info(f"Need at least one positive and one negative, got {labeled_df['_in_gt'].value_counts().to_dict()}")
            return None
                
        def _balance_pos_neg(df, random_state=42):
            """Balances positive and negative edges by downsampling the majority class."""
            rng = np.random.default_rng(random_state)

            y = df["_in_gt"].to_numpy() == 1
            pos_idx = np.flatnonzero(y)
            neg_idx = np.flatnonzero(~y)

            n_pos = pos_idx.size
            n_neg = neg_idx.size
            if n_pos == 0 or n_neg == 0:
                return df  # no copy

            n = min(n_pos, n_neg)
            if n_pos > n:
                pos_idx = rng.choice(pos_idx, size=n, replace=False)
            if n_neg > n:
                neg_idx = rng.choice(neg_idx, size=n, replace=False)

            # iloc keeps column dtypes; copy only the subset
            return df.iloc[np.concatenate([pos_idx, neg_idx])].reset_index(drop=True)
        
        def _create_random_distribution(scores, seed: int = 42) -> np.ndarray:
            rng = np.random.default_rng(seed)
            arr = np.asarray(scores)   # works for Series or ndarray, no copy if already ndarray
            return rng.uniform(arr.min(), arr.max(), size=arr.shape[0])
        
        y = labeled_df["_in_gt"].fillna(0).astype(int).to_numpy()
        s = labeled_df["Score"].to_numpy()
        
        if balance:
            balanced = _balance_pos_neg(labeled_df, random_state=42)
            y = balanced["_in_gt"].astype(int).to_numpy()
            s = balanced["Score"].to_numpy()
        
        auroc = roc_auc_score(y, s)
        fpr, tpr, _ = roc_curve(y, s)
        rand_fpr, rand_tpr, _ = roc_curve(y, _create_random_distribution(s))
        
        auprc = average_precision_score(y, s)
        prec, rec, _ = precision_recall_curve(y, s)
        rand_prec, rand_rec, _ = precision_recall_curve(y, _create_random_distribution(s))
        
        metric_df = pd.DataFrame({
            "experiment": self.experiment_name,
            "ground_truth": ground_truth_name,
            "auroc": auroc,
            "auprc": auprc,
        }, index=[0])
        
        
        if self.auroc_auprc_scores is None:
            self.auroc_auprc_scores = metric_df
        else:
            self.auroc_auprc_scores = pd.concat([self.auroc_auprc_scores, metric_df], ignore_index=True)
                
        if not no_fig:
            # ROC plot
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 4))
            ax[0].plot(rand_fpr, rand_tpr, color="#747474", linestyle="--", lw=2)
            ax[0].plot(fpr, tpr, lw=2, color="#4195df", label=f"AUROC = {auroc:.3f}")
            ax[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
            ax[0].set_xlabel("False Positive Rate", fontsize=12)
            ax[0].set_ylabel("True Positive Rate", fontsize=12)
            ax[0].set_title(f"AUROC", fontsize=12)
            ax[0].legend(
                bbox_to_anchor=(0.5, -0.28),
                loc="upper center",
                borderaxespad=0.0
            )
            ax[0].set_xlim(0, 1)
            ax[0].set_ylim(0, 1)
            
            # Precision-Recall plot
            ax[1].plot(rand_rec, rand_prec, color="#747474", linestyle="--", lw=2)
            ax[1].plot(rec, prec, lw=2, color="#4195df", label=f"AUPRC = {auprc:.3f}")
            ax[1].set_xlabel("Recall", fontsize=12)
            ax[1].set_ylabel("Precision", fontsize=12)
            ax[1].set_title(f"AUPRC", fontsize=12)
            ax[1].legend(
                bbox_to_anchor=(0.5, -0.28),
                loc="upper center",
                borderaxespad=0.0
            )
            ax[1].set_ylim(0, 1.0)
            ax[1].set_xlim(0, 1.0)
            plt.suptitle(f"{self.experiment_name} vs {ground_truth_name}", fontsize=14)
            plt.tight_layout()
            
            if save_fig:
                fig_dir = os.path.join(self.experiment_dir, self.experiment_name, ground_truth_name)
                if not os.path.exists(fig_dir):
                    os.makedirs(fig_dir)
                auroc_fig_path = os.path.join(fig_dir, f"{ground_truth_name}_auroc_auprc.png")

                fig.savefig(auroc_fig_path, dpi=300)
        else:
            fig = None
            
        if return_overlap_info:
            overlap_info_df = self.create_grn_ground_truth_overlap_comparison_df(
                score_df, labeled_df, ground_truth_df, ground_truth_name
            )
        else:
            overlap_info_df = None
        
        return fig, overlap_info_df
    
    def plot_train_val_loss(self):
        fig = plt.figure(figsize=(6, 5))
        df = self.training_df.iloc[5:, :]
        
        plt.plot(df["Epoch"], df["Train MSE"], label="Train MSE", linewidth=2)
        plt.plot(df["Epoch"], df["Val MSE"], label="Val MSE", linewidth=2)
        plt.title(f"Train Val Loss Curves", fontsize=17)
        plt.xlabel("Epoch", fontsize=17)
        plt.ylabel("Loss", fontsize=17)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)

        plt.legend(fontsize=15)
        plt.tight_layout()
        
        return fig
    
    def plot_train_correlation(self):
        fig = plt.figure(figsize=(6, 5))
        
        df = self.training_df
        plt.plot(df.index, df["R2_u"], linewidth=2, label=f"Best R2 (unscaled) = {df['R2_u'].max():.2f}")
        plt.plot(df.index, df["R2_s"], linewidth=2, label=f"Best R2 (scaled)     = {df['R2_s'].max():.2f}")

        plt.title(f"TG Expression R2 Across Training", fontsize=17)
        plt.ylim((0,1))
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        plt.xlabel("Epoch", fontsize=17)
        plt.ylabel("R2", fontsize=17)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        return fig
    
    def plot_gpu_usage(self, smooth=None):
        """
        align_to_common_duration: if True, truncate each run to the shortest duration so curves end together.
        smooth: optional int window (in seconds) for a centered rolling mean on memory (e.g., smooth=5).
        """
        if self.gpu_usage_df is None:
            logging.warning(f"GPU usage data not available for {self.experiment_name}. Cannot plot GPU usage.")
            return
        
        fig, ax = plt.subplots(figsize=(7,4))

        if smooth and smooth > 1:
            self.gpu_usage_df["memory_used_gib"] = self.gpu_usage_df["memory_used_gib"].rolling(smooth, center=True, min_periods=1).mean()

        ax.plot(self.gpu_usage_df["elapsed_hr"], self.gpu_usage_df["memory_used_gib"], label=f"Avg GPU RAM Used", linewidth=2)

        ax.axhline(self.gpu_memory_limit_gib, linestyle="--", label=f"Max RAM")
        ax.set_ylabel("GiB")
        ax.set_xlabel("Minutes since start")
        ax.set_ylim(0, self.gpu_memory_limit_gib + 5)
        ax.xaxis.set_major_locator(MultipleLocator(1))  # tick every 1 hour
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}"))
        ax.set_xlabel("Hours since start")

        handles, legend_labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                legend_labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0.0,
            )
        ax.set_title(
            f"Average GPU Memory During Training"
        )
        plt.tight_layout()
        plt.show()
        return fig
    
    def plot_true_vs_predicted_tg_expression(self, num_batches: int=15, rerun_forward_pass: bool = False):
        fig, ax = plt.subplots(figsize=(5,5))

        if self.tg_prediction_df is None or self.tg_true_df is None or rerun_forward_pass:
            logging.info("Running forward pass to get predicted vs true TG expression for a subset of test batches...")
            self.tg_prediction_df, self.tg_true_df, _ = self.run_forward_pass(num_batches=num_batches)
        
        x = self.tg_prediction_df.mean(axis=1).values
        y = self.tg_true_df.mean(axis=1).values
        
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        
        correlation = np.corrcoef(x, y)[0, 1]
        ax.set_title(f"Model Prediction vs True TG Expression\nPearson Correlation = {correlation:.2f}")

        ax.scatter(x, y, s=10, alpha=0.6)

        lims = [
            min(x.min(), y.min()),
            max(x.max(), y.max()),
        ]

        ax.plot(lims, lims, color="grey", linestyle="--", linewidth=1)

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel("True mean expression")
        ax.set_ylabel("Predicted mean expression")

        return fig
    
    def plot_per_tf_auroc_boxplot(self, agg_by_gt: bool = True, ylim: tuple = (0.3, 0.7)):
        """
        Plots a boxplot of per-TF AUROC scores for each GRN inference method.

        Parameters
        ----------
        agg_by_gt : bool, optional
            * If **True**, plots the average per-TF AUROC score for all TFs for each ground truth
            * If **False**, plots the per-TF AUROC scores for each TF for each ground truth

        """
        if self.per_tf_all_df is None:
            
            assert (self.model_training_dir / "per_tf_auroc_auprc_results.csv").exists(), \
                f"Per-TF AUROC/AUPRC results file does not exist for {self.experiment_name}, model_training_00{self.model_num}"
            
            self.per_tf_all_df = pd.read_csv(self.model_training_dir / "per_tf_auroc_auprc_results.csv")
        
        if not agg_by_gt:
            # Average the per-TF AUROC scores across ground truths for each method
            per_tf_plot_df = (
                self.per_tf_all_df.dropna(subset=["auroc"])
                .groupby(['method', 'tf'], as_index=False)
                .agg(
                    auroc=('auroc', 'mean'),
                    n_gt=('gt', 'nunique'),
                )
            )
            
        elif agg_by_gt:
            # Average the per-TF AUROC scores across ground truths for each method
            per_tf_plot_df = (
                self.per_tf_all_df
                .dropna(subset=["auroc"])
                .groupby(["method", "gt"], as_index=False)
                .agg(
                    auroc=("auroc", "mean"),
                    n_gt=("gt", "nunique"),
                )
            )


        fig = self._plot_all_results_auroc_boxplot(
            per_tf_plot_df, 
            per_tf=True,
            ylim=ylim
            )
        return fig
        
    def plot_pooled_auroc_boxplot(self, ylim: tuple = (0.3, 0.7)):
        if self.results_df is None:
            assert (self.model_training_dir / "pooled_auroc_auprc_results.csv").exists(), \
                f"Pooled AUROC/AUPRC results file does not exist for {self.experiment_name}, model_training_00{self.model_num}"
            self.results_df = pd.read_csv(self.model_training_dir / "pooled_auroc_auprc_results.csv")
            
        fig = self._plot_all_results_auroc_boxplot(
            self.results_df, 
            per_tf=False,
            ylim=ylim,
            )
        
        return fig
    
    def _plot_all_results_auroc_boxplot(
        self,
        df: pd.DataFrame, 
        per_tf: bool = False, 
        override_color: bool = False,
        ylim: tuple = (0.3, 0.7),
        sort_by_mean: bool = True,
        ) -> plt.Figure:
        """
        Plots AUROC boxplots for all GRN inference methods in the provided DataFrame.
        
        Parameters
        -------------
        df : pd.DataFrame
            DataFrame containing AUROC results with columns 'method' and 'auroc'.
        per_tf : bool, optional
            If True, indicates that the DataFrame contains per-TF AUROC scores. Default is False.
        override_color : bool, optional
            If True, overrides the default coloring scheme for methods to plot all boxes as blue. Default is False.
        """
        
        
        # 1. Order methods by mean AUROC (highest → lowest)
        if sort_by_mean:
            method_order = (
                df.groupby("method")["auroc"]
                .mean()
                .sort_values(ascending=False)
                .index
                .tolist()
            )
        else:
            method_order = (
                df.groupby("method")["auroc"]
                .mean()
                .index
                .tolist()
            )

        if "No Filtering" in method_order:
            method_order = [m for m in method_order if m != "No Filtering"] + ["No Filtering"]
        
        mean_by_method = (
            df.groupby("method")["auroc"]
            .mean()
        )
        
        # 2. Prepare data in that order
        data = [df.loc[df["method"] == m, "auroc"].values for m in method_order]

        feature_list = [
            "Gradient Attribution",
            "TF Knockout",
        ]
        my_color = "#4195df"
        other_color = "#747474"

        fig, ax = plt.subplots(figsize=(9, 5))

        # Baseline random line
        ax.axhline(y=0.5, color="#2D2D2D", linestyle='--', linewidth=1)

        # --- Boxplot (existing styling) ---
        bp = ax.boxplot(
            data,
            tick_labels=method_order,
            patch_artist=True,
            showfliers=False
        )

        # Color boxes: light blue for your methods, grey for others
        for box, method in zip(bp["boxes"], method_order):
            if method in feature_list or override_color:
                box.set_facecolor(my_color)
            else:
                box.set_facecolor(other_color)

        # Medians in black
        for median in bp["medians"]:
            median.set_color("black")

        # --- NEW: overlay jittered points for each method ---
        for i, method in enumerate(method_order, start=1):
            y = df.loc[df["method"] == method, "auroc"].values
            if len(y) == 0:
                continue

            # Small horizontal jitter around the box center (position i)
            x = np.random.normal(loc=i, scale=0.06, size=len(y))

            # Match point color to box color
            point_color = my_color if method in feature_list or override_color else other_color

            ax.scatter(
                x, y,
                color=point_color,
                alpha=0.7,
                s=18,
                edgecolor="k",
                linewidth=0.3,
                zorder=3,
            )
            
            mean_val = y.mean()
            ax.scatter(
                i, mean_val,
                color="white",
                edgecolor="k",
                s=30,
                zorder=4,
            )
            
            # # Annotate the mean value above the mean point
            # ax.text(i, y.max() + 0.015, f"{mean_val:.3f}", ha="center", va="bottom", fontsize=12)

        legend_handles = [
            Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                markerfacecolor=(
                    my_color if (method in feature_list or override_color) else other_color
                ),
                markeredgecolor="k",
                markersize=7,
                label=f"{method}: {mean_by_method.loc[method]:.3f}"
            )
            for method in method_order
        ]
        
        ax.legend(
            handles=legend_handles,
            title="Mean AUROC",
            bbox_to_anchor=(1.05, 0.5),
            loc="center left",
            borderaxespad=0.0,
            ncol=1,
        )

        ax.set_ylabel("AUROC across ground truths")
        if per_tf == True:
            ax.set_title("per-TF AUROC Scores per method")
            ax.set_ylim(ylim)
        else:
            ax.set_title("AUROC Scores per method")
            ax.set_ylim(ylim)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        
        return fig
    
    def _locate_last_checkpoint(self):
        """
        Locate the checkpoint_<N>.pt file with the largest N in self.model_training_dir.
        
        Returns:
            str: The name of the checkpoint file (not full path)
        """
        checkpoint_files = sorted(self.model_training_dir.glob("checkpoint_*.pt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {self.model_training_dir}")
        last_checkpoint = checkpoint_files[-1]
        return last_checkpoint.name
    
    def _load_json(self, path: Path) -> dict:
        with open(path, "r") as f:
            data = json.load(f)
        return data
    
    def _load_run_params(self) -> dict:
        """
        Load the run parameters saved during model training.

        Parameters:
        None

        Returns:
        dict: The run parameters saved during model training
        """
        run_params_path = self.model_training_dir / "run_params.json"
        if not run_params_path.exists():
            raise FileNotFoundError(f"Run parameters file {run_params_path} does not exist.")
        
        with open(run_params_path, "r") as f:
            run_params = json.load(f)
        
        return run_params
    
    def _format_gpu_usage_file(self):
        """
        Format the GPU usage log file into a more usable format.

        The function assumes that the input file contains columns for "timestamp", "memory.used [MiB]", and "memory.total [MiB]" and
        that the timestamp is in seconds since the epoch.

        memory usage per second, and a float containing the total memory available on the GPU in GiB.

        """
        try:
            gpu_usage_file = Path(f"{self.experiment_dir}/{self.experiment_name}") / "logs" / "gpu_usage.csv"

            gpu_usage_df = pd.read_csv(gpu_usage_file)
            gpu_usage_df.columns = gpu_usage_df.columns.str.strip()
            gpu_usage_df["timestamp"] = pd.to_datetime(gpu_usage_df["timestamp"], errors="coerce")
            gpu_usage_df["tsec"] = gpu_usage_df["timestamp"].dt.floor("s")

            gpu_usage_df["memory_used_gib"]  = gpu_usage_df["memory.used [MiB]"].astype(str).str.extract(r"(\d+)").astype(float) / 1024
            gpu_usage_df["memory_total_gib"] = gpu_usage_df["memory.total [MiB]"].astype(str).str.extract(r"(\d+)").astype(float) / 1024

            t0 = gpu_usage_df["tsec"].min()
            gpu_usage_df["elapsed_s"] = (gpu_usage_df["tsec"] - t0).dt.total_seconds().astype(int)
            gpu_usage_df["elapsed_min"] = gpu_usage_df["elapsed_s"] / 60.0
            gpu_usage_df["elapsed_hr"] = gpu_usage_df["elapsed_s"] / 3600.0
            

            # mean per second, then carry minutes as a column
            mean_per_sec_df = (
                gpu_usage_df.groupby("elapsed_s", as_index=False)["memory_used_gib"]
                .mean()
                .sort_values("elapsed_s")
            )
            mean_per_sec_df["elapsed_min"] = mean_per_sec_df["elapsed_s"] / 60.0
            mean_per_sec_df["elapsed_hr"] = mean_per_sec_df["elapsed_s"] / 3600.0

            gpu_memory_limit_gib = float(gpu_usage_df["memory_total_gib"].iloc[0])
            return gpu_usage_df, mean_per_sec_df, gpu_memory_limit_gib
        
        except FileNotFoundError:
            logging.warning(f"WARNING: GPU usage file not found for {self.experiment_name}. GPU usage plotting will be unavailable.")
            return None, None, None