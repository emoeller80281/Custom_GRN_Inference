import json
import os
from numpy import gradient
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import sys
import logging

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
        
        if "chr19" in [p.name for p in (Path(experiment_dir) / experiment_name).iterdir()]:
            self.model_training_dir = Path(f"{experiment_dir}/{experiment_name}/chr19/model_training_00{model_num}")
        else:
            self.model_training_dir = Path(f"{experiment_dir}/{experiment_name}/model_training_00{model_num}")
                
        assert self.model_training_dir.exists(), f"Model training directory {self.model_training_dir} does not exist."
        
        # Load the run parameters saved during model training
        self.model_training_params = self._load_json(self.model_training_dir / "run_parameters.json")
        
        # Save the full list of experiment settings
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
    
    def load_gradient_attribution(self):
        gradient_attribution_file = self.model_training_dir / "tf_tg_grad_attribution.npy"
        
        assert gradient_attribution_file.exists(), f"Gradient attribution file {gradient_attribution_file} does not exist."
        
        grad = np.load(gradient_attribution_file).astype(np.float32)
        assert grad.shape == (len(self.tf_names), len(self.tg_names))

        grad = np.nan_to_num(grad, nan=0.0)
        grad_abs = np.abs(grad)

        score_pooled = np.log1p(grad_abs)

        # Calculate per-TF robust z-score
        median_val = np.median(grad_abs, axis=1, keepdims=True)
        mad = np.median(np.abs(grad_abs - median_val), axis=1, keepdims=True) + 1e-6
        score_per_tf = (grad_abs - median_val) / mad

        T, G = grad_abs.shape
        tf_idx, tg_idx = np.meshgrid(np.arange(T), np.arange(G), indexing="ij")
                
        eps = 1e-12
        mask = (score_pooled > eps) | (np.abs(score_per_tf) > eps)

        tf_idx, tg_idx = np.where(mask)

        df = pd.DataFrame({
            "Source": np.asarray(self.tf_names, dtype=object)[tf_idx],
            "Target": np.asarray(self.tg_names, dtype=object)[tg_idx],
            "Score_pooled": score_pooled[tf_idx, tg_idx],
            "Score_per_tf": score_per_tf[tf_idx, tg_idx],
        })
        
        df["Source"] = df["Source"].astype(str).str.upper()
        df["Target"] = df["Target"].astype(str).str.upper()
        
        return df
    
    def load_tf_knockout(self, positive_only: bool = True, eps: float = 1e-6):
        """
        Loads TF-knockout effects and returns a long-form DF with two scores:

        - Score_pooled: log1p(effect_used)  (global magnitude-compressed score)
        - Score_per_tf: robust per-TF score = (effect_used - median_tf) / MAD_tf

        Where effect_used is either:
        - clip(effect, 0, inf) if positive_only=True
        - effect (signed) if positive_only=False

        Notes:
        - Unobserved entries (counts==0) are set to NaN and dropped in the output.
        - If positive_only=True, effects at 0 are valid and retained.
        """
        tf_knockout_file = self.model_training_dir / "tf_tg_fullmodel_knockout.npy"
        assert tf_knockout_file.exists(), f"TF-knockout file {tf_knockout_file} does not exist."
        
        
        effect = np.load(tf_knockout_file).astype(np.float32)         # [T, G]
        counts = np.load(self.model_training_dir / "tf_tg_fullmodel_knockout_count.npy").astype(np.int32)    # [T, G]
        assert effect.shape == (len(self.tf_names), len(self.tg_names))
        assert counts.shape == effect.shape

        # Mark unobserved as NaN
        mask_observed = counts > 0
        effect = effect.copy()
        effect[~mask_observed] = np.nan

        # Choose effect representation
        if positive_only:
            effect_used = np.clip(effect, 0, None)  # keep NaNs
        else:
            effect_used = effect  # signed, keep NaNs

        # --- pooled score ---
        # If signed, use abs for pooled magnitude (keeps "strength" notion comparable to gradient pooled)
        pooled_base = effect_used if positive_only else np.abs(effect_used)
        score_pooled = np.log1p(pooled_base)

        # --- per-TF robust score ---
        med = np.nanmedian(effect_used, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(effect_used - med), axis=1, keepdims=True) + eps
        score_per_tf = (effect_used - med) / mad

        # --- build long-form DF ---
        T, G = effect_used.shape
        tf_idx, tg_idx = np.meshgrid(np.arange(T), np.arange(G), indexing="ij")

        df = pd.DataFrame({
            "Source": np.asarray(self.tf_names, dtype=object)[tf_idx.ravel()],
            "Target": np.asarray(self.tg_names, dtype=object)[tg_idx.ravel()],
            "Score_pooled": score_pooled.ravel(),
            "Score_per_tf": score_per_tf.ravel(),
            "counts": counts.ravel(),
        })

        # Drop unobserved (Score_pooled will be NaN there)
        df = df.dropna(subset=["Score_pooled"]).reset_index(drop=True)

        df["Source"] = df["Source"].astype(str).str.upper()
        df["Target"] = df["Target"].astype(str).str.upper()
        
        return df
    
    def run_forward_pass(self, num_batches: int = 1):
        if self.model is None:
            self.load_trained_model("trained_model.pt")

        device = self.device
        self.model.eval()

        global_tg_names = self.test_loader.dataset.tg_names

        pred_blocks = []
        true_blocks = []

        with torch.no_grad():
            for b, batch in enumerate(self.test_loader):
                if b >= num_batches:
                    break

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

                cols = [f"batch{b}_cell{i}" for i in range(pred.shape[0])]

                pred_blocks.append(pd.DataFrame(pred.T, index=tg_names_batch, columns=cols))
                true_blocks.append(pd.DataFrame(true.T, index=tg_names_batch, columns=cols))

        pred_df = pd.concat(pred_blocks, axis=1, copy=False) if pred_blocks else pd.DataFrame()
        true_df = pd.concat(true_blocks, axis=1, copy=False) if true_blocks else pd.DataFrame()

        pred_df = pred_df.dropna(axis=0, how="all")
        true_df = true_df.loc[pred_df.index]

        return pred_df, true_df

    def visualize_model_structure(self):
        if self.model is None:
            self.load_trained_model("trained_model.pt")

        return self.model.module
    
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
    