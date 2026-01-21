import os, sys, json
import numpy as np
import torch
from pathlib import Path
import logging
from tqdm import tqdm
import argparse

import torch.distributed as dist
from torch.amp import autocast

import sys
PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
SRC_DIR = str(Path(PROJECT_DIR) / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

logging.basicConfig(level=logging.INFO, format='%(message)s')

from multiomic_transformer.models.model import MultiomicTransformer
from multiomic_transformer.datasets.dataset_refactor import SimpleScaler

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        distributed = True
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        distributed = False

    if distributed:
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=device,
        )
    return rank, world_size, local_rank, distributed

def load_model(selected_experiment_dir, checkpoint_file, device):
    params_path = selected_experiment_dir / "run_parameters.json"
    with open(params_path, "r") as f:
        params = json.load(f)

    # Pull out architecture hyperparameters
    d_model   = params.get("d_model")
    num_heads = params.get("num_heads")
    num_layers = params.get("num_layers")
    d_ff      = params.get("d_ff")
    dropout   = params.get("dropout", 0.0)
    use_shortcut   = params.get("use_shortcut", False)
    use_dist_bias  = params.get("use_dist_bias", False)
    use_motif_mask = params.get("use_motif_mask", False)

    
    # 1) Load test loader and checkpoint
    test_loader = torch.load(selected_experiment_dir / "test_loader.pt", weights_only=False)

    ckpt_path = os.path.join(selected_experiment_dir, checkpoint_file)
    state = torch.load(ckpt_path, map_location="cpu")
    
    # 2) Recreate model EXACTLY as in training
    model = MultiomicTransformer(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        tf_vocab_size=len(state["tf_scaler_mean"]),
        tg_vocab_size=len(state["tg_scaler_mean"]),
        use_bias=use_dist_bias,
        use_shortcut=use_shortcut,
        use_motif_mask=use_motif_mask,
    )

    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.to(device).eval()

    # 3) Rebuild scalers on the SAME DEVICE as inputs
    tg_scaler = SimpleScaler(
        mean=torch.as_tensor(state["tg_scaler_mean"], device=device, dtype=torch.float32),
        std=torch.as_tensor(state["tg_scaler_std"],  device=device, dtype=torch.float32),
    )
    tf_scaler = SimpleScaler(
        mean=torch.as_tensor(state["tf_scaler_mean"], device=device, dtype=torch.float32),
        std=torch.as_tensor(state["tf_scaler_std"],  device=device, dtype=torch.float32),
    )

    return model, test_loader, tg_scaler, tf_scaler, state

def run_tf_knockout(
    selected_experiment_dir: Path, 
    model, 
    test_loader, 
    tg_scaler,
    tf_scaler,
    state,
    device,
    use_amp, 
    rank, 
    world_size, 
    distributed,
    max_batches=None,
    use_dataloader=False,
    ):

    T_total = len(state["tf_scaler_mean"])   # total TF vocab size
    G_total = len(state["tg_scaler_mean"])   # total TG vocab size

    # Accumulators for TF knockout effect (CPU)
    effect_sum   = torch.zeros(T_total, G_total, device=device, dtype=torch.float64)
    effect_count = torch.zeros_like(effect_sum, device=device)

    model.to(device).eval()

    iterator = test_loader
    if rank == 0:
        iterator = tqdm(test_loader, desc="TF knockout", unit="batches", total=max_batches)

    with torch.no_grad():
        for b_idx, batch in enumerate(iterator):
            if (max_batches is not None) and (b_idx >= max_batches):
                break

            # Only process batches assigned to this rank
            if not use_dataloader and (b_idx % world_size != rank):
                continue
            
            atac_wins, tf_tensor, targets, bias, tf_ids, tg_ids, motif_mask = batch
            atac_wins  = atac_wins.to(device, non_blocking=True)
            tf_tensor  = tf_tensor.to(device, non_blocking=True)   # unscaled
            bias       = bias.to(device, non_blocking=True)
            tf_ids     = tf_ids.to(device)
            tg_ids     = tg_ids.to(device)
            motif_mask = motif_mask.to(device)

            # Shape of TF input
            if tf_tensor.dim() == 2:
                B, T_eval = tf_tensor.shape
                F_dim = 1
            else:
                B, T_eval, F_dim = tf_tensor.shape

            # --------- 1) Scale TFs ONCE per batch ---------
            if tf_scaler is not None:
                tf_scaled_base = tf_scaler.transform(tf_tensor, tf_ids)  # [B, T_eval] or [B, T_eval, F_dim]
            else:
                tf_scaled_base = tf_tensor

            # Work internally as 3D: [B, T_eval, F_dim]
            if tf_tensor.dim() == 2:
                tf_scaled_base_3d = tf_scaled_base.unsqueeze(-1)  # [B, T_eval, 1]
            else:
                tf_scaled_base_3d = tf_scaled_base               # [B, T_eval, F_dim]

            # --------- 2) Scaled value for "expression = 0" per TF (per position) ---------
            if tf_scaler is not None:
                # Build a single zero-expression tensor and transform once
                zeros_expr_1 = torch.zeros_like(tf_tensor[:1])   # [1, T_eval] or [1, T_eval, F_dim]
                zeros_scaled_1 = tf_scaler.transform(zeros_expr_1, tf_ids)  # [1, T_eval] or [1, T_eval, F_dim]

                if tf_tensor.dim() == 2:
                    # [1, T_eval] -> [T_eval, 1]
                    zeros_scaled = zeros_scaled_1.squeeze(0).unsqueeze(-1)  # [T_eval, 1]
                else:
                    # [1, T_eval, F_dim] -> [T_eval, F_dim]
                    zeros_scaled = zeros_scaled_1.squeeze(0)                # [T_eval, F_dim]
            else:
                zeros_scaled = torch.zeros(
                    (T_eval, F_dim), device=device, dtype=tf_tensor.dtype
                )  # KO really is 0 in model input space

            # --------- 3) Baseline predictions (once per batch) ---------
            with autocast(device_type=device.type, enabled=use_amp):
                # Use original scaled base (2D/3D) for the model
                tf_scaled_for_model = tf_scaled_base if tf_tensor.dim() == 2 else tf_scaled_base_3d

                preds_base_s, _, _ = model(
                    atac_wins, tf_scaled_for_model,
                    tf_ids=tf_ids, tg_ids=tg_ids,
                    bias=bias, motif_mask=motif_mask,
                    return_shortcut_contrib=False,
                )

                if tg_scaler is not None:
                    preds_base_u = tg_scaler.inverse_transform(preds_base_s, tg_ids)
                else:
                    preds_base_u = preds_base_s

            preds_base_u = torch.nan_to_num(
                preds_base_u.float(), nan=0.0, posinf=1e6, neginf=-1e6
            )  # [B, G_eval]
            B, G_eval = preds_base_u.shape

            # --------- 4) Prepare a working scaled TF tensor (cloned ONCE) ---------
            # We will modify one TF position at a time, run the model, then restore.
            tf_scaled_work = tf_scaled_base_3d.clone()  # [B, T_eval, F_dim]
            
            zero_eps = 1e-8  # tweak if needed

            for t_pos in range(T_eval):
                # ----------------------------------
                # 5a) Skip positions that are ~zero
                # ----------------------------------
                if tf_tensor.dim() == 2:
                    unscaled_slice = tf_tensor[:, t_pos]          # [B]
                else:
                    # if F_dim>1, first feature is usually expression;
                    # adjust if your layout is different
                    unscaled_slice = tf_tensor[:, t_pos, 0]       # [B]

                if torch.all(unscaled_slice.abs() < zero_eps):
                    # KO is identical to baseline -> no effect, no need to run the model
                    continue

                # ----------------------------------
                # 5b) Apply KO in *scaled* space
                # ----------------------------------
                # zeros_scaled[t_pos]: [F_dim]
                ko_val = zeros_scaled[t_pos].unsqueeze(0).expand(B, F_dim)  # [B, F_dim]
                tf_scaled_work[:, t_pos, :] = ko_val

                # Match original dimensionality for the model input
                if tf_tensor.dim() == 2:
                    tf_scaled_input = tf_scaled_work.squeeze(-1)   # [B, T_eval]
                else:
                    tf_scaled_input = tf_scaled_work              # [B, T_eval, F_dim]

                # Run knockout forward pass
                with autocast(device_type=device.type, enabled=use_amp):
                    preds_ko_s, _, _ = model(
                        atac_wins, tf_scaled_input,
                        tf_ids=tf_ids, tg_ids=tg_ids,
                        bias=bias, motif_mask=motif_mask,
                        return_shortcut_contrib=False,
                    )

                    if tg_scaler is not None:
                        preds_ko_u = tg_scaler.inverse_transform(preds_ko_s, tg_ids)
                    else:
                        preds_ko_u = preds_ko_s

                preds_ko_u = torch.nan_to_num(
                    preds_ko_u.float(), nan=0.0, posinf=1e6, neginf=-1e6
                )  # [B, G_eval]

                # delta = baseline - knockout (positive: TF supports expression)
                delta = preds_base_u - preds_ko_u          # [B, G_eval]
                delta_mean = delta.mean(dim=0)             # [G_eval]
                                
                tf_global = int(tf_ids[t_pos].item())
                effect_sum[tf_global, tg_ids] += delta_mean
                effect_count[tf_global, tg_ids] += 1

                # ----------------------------------
                # 5c) Restore baseline slice *without* cloning
                # ----------------------------------
                tf_scaled_work[:, t_pos, :] = tf_scaled_base_3d[:, t_pos, :]


    if distributed:
        dist.all_reduce(effect_sum,   op=dist.ReduceOp.SUM)
        dist.all_reduce(effect_count, op=dist.ReduceOp.SUM)

    if rank == 0:
        tf_tg_effect_np = (effect_sum / (effect_count + 1e-12)).detach().cpu().numpy()
        np.save(selected_experiment_dir / "tf_tg_fullmodel_knockout.npy", tf_tg_effect_np)
        np.save(selected_experiment_dir / "tf_tg_fullmodel_knockout_count.npy",
                effect_count.detach().cpu().numpy())
        logging.info("Finished TF Knockout calculation!")
        
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Compute gradient-based TF->TG attributions")
    argparser.add_argument("--selected_experiment_dir", type=str, required=True,
                        help="Name of the experiment directory (under experiments/mESC_no_scale_linear/)")
    argparser.add_argument("--use_amp", action="store_true",
                        help="Enable mixed-precision inference (defaults to enabled on CUDA)")
    argparser.add_argument("--model_file", default="trained_model.pt", type=str,
                        help="File for model checkpoint (default: trained_model.pt)")
    argparser.add_argument("--max_batches", default=None, type=int,
                        help="Maximum number of batches to process (for debugging, defualts to all)")
    args = argparser.parse_args()

    selected_experiment_dir = Path(args.selected_experiment_dir)

    rank, world_size, local_rank, distributed = setup_distributed()
    
    # Use consistent device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Check GPU capability for mixed precision
    use_amp = args.use_amp and device.type == 'cuda'
    if use_amp:
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(device)
            if capability[0] < 7:
                logging.warning(f"GPU compute capability {capability[0]}.{capability[1]} < 7.0, disabling AMP")
                use_amp = False
        else:
            use_amp = False
    
    model, test_loader, tg_scaler, tf_scaler, state = load_model(
        selected_experiment_dir=selected_experiment_dir,
        checkpoint_file=args.model_file,
        device=device
    )
    
    run_tf_knockout(
        selected_experiment_dir=selected_experiment_dir,
        model=model,
        test_loader=test_loader,
        tg_scaler=tg_scaler,
        tf_scaler=tf_scaler,
        state=state,
        device=device,
        use_amp = use_amp,
        rank=rank, 
        world_size=world_size, 
        distributed=distributed,
        max_batches=args.max_batches,
        use_dataloader=False,
        )

    if distributed:
        dist.barrier()
        dist.destroy_process_group()