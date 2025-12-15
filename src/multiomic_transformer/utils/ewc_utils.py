# ewc_utils.py
import torch, copy, os
import torch.nn.functional as F

@torch.no_grad()
def clone_params(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

def zero_like_params(model):
    return {n: torch.zeros_like(p, device=p.device) for n, p in model.named_parameters() if p.requires_grad}

def param_name_strip_module(named_params):
    """
    Returns dict mapping WITHOUT 'module.' prefix if present, to make
    checkpoints usable both with and without DDP.
    """
    out = {}
    for n, p in named_params:
        key = n[7:] if n.startswith("module.") else n
        out[key] = p
    return out

def state_dict_strip_module(d):
    return { (k[7:] if k.startswith("module.") else k): v for k, v in d.items() }

def compute_fisher_diag(model, loader, device, n_batches=50, loss_fn="mse"):
    """
    Diagonal Fisher via squared gradients. Run on SOURCE data (mESC).
    Use model.eval() (no dropout). Still backprop to get grads.
    """
    model.eval()
    fisher = zero_like_params(model)

    batches = 0
    for batch in loader:
        batches += 1
        atac_wins, tf_tensor, tg_true, bias, tf_ids, tg_ids, motif_mask = batch
        atac_wins = atac_wins.to(device)
        tf_tensor = tf_tensor.to(device)
        tg_true   = tg_true.to(device)
        bias      = bias.to(device)
        tf_ids    = tf_ids.to(device)
        tg_ids    = tg_ids.to(device)
        motif_mask = motif_mask.to(device)

        # match your inference-time normalization
        mu = tf_tensor.mean(dim=1, keepdim=True)
        sd = tf_tensor.std(dim=1, keepdim=True).clamp_min(1e-6)
        tf_norm = (tf_tensor - mu) / sd

        model.zero_grad(set_to_none=True)

        preds, _, _, _ = model(atac_wins, tf_norm, tf_ids=tf_ids, tg_ids=tg_ids, bias=bias)
        
        if loss_fn == "mse":
            loss = F.mse_loss(preds, tg_true)
        else:
            # If you switch to Gaussian NLL: loss = nll_gaussian(...)
            loss = F.mse_loss(preds, tg_true)

        loss.backward()

        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += (p.grad.detach() ** 2)

        if batches >= n_batches:
            break

    for n in fisher:
        fisher[n] = (fisher[n] / max(1, batches)).clamp_min_(1e-12) # Floored to avoid zeroes params
    return fisher

def save_ewc_bundle(path, model, fisher_diag):
    """
    Save ref params θ* and Fisher diag. Strips 'module.' if present.
    """
    ref = {n: p.detach().clone() for n, p in param_name_strip_module(model.named_parameters()).items()}
    fish = state_dict_strip_module(fisher_diag)
    torch.save({"ref_params": ref, "fisher_diag": fish}, path)

def load_ewc_bundle(path, device):
    chk = torch.load(path, map_location=device)
    # Everything is already CPU/GPU tensors thanks to map_location
    return chk["ref_params"], chk["fisher_diag"]

def ewc_penalty(model, fisher_diag, ref_params, lambda_ewc=100.0, include=None, exclude=None):
    """
    include/exclude: sets of substrings; keep param if any include matches
    and no exclude matches. If include is None -> include all (except excluded).

    This version is robust to changes in parameter shape between when the
    EWC bundle was computed and the current model:
    - If F_n or theta_star exist but do not match p.numel(), that parameter
      is skipped instead of raising a RuntimeError.
    """
    # to be robust to 'module.' prefixes at runtime:
    name_map = {}
    for n, p in model.named_parameters():
        key = n[7:] if n.startswith("module.") else n
        name_map[n] = key

    device = next(model.parameters()).device
    loss_ewc = torch.zeros((), device=device)

    if lambda_ewc == 0.0 or fisher_diag is None or ref_params is None:
        return loss_ewc

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = name_map[n]

        if include and not any(tag in key for tag in include):
            continue
        if exclude and any(tag in key for tag in exclude):
            continue

        F_n = fisher_diag.get(key)
        theta_star = ref_params.get(key)
        if F_n is None or theta_star is None:
            continue

        # safety: ensure tensors are on same device + dtype
        F_n = F_n.to(device=device, dtype=p.dtype)
        theta_star = theta_star.to(device=device, dtype=p.dtype)

        # If the stored Fisher / theta_star shapes don't match this param,
        # skip this parameter rather than crashing.
        if F_n.numel() != p.numel() or theta_star.numel() != p.numel():
            # Optional: you could log debug info here if desired:
            # logging.debug(f"[EWC] Skipping {key}: bundle shapes {F_n.shape}, {theta_star.shape}, param {tuple(p.shape)}")
            continue

        # Reshape to match the current parameter exactly
        F_n = F_n.view_as(p)
        theta_star = theta_star.view_as(p)

        loss_ewc = loss_ewc + 0.5 * lambda_ewc * (F_n * (p - theta_star)**2).sum()

    return loss_ewc    
    
def merge_fishers(old_fisher, new_fisher, old_size, new_size):
    """
    Merge two Fisher diagonals weighted by dataset sizes.
    Handles shape mismatches by skipping incompatible parameters.

    Args:
        old_fisher (dict): parameter_name -> tensor of Fisher estimates (from previous data)
        new_fisher (dict): parameter_name -> tensor of Fisher estimates (from new data)
        old_size (int): number of samples used to compute old_fisher
        new_size (int): number of samples used to compute new_fisher

    Returns:
        dict: merged Fisher diagonal
    """
    merged = {}
    total = old_size + new_size
    skipped = []
    
    for n in old_fisher:
        if n not in new_fisher:
            # Parameter exists in old but not new - keep old value
            merged[n] = old_fisher[n]
            continue
            
        old_shape = old_fisher[n].shape
        new_shape = new_fisher[n].shape
        
        if old_shape != new_shape:
            # Shape mismatch (e.g., vocab size changed) - use new Fisher only
            # This happens when fine-tuning uses different vocab than pretraining
            merged[n] = new_fisher[n]
            skipped.append(f"{n}: old={old_shape} vs new={new_shape}")
        else:
            # Shapes match - do weighted merge
            merged[n] = (old_fisher[n] * old_size + new_fisher[n] * new_size) / total
    
    # Add any new parameters that weren't in old_fisher
    for n in new_fisher:
        if n not in merged:
            merged[n] = new_fisher[n]
    
    if skipped:
        import logging
        logging.warning(f"Fisher merge: {len(skipped)} params had shape mismatch, using new Fisher only: {skipped[:3]}...")
    
    return merged
