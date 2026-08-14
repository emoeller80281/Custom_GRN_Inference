#!/bin/bash -l
#SBATCH --job-name=probe_tftg
#SBATCH --output=LOGS/tf_tg_model/%x_%j.log
#SBATCH --error=LOGS/tf_tg_model/%x_%j.err
#SBATCH --time=1:00:00
#SBATCH -p dense
#SBATCH -N 1
#SBATCH --gres=gpu:v100:1
#SBATCH -c 16
#SBATCH --mem=192G

# Single-GPU throughput sweep over (batch_size, cells_per_edge), used to pick the
# training settings instead of guessing. Measures forward+backward on real cached edge
# bags, with and without the resident TF embedding table.

set -eo pipefail
PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER"
cd $PROJECT_DIR
mkdir -p LOGS/tf_tg_model
source activate my_env

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export OMP_NUM_THREADS=1

python3 - << 'PYEOF'
import importlib.util, sys, time, warnings
import torch
warnings.filterwarnings("ignore")

ROOT = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER"
sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/scripts")
import config
spec = importlib.util.spec_from_file_location("t", f"{ROOT}/scripts/train_tf_to_tg_model.py")
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
from torch.utils.data import DataLoader

print(f"GPU: {torch.cuda.get_device_name(0)}  "
      f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)

emb   = torch.load(config.tf_embedding_cache_path, weights_only=True)
msk   = torch.load(config.tf_mask_cache_path, weights_only=True)
peaks = torch.load(config.tf_tg_atac_peak_cache_path, weights_only=True)
bags  = torch.load(config.tf_tg_val_cache_path, weights_only=False)
print(f"loaded cache; val edges={len(bags['label']):,}", flush=True)

model = t.create_new_tf_tg_regulation_model(
    tf_bind_model_path=config.tf_dna_model_checkpoints[config.cell_type],
    tf_embeddings_tensor=emb, tf_mask_tensor=msk,
    # Must match training: 03b passes --keep_tf_dna_in_eval, which pins the frozen
    # TF-DNA submodule to eval (running BatchNorm stats) and enables its fast path.
    # Probing without it measures a code path training never takes.
    keep_tf_peak_model_in_eval=True,
).cuda().train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
scaler = torch.amp.GradScaler("cuda")
lossfn = torch.nn.BCEWithLogitsLoss()

def make_loader(bs, on_device, workers=6):
    ds = t.TFTGEdgeBagDataset(
        bags, tf_embeddings_tensor=emb, tf_mask_tensor=msk,
        atac_peak_tensor=peaks, return_tf_indices=on_device,
    )
    return DataLoader(ds, batch_size=bs, shuffle=True, num_workers=workers,
                      pin_memory=True, persistent_workers=True, prefetch_factor=4,
                      collate_fn=t.collate_tftg_edge_bags, drop_last=True)

def set_C(batch, C):
    """Reshape the cell axis to C so timing reflects the target cells/edge."""
    cur = batch["tf_expression"].shape[1]
    if C == cur:
        return batch
    idx = torch.arange(C) % cur
    batch["tf_expression"]      = batch["tf_expression"][:, idx]
    batch["tg_expression"]      = batch["tg_expression"][:, idx]
    batch["peak_accessibility"] = batch["peak_accessibility"][:, idx, :]
    batch["cell_mask"]          = torch.ones(batch["tf_expression"].shape[:2], dtype=torch.bool)
    return batch

def bench(bs, C, on_device, n_warm=6, n_time=18):
    if on_device:
        model.set_tf_embedding_table(emb, msk)
    else:
        model.tf_embedding_table = None
    loader = make_loader(bs, on_device)
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    it, seen, t0 = iter(loader), 0, None
    for i in range(n_warm + n_time):
        try: b = next(it)
        except StopIteration:
            it = iter(loader); b = next(it)
        b = set_C(b, C)
        b = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
        if i == n_warm:
            torch.cuda.synchronize(); t0 = time.time(); seen = 0
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits, _ = model(
                tf_embedding=b.get("tf_embedding"), tf_idx=b.get("tf_idx"),
                tf_mask=b.get("tf_mask"), peak_sequences=b["peak_sequences"],
                peak_accessibility=b["peak_accessibility"], peak_distance=b["peak_distance"],
                peak_mask=b["peak_mask"], tf_expression=b["tf_expression"],
                tg_expression=b["tg_expression"], cell_mask=b["cell_mask"],
            )
            loss = lossfn(logits.float(), b["label"].float())
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if i >= n_warm: seen += bs
    torch.cuda.synchronize()
    dt = time.time() - t0
    del loader
    return seen/dt, torch.cuda.max_memory_allocated()/1e9

print(f"\n{'mode':<14}{'batch':>6}{'cells':>7}{'edges/s':>10}{'rel':>7}{'peakGB':>9}", flush=True)
base = None
for on_dev, bs, C in [(False, 8, 24), (True, 8, 24),
                      (False, 64, 24), (True, 64, 24),
                      (True, 128, 24), (True, 256, 24),
                      (True, 128, 64), (True, 128, 128),
                      (True, 256, 64)]:
    try:
        r, m = bench(bs, C, on_dev)
        if base is None: base = r
        tag = "device-table" if on_dev else "per-edge emb"
        print(f"{tag:<14}{bs:>6}{C:>7}{r:>10.1f}{r/base:>6.1f}x{m:>9.1f}", flush=True)
    except RuntimeError as e:
        print(f"{'device-table' if on_dev else 'per-edge emb':<14}{bs:>6}{C:>7}   FAILED: {str(e)[:60]}", flush=True)
        torch.cuda.empty_cache()
PYEOF
