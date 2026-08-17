#!/bin/bash -l
#SBATCH --job-name=startup_prof
#SBATCH --output=LOGS/tf_tg_model/%x_%j.log
#SBATCH --error=LOGS/tf_tg_model/%x_%j.err
#SBATCH --time=1:00:00
#SBATCH -p dense
#SBATCH -N 1
#SBATCH --gres=gpu:v100:1
#SBATCH -c 16
#SBATCH --mem=192G
set -eo pipefail
cd /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER
source activate my_env
python3 - << 'PYEOF'
import importlib.util, sys, time, warnings, torch
warnings.filterwarnings("ignore")
ROOT="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER"
sys.path.insert(0,ROOT); sys.path.insert(0,f"{ROOT}/scripts")

t_import=time.time()
import config
spec=importlib.util.spec_from_file_location("t",f"{ROOT}/scripts/train_tf_to_tg_model.py")
t=importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
print(f"{'import train script (gtfparse, torch, lightning)':<52}{time.time()-t_import:8.1f}s", flush=True)

def timed(label, fn):
    s=time.time(); r=fn(); d=time.time()-s
    print(f"{label:<52}{d:8.1f}s", flush=True); return r

emb   = timed("torch.load tf_embeddings.pt (1.27 GB)", lambda: torch.load(config.tf_embedding_cache_path, weights_only=True))
msk   = timed("torch.load tf_masks.pt", lambda: torch.load(config.tf_mask_cache_path, weights_only=True))
peaks = timed("torch.load atac_peak_tensor.pt (620 MB)", lambda: torch.load(config.tf_tg_atac_peak_cache_path, weights_only=True))
tr_   = timed("torch.load tftg_inputs_train.pt (15 GB)", lambda: torch.load(config.tf_tg_train_cache_path, weights_only=False))
va_   = timed("torch.load tftg_inputs_val.pt (2.0 GB)", lambda: torch.load(config.tf_tg_val_cache_path, weights_only=False))
te_   = timed("torch.load tftg_inputs_test.pt (7.4 GB) [now deferred]", lambda: torch.load(config.tf_tg_test_cache_path, weights_only=False))
am    = timed("torch.load atac_mat.pt (1.2 GB)", lambda: torch.load(config.tf_tg_atac_mat_cache_path, weights_only=True))
rm    = timed("torch.load rna_mat.pt (233 MB)", lambda: torch.load(config.tf_tg_rna_mat_cache_path, weights_only=True))

model = timed("create_new_tf_tg_regulation_model (incl torch.compile)",
    lambda: t.create_new_tf_tg_regulation_model(
        tf_bind_model_path=config.tf_dna_model_checkpoints[config.cell_type],
        tf_embeddings_tensor=emb, tf_mask_tensor=msk, keep_tf_peak_model_in_eval=True))
model = timed("model.cuda()", lambda: model.cuda())
timed("set_tf_embedding_table (1.27 GB H2D)", lambda: model.set_tf_embedding_table(emb, msk))

from torch.utils.data import DataLoader
ds = t.TFTGEdgeBagDataset(va_, tf_embeddings_tensor=emb, tf_mask_tensor=msk,
                          atac_peak_tensor=peaks, return_tf_indices=True)
dl = DataLoader(ds, batch_size=64, num_workers=6, collate_fn=t.collate_tftg_edge_bags,
                pin_memory=True, persistent_workers=True, prefetch_factor=4)
it=iter(dl)
b=timed("first batch from DataLoader (worker spawn)", lambda: next(it))
b={k:(v.cuda() if torch.is_tensor(v) else v) for k,v in b.items()}
def fwd():
    with torch.autocast("cuda",dtype=torch.float16):
        o=model(tf_idx=b["tf_idx"], tf_embedding=None, tf_mask=None,
                peak_sequences=b["peak_sequences"], peak_accessibility=b["peak_accessibility"],
                peak_distance=b["peak_distance"], peak_mask=b["peak_mask"],
                tf_expression=b["tf_expression"], tg_expression=b["tg_expression"],
                cell_mask=b["cell_mask"])
    torch.cuda.synchronize(); return o
timed("FIRST forward (triggers torch.compile of TF-DNA)", fwd)
timed("second forward (compiled, steady state)", fwd)
timed("third forward", fwd)
PYEOF
