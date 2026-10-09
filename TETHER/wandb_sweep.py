"""W&B sweep agent entry point for the cell-type TF->TG model.

Each sweep run calls scripts/train_cached_tf_to_tg_celltype_model.py in-process with
the run's W&B config converted to command-line flags. wandb_sweep.yaml holds both the
fixed settings (datasets, holdouts) and the swept hyperparameters:
  - list value   -> repeated flag  (dataset: [a, b] -> --dataset a --dataset b)
  - bool value   -> --flag / --no-flag (only for BooleanOptionalAction flags)
  - other values -> --flag value

The training script reads only the per-sample caches built by
bash_scripts/03c_build_celltype_data.sh. The settings that select those caches
(species, max_peaks_per_tg, min_cells_per_slice, peak_flank_size, true_false_ratio,
...) must match 03c. min_cells_per_slice is pinned so that max_cells_per_pair can be
swept without a cache rebuild; sweeping max_peaks_per_tg needs a 03c build per value.
"""
import logging
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerFn
import wandb

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

import scripts.train_cached_tf_to_tg_celltype_model as train_celltype  # noqa: E402

WANDB_PROJECT = "celltype-TF-TG"

# Best-so-far summaries for these validation metrics, stored as "<metric>_best".
# The sweep optimizes one of them (see metric.name in wandb_sweep.yaml), because the
# last-epoch value is not the best epoch's value.
TRACKED_METRICS = (
    "val/auroc",
    "val/auprc",
    "val/macro_celltype_auroc",
    "val/macro_celltype_auprc",
)


class BestMetricSummary(pl.Callback):
    """Write the best value of each tracked validation metric to the W&B summary."""

    def __init__(self, metric_names):
        self.metric_names = metric_names
        self.best = {}

    def on_validation_end(self, trainer, pl_module):
        # Skip the step-zero validation of the untrained model and the sanity check.
        if trainer.sanity_checking or trainer.state.fn != TrainerFn.FITTING:
            return
        for name in self.metric_names:
            value = trainer.callback_metrics.get(name)
            if value is None:
                continue
            value = float(value)
            if value > self.best.get(name, float("-inf")):
                self.best[name] = value
                wandb.run.summary[f"{name}_best"] = value
                wandb.run.summary[f"{name}_best_epoch"] = trainer.current_epoch


def config_to_argv(run_config):
    argv = []
    for key, value in run_config.items():
        if key.startswith("_") or value is None:
            continue
        if isinstance(value, bool):
            argv.append(f"--{key}" if value else f"--no-{key}")
        elif isinstance(value, (list, tuple)):
            for item in value:
                argv.extend([f"--{key}", str(item)])
        else:
            argv.extend([f"--{key}", str(value)])
    return argv


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    run = wandb.init(project=WANDB_PROJECT, job_type="celltype_tf_tg_sweep")
    run_config = dict(run.config)

    # Each run gets its own output directory. The training script would otherwise
    # reuse the newest celltype_* directory under one SLURM job ID, which every run
    # of a sweep agent shares.
    sweep_id = run.sweep_id or "no_sweep"
    output_dir = PROJECT_DIR / "checkpoints" / "wandb_sweep" / "celltype" / sweep_id / run.id

    argv = config_to_argv(run_config) + [
        "--output_dir", str(output_dir),
        "--run_name", f"sweep_{run.id}",
        "--wandb_project", WANDB_PROJECT,
    ]
    if "num_workers" not in run_config:
        argv += ["--num_workers", os.environ.get("SLURM_CPUS_PER_TASK", "4")]

    logging.info("Training arguments: %s", " ".join(argv))
    # The training script attaches its WandbLogger to this run and calls wandb.finish().
    train_celltype.main(argv, extra_callbacks=[BestMetricSummary(TRACKED_METRICS)])


if __name__ == "__main__":
    main()
