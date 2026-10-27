#!/bin/bash -l
#SBATCH --job-name=compare_grn_methods
#SBATCH --output=LOGS/compare_grn_methods/compare_grn_methods_%j.log
#SBATCH --error=LOGS/compare_grn_methods/compare_grn_methods_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G

# Compares the holdout TF-TG models against the other GRN inference methods: metrics, random
# baselines, boxplots, heatmaps and rank plots (celltype_model_evaluation.ipynb, cells 17-30).
# Needs the GRNs from run_generate_grns_from_model.sh. Submit from the repository root:
#   mkdir -p LOGS/compare_grn_methods
#   sbatch TETHER/bash_scripts/run_compare_grn_methods.sh [extra args]
# Extra arguments go to compare_grn_methods.py, for example:
#   --runs no_liver=celltype_joint_3926071_* no_mesc=celltype_joint_3926072_* ...
#   --plot_dir TETHER/new_plots/method_comparison_<tag>   --force_relabel
# To start after the GRN array job: sbatch --dependency=afterok:<grn_job_id> ...
set -eo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname)" == *hgtw* ]]; then
    echo "Submit with sbatch; this script must run on a compute node." >&2
    exit 1
fi

PROJECT_DIR=/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER
cd /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER

source activate my_env
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "Host: $(hostname)  Job: ${SLURM_JOB_ID}"
python "${PROJECT_DIR}/scripts/model_evaluation/compare_grn_methods.py" \
    --n_jobs "${SLURM_CPUS_PER_TASK:-8}" \
    "$@"
