#!/bin/bash -l
#SBATCH --job-name=tf_dna_landscape
#SBATCH --output=LOGS/tf_dna_landscape/tf_dna_landscape_%j.log
#SBATCH --error=LOGS/tf_dna_landscape/tf_dna_landscape_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=dense
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Writes new_plots/prediction_landscapes/plotly_tf_dna_landscape_<species>.html and its
# _summary.tsv. Submit from the repository root (which contains LOGS):
#   sbatch TETHER/bash_scripts/run_tf_dna_landscape.sh [extra args]
# Extra arguments go to create_plotly_tf_dna_landscape.py, for example:
#   --species hg38  --tfs Sox2 Pou5f1  --include-inferred-motifs
set -eo pipefail

PROJECT_DIR=/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER

cd "$PROJECT_DIR"

conda activate my_env

python scripts/model_evaluation/create_plotly_tf_dna_landscape.py "$@"
