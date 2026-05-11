#!/bin/bash
#SBATCH --job-name=extract_tf_embeddings
#SBATCH --output=LOGS/extract_tf_embeddings/%x_%j.log
#SBATCH --error=LOGS/extract_tf_embeddings/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 16
#SBATCH --mem=128G

set -eo pipefail

source activate my_env

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/notebooks/simple_model_testing"

cd ${PROJECT_DIR}

python ${PROJECT_DIR}/extract_tf_embeddings.py \
  --aa_dir ${PROJECT_DIR}/data/tf_data/tf_sequences \
  --di_fasta ${PROJECT_DIR}/data/tf_data/tf_3di_output/tf_proteins_3di.fasta \
  --out_dir ${PROJECT_DIR}/data/tf_data/tf_embeddings/