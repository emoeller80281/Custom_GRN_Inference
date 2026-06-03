#!/bin/bash -l
#SBATCH --job-name=generate_tf_embeddings
#SBATCH --output=LOGS/generate_tf_embeddings/%x_%A.log
#SBATCH --error=LOGS/generate_tf_embeddings/%x_%A.err
#SBATCH --time=12:00:00
#SBATCH -p dense
#SBATCH -N 1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --mem=128G

set -eo pipefail

source activate my_env

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/notebooks/simple_model_testing"
cd "${PROJECT_DIR}"

species="hg38"
entrez_email="luminarada@gmail.com"

# Download TF protein sequences from ChIP-Atlas and save as FASTA files
python ${PROJECT_DIR}/download_chipatlas.py \
    --species ${species} \
    --entrez_email ${entrez_email} \
    --num_workers 32

source activate tfbindformer
# Generate 3Di tokens for TF proteins using Foldseek and ProstT5
echo ""
echo "Generating 3Di tokens for TF proteins using Foldseek and ProstT5..."
FASTA_DIR="${PROJECT_DIR}/data/tf_data/${species}/tf_sequences"
OUT_DIR="${PROJECT_DIR}/data/tf_data/${species}/tf_3di_output"
TMP_DIR="${OUT_DIR}/tmp"
WEIGHTS_DIR="${OUT_DIR}/prostt5_weights"

mkdir -p "${OUT_DIR}" "${TMP_DIR}" "${WEIGHTS_DIR}" "LOGS/generate_tf_embeddings"

COMBINED_FASTA="${OUT_DIR}/tf_proteins.fasta"
DB_PREFIX="${OUT_DIR}/tf_proteins_3di_db"

# Combine individual FASTA files into one FASTA file
cat "${FASTA_DIR}"/*.fasta > "${COMBINED_FASTA}"

# Download ProstT5 weights once
# This creates/uses the directory specified by WEIGHTS_DIR
foldseek databases ProstT5 "${WEIGHTS_DIR}" "${TMP_DIR}"

# Create Foldseek DB from amino-acid FASTA using ProstT5
foldseek createdb \
    "${COMBINED_FASTA}" \
    "${DB_PREFIX}" \
    --prostt5-model "${WEIGHTS_DIR}" \
    --threads "${SLURM_CPUS_PER_TASK:-24}"

# Extract predicted 3Di states as FASTA
foldseek lndb \
    "${DB_PREFIX}_h" \
    "${DB_PREFIX}_ss_h"

foldseek convert2fasta \
    "${DB_PREFIX}_ss" \
    "${OUT_DIR}/tf_proteins_3di.fasta"

echo "Done! 3Di FASTA written to ${OUT_DIR}/tf_proteins_3di.fasta"

echo ""
echo "Extracting TF embeddings..."
python ${PROJECT_DIR}/scripts/extract_tf_embeddings.py \
  --aa_dir ${FASTA_DIR} \
  --di_fasta ${OUT_DIR}/tf_proteins_3di.fasta \
  --out_dir ${PROJECT_DIR}/data/tf_data/${species}/tf_embeddings/ \
  --d_model 128 \
  --device cuda

echo "Done! All steps finished."