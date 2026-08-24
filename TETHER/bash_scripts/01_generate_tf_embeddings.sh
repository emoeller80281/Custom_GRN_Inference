#!/bin/bash -l
#SBATCH --job-name=generate_tf_embeddings
#SBATCH --output=LOGS/generate_tf_embeddings/%x_%A.log
#SBATCH --error=LOGS/generate_tf_embeddings/%x_%A.err
#SBATCH --time=24:00:00
#SBATCH -p dense
#SBATCH -N 1
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --mem=128G

# Build TF protein embeddings for every species, in ONE shared coordinate system.
#
# Both species run through here together on purpose. The 2048 -> 128 reduction has to be
# the same function for every TF or the embeddings are not comparable, and fitting it once
# across mm10 + hg38 is what makes cross-species work (and orthologs) meaningful. Fitting
# per species would leave mouse and human in different bases.
#
# Stages:
#   per species : ChIP-Atlas FASTAs -> Foldseek 3Di -> ProstT5 raw 2048-d embeddings
#   once        : fit one PCA over the residues of every TF of every species
#   per species : apply that saved projection -> 128-d embeddings
#   once        : validate (basis / orthologs / DBD family) before anything consumes them
#
# The raw stage skips TFs it has already embedded, so a timeout can simply be resubmitted.
# Everything downstream of this script must be rebuilt afterwards: the TF-DNA and TF-TG
# caches with --force_reload (02a, 03a) and both models retrained (02b, 03b). Existing
# checkpoints cannot be reused -- their input distribution changes completely.

set -eo pipefail

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER"
DATA_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data"
cd "${PROJECT_DIR}"

SPECIES_LIST=("mm10" "hg38")
entrez_email="luminarada@gmail.com"
D_MODEL=128

# Shared by every species -- deliberately not under any one species' directory.
PROJECTION="${DATA_DIR}/tf_data/tf_embedding_projection_d${D_MODEL}.pt"

mkdir -p "LOGS/generate_tf_embeddings"

# ---------------------------------------------------------------------------
# 1. Per species: sequences -> 3Di -> raw 2048-d ProstT5 embeddings
# ---------------------------------------------------------------------------
for species in "${SPECIES_LIST[@]}"; do

    FASTA_DIR="${DATA_DIR}/tf_data/${species}/tf_sequences"
    OUT_DIR="${DATA_DIR}/tf_data/${species}/tf_3di_output"
    TMP_DIR="${OUT_DIR}/tmp"
    WEIGHTS_DIR="${OUT_DIR}/prostt5_weights"
    RAW_DIR="${DATA_DIR}/tf_data/${species}/tf_embeddings_raw"

    COMBINED_FASTA="${OUT_DIR}/tf_proteins.fasta"
    DB_PREFIX="${OUT_DIR}/tf_proteins_3di_db"
    DI_FASTA="${OUT_DIR}/tf_proteins_3di.fasta"

    mkdir -p "${OUT_DIR}" "${TMP_DIR}" "${WEIGHTS_DIR}" "${RAW_DIR}"

    echo ""
    echo "=============================================================="
    echo "  ${species}: TF sequences and 3Di tokens"
    echo "=============================================================="

    source activate my_env

    # Download TF protein sequences from ChIP-Atlas and save as FASTA files
    python ${PROJECT_DIR}/download_organism_chipatlas.py \
        --species ${species} \
        --entrez_email ${entrez_email} \
        --num_workers 32

    source activate tfbindformer

    # Foldseek/ProstT5 3Di prediction is the slow part (~25 min per species, on CPU) and
    # its output only changes when the TF set does. Decide by COVERAGE, not timestamps: an
    # mtime comparison reports "stale" whenever anything touches the sequence directory,
    # which triggered a full 25-minute regeneration for a 3Di FASTA that already had a
    # record for all 476 sequences.
    DI_STATUS=$(python - "${FASTA_DIR}" "${DI_FASTA}" <<'PY'
import glob, re, sys
from pathlib import Path

fasta_dir, di_fasta = sys.argv[1], sys.argv[2]


def accessions(paths):
    found = set()
    for path in paths:
        with open(path) as handle:
            for line in handle:
                if line.startswith(">"):
                    match = re.search(r"(NP_\d+\.\d+)", line)
                    found.add(match.group(1) if match else line[1:].split()[0])
    return found


if not Path(di_fasta).is_file() or Path(di_fasta).stat().st_size == 0:
    print("stale")
    sys.exit()

wanted = accessions(glob.glob(f"{fasta_dir}/*.fasta"))
have = accessions([di_fasta])
missing = wanted - have

print(f"{len(wanted)} sequences, {len(have)} 3Di records, {len(missing)} missing", file=sys.stderr)
print("current" if not missing else "stale")
PY
)

    if [[ "${DI_STATUS}" == "current" ]]; then
        echo "3Di FASTA already covers every sequence, skipping Foldseek: ${DI_FASTA}"
    else
        echo "Generating 3Di tokens for TF proteins using Foldseek and ProstT5..."

        # Foldseek refuses to overwrite an existing lndb symlink, so a rerun dies on
        # leftovers from a previous run -- including links pointing at paths that no longer
        # exist. Everything matching this prefix is derived data that createdb regenerates.
        rm -f "${DB_PREFIX}"*

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
            "${DI_FASTA}"

        echo "Done! 3Di FASTA written to ${DI_FASTA}"
    fi

    echo ""
    echo "Extracting raw ${species} ProstT5 embeddings (2048-d per residue)..."
    python ${PROJECT_DIR}/scripts/extract_tf_embeddings.py \
        --stage raw \
        --aa_dir "${FASTA_DIR}" \
        --di_fasta "${DI_FASTA}" \
        --raw_dir "${RAW_DIR}" \
        --device cuda
done

# ---------------------------------------------------------------------------
# 2. Once: fit the shared projection across every species
# ---------------------------------------------------------------------------
source activate my_env

RAW_DIRS=()
for species in "${SPECIES_LIST[@]}"; do
    RAW_DIRS+=("${DATA_DIR}/tf_data/${species}/tf_embeddings_raw")
done

echo ""
echo "=============================================================="
echo "  Fitting the shared ${D_MODEL}-d projection across: ${SPECIES_LIST[*]}"
echo "=============================================================="
python ${PROJECT_DIR}/scripts/extract_tf_embeddings.py \
    --stage fit \
    --raw_dir "${RAW_DIRS[@]}" \
    --projection "${PROJECTION}" \
    --d_model ${D_MODEL}

# ---------------------------------------------------------------------------
# 3. Per species: apply the shared projection
# ---------------------------------------------------------------------------
# --overwrite on purpose: any embedding left over from an earlier run is in a different
# basis, and mixing bases is exactly the failure this rewrite exists to prevent.
for species in "${SPECIES_LIST[@]}"; do
    echo ""
    echo "Projecting ${species} embeddings to ${D_MODEL}-d..."
    python ${PROJECT_DIR}/scripts/extract_tf_embeddings.py \
        --stage project \
        --raw_dir "${DATA_DIR}/tf_data/${species}/tf_embeddings_raw" \
        --projection "${PROJECTION}" \
        --out_dir "${DATA_DIR}/tf_data/${species}/tf_embeddings" \
        --overwrite
done

# ---------------------------------------------------------------------------
# 4. Once: validate before anything downstream consumes these
# ---------------------------------------------------------------------------
VALIDATE_DIRS=()
for species in "${SPECIES_LIST[@]}"; do
    VALIDATE_DIRS+=("${species}=${DATA_DIR}/tf_data/${species}/tf_embeddings")
done

echo ""
echo "=============================================================="
echo "  Validating embeddings"
echo "=============================================================="
python ${PROJECT_DIR}/scripts/validate_tf_embeddings.py \
    --embedding_dir "${VALIDATE_DIRS[@]}" \
    --strict

echo ""
echo "Done! All steps finished."
echo "Next: rebuild the caches (02a, 03a with --force_reload) and retrain (02b, 03b)."
