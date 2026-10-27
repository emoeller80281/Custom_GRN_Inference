#!/bin/bash -l
#SBATCH --job-name=homer_background
#SBATCH --output=LOGS/homer_background/%x_%j.log
#SBATCH --error=LOGS/homer_background/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --mem=16G

# Generates synthetic 256 bp background sequences that match a genome's sequence content.
#   1. Sample random 256 bp windows from the genome (numbered chromosomes and chrX; no N
#      bases) as targets.
#   2. homer2 background -model fits a Markov chain (-ikmer 3: each base depends on the
#      previous 3) to the targets in each GC bin and generates new sequences from it,
#      so the output matches the genome's 4-mer content (incl. CpG depletion) and GC
#      spread, without copying real genomic TF sites.
# Submit from TETHER (which contains LOGS):
#   sbatch bash_scripts/generate_homer_background_sequences.sh [mm10|hg38]

set -eo pipefail

SPECIES="${1:-mm10}"

PROJECT_DIR="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/TETHER"
cd $PROJECT_DIR

HOMER2="/gpfs/Home/esm5360/.conda/envs/LINGER/share/homer/bin/homer2"
GENOME_FASTA="${PROJECT_DIR}/../data/genome_data/reference_genome/${SPECIES}/${SPECIES}.fa"
OUTPUT_DIR="${PROJECT_DIR}/../data/genome_data/homer_background/${SPECIES}"
OUTPUT_PREFIX="${OUTPUT_DIR}/${SPECIES}_256bp"

SEQ_LEN=256
NUM_TARGET_WINDOWS=20000
NUM_BG_SEQS=50000
KMER=3
SEED=42

mkdir -p "$OUTPUT_DIR"

source activate my_env

echo "[INFO] Sampling ${NUM_TARGET_WINDOWS} random ${SEQ_LEN} bp ${SPECIES} windows..."
python3 - "$GENOME_FASTA" "${OUTPUT_PREFIX}.random_windows.fasta" $SEQ_LEN $NUM_TARGET_WINDOWS $SEED <<'EOF'
import re
import sys
import numpy as np
import pyfaidx

genome_fasta, out_fasta, seq_len, num_windows, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
rng = np.random.default_rng(seed)

genome = pyfaidx.Fasta(genome_fasta)
chroms = [c for c in genome.keys() if re.fullmatch(r"chr(\d+|X)", c)]
chrom_lens = np.array([len(genome[c]) for c in chroms])
chrom_probs = chrom_lens / chrom_lens.sum()

with open(out_fasta, "w") as f:
    n = 0
    while n < num_windows:
        chrom = chroms[rng.choice(len(chroms), p=chrom_probs)]
        start = int(rng.integers(0, len(genome[chrom]) - seq_len))
        seq = str(genome[chrom][start:start + seq_len]).upper()
        if "N" in seq:
            continue
        f.write(f">{chrom}:{start}-{start + seq_len}\n{seq}\n")
        n += 1
EOF

echo "[INFO] Generating ${NUM_BG_SEQS} background sequences with homer2 background (-model -ikmer ${KMER})..."
$HOMER2 background \
    -i "${OUTPUT_PREFIX}.random_windows.fasta" \
    -model \
    -excludeNs \
    -ikmer $KMER \
    -size $SEQ_LEN \
    -N $NUM_BG_SEQS \
    -seed $SEED \
    -o "$OUTPUT_PREFIX"

echo "[INFO] Comparing sequence content: ${SPECIES} windows vs. HOMER background vs. independent-base sampling..."
python3 - "${OUTPUT_PREFIX}.random_windows.fasta" "${OUTPUT_PREFIX}.bg.sequences.fasta" $SEED <<'EOF'
import sys
import numpy as np

def read_fasta(path):
    seqs, cur = [], []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur).upper())
            cur = []
        elif line:
            cur.append(line)
    if cur:
        seqs.append("".join(cur).upper())
    return seqs

def summarize(name, seqs):
    joined = "".join(seqs)
    gc_per_seq = np.array([(s.count("G") + s.count("C")) / len(s) for s in seqs])
    base_freq = {b: joined.count(b) / len(joined) for b in "ACGT"}
    cpg = sum(s.count("CG") for s in seqs) / sum(len(s) - 1 for s in seqs)
    cpg_obs_exp = cpg / (base_freq["C"] * base_freq["G"])
    lengths = {len(s) for s in seqs}
    print(f"{name:<28} n={len(seqs):>6}  len={sorted(lengths)}  N={joined.count('N')}  "
          f"GC={np.mean(gc_per_seq):.3f} (sd {np.std(gc_per_seq):.3f})  "
          f"A={base_freq['A']:.3f} C={base_freq['C']:.3f} G={base_freq['G']:.3f} T={base_freq['T']:.3f}  "
          f"CpG obs/exp={cpg_obs_exp:.2f}")

targets = read_fasta(sys.argv[1])
background = read_fasta(sys.argv[2])
rng = np.random.default_rng(int(sys.argv[3]))

# Each base drawn independently with the genome windows' base frequencies (the old sampler)
joined = "".join(targets)
base_freq = np.array([joined.count(b) for b in "ACGT"], dtype=float)
independent = ["".join(rng.choice(list("ACGT"), size=256, p=base_freq / base_freq.sum())) for _ in range(5000)]

summarize("genome random windows", targets)
summarize("homer2 background -model", background)
summarize("independent-base sampling", independent)
EOF

echo "[INFO] Done. Background FASTA: ${OUTPUT_PREFIX}.bg.sequences.fasta"
