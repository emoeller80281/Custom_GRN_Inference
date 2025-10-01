from Bio import motifs
from glob import glob
import os

in_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/motif_information/hg38/JASPAR/jaspar_files"
out_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/motif_information/hg38/JASPAR/pfm_files"
os.makedirs(out_dir, exist_ok=True)

for fp in glob(f"{in_dir}/*.jaspar"):
    with open(fp) as handle:
        for m in motifs.parse(handle, "jaspar"):
            out_path = os.path.join(out_dir, f"{m.name or m.matrix_id}.pfm")
            os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "w") as out:
                for row in m.counts.values():
                    out.write(" ".join(str(x) for x in row) + "\n")
print("Converted JASPAR → PFM")
