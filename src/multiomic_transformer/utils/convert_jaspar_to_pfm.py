from Bio import motifs
from glob import glob
import os
import re

in_dir = "/home/emoeller/github/grn_inference/data/motif_information/mm10/JASPAR/jaspar_files"
out_dir = "/home/emoeller/github/grn_inference/data/motif_information/mm10/JASPAR/pfm_files"
os.makedirs(out_dir, exist_ok=True)

def safe_name(name: str) -> str:
    # Replace unsafe path characters with underscores
    return re.sub(r'[\\/<>:"|?*]', '_', name)

for fp in glob(f"{in_dir}/*.jaspar"):
    with open(fp) as handle:
        for m in motifs.parse(handle, "jaspar"):
            motif_name = safe_name(m.name or m.matrix_id)
            out_path = os.path.join(out_dir, f"{motif_name}.pfm")
            with open(out_path, "w") as out:
                for row in m.counts.values():
                    out.write(" ".join(str(x) for x in row) + "\n")

print("Converted JASPAR → PFM")
