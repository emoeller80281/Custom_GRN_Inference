from Bio import motifs

motif_meme_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/yasin_motif_binding_code/MEME_ALL_MOTIFS"

first_meme_file = f'{motif_meme_dir}/M00008_2.00.meme'

with open(first_meme_file, 'r') as memefile:
    for m in motifs.parse(memefile, "MEME"):
        print(m.consensus)