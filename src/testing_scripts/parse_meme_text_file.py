hocomoco_meme_human = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/motif_information/hg38/motif_databases/HUMAN/HOCOMOCOv9.meme"

header = []
motif_dict = {}
with open(hocomoco_meme_human, 'r') as meme_file:
    current_motif = None
    for line in meme_file:
        if "MOTIF" in line:
            current_motif = line.split(" ")[2].strip()
            # print(current_motif)
            motif_dict[current_motif] = [line]
        elif current_motif:
            # print(line)
            motif_dict[current_motif].append(line)
        else:
            header.append(line)

for motif, motif_lines in motif_dict.items():
    print(motif)
    with open(f'/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/motif_information/hg38/memefiles/{motif}.meme', 'w') as outfile:
        outfile.writelines(header)
        outfile.writelines(motif_lines)
            
            