import pandas as pd
import re

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
    # with open(f'/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/motif_information/hg38/memefiles/{motif}.meme', 'w') as outfile:
    #     outfile.writelines(header)
    #     outfile.writelines(motif_lines)
            
    # Find the line where the letter-probability matrix starts.
    matrix_start = None
    for i, line in enumerate(motif_lines):
        if line.strip().startswith("letter-probability matrix:"):
            matrix_start = i + 1  # the matrix numbers start on the next line
            break

    if matrix_start is None:
        raise ValueError("Could not find the letter-probability matrix in the file.")

    # Collect the matrix rows.
    matrix_lines = []
    for line in motif_lines[matrix_start:]:
        # Strip whitespace; stop if the line is empty or doesn't start with a digit.
        striped_line = line.strip()
        if not striped_line or not re.search(r'\d', striped_line):
            break
        matrix_lines.append(striped_line)

    # Parse each line into a list of floats.
    matrix = [list(map(float, re.split(r'\s+', line))) for line in matrix_lines]

    # If ALPHABET is defined in the file (e.g., "ALPHABET= ACGT") we can use that.
    alphabet = "ACGT"  # Adjust if your alphabet differs.

    # Create a pandas DataFrame.
    df = pd.DataFrame(matrix, columns=list(alphabet))
    df.index = df.index + 1  # Position numbers starting at 1
    df.index.name = "Pos"
    
    df.to_csv(f'/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/motif_information/hg38/hg38_motif_meme_files/{motif}.txt', sep="\t", header=True, index=True)