import pathlib
import os

knownResults_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/K562/K562_human_filtered/homer_results/knownResults"
homerResults_dir = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/K562/K562_human_filtered/homer_results/homerResults"

gene_aliases = ["CXXC1", "PCCX1", "CGBP", "CFP1", "HsT2645", "ZCGPC1", "HCGBP", "SPP1"]

def find_homer_motif_tf_names(path):
    tf_list = []
    for file in os.listdir(path):
        if pathlib.Path(file).suffix == ".motif":
            file_path = os.path.join(path, file)
            with open(file_path, "r") as motif_file:
                first_line = motif_file.readline()
                tf_string = first_line.split("\t")[1].split("(")[0]
                tf_list.append(tf_string)
                if tf_string in gene_aliases:
                    print(f"CXXC1 alias '{tf_string}' is in {file}")
    return tf_list

known_tf_list = find_homer_motif_tf_names(knownResults_dir)
homer_tf_list = find_homer_motif_tf_names(homerResults_dir)


if "CXXC1" in known_tf_list:
    print("True")
else:
    print("False")
    
if "CXXC1" in homer_tf_list:
    print("True")
else:
    print("False")