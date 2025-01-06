#!/bin/bash -l

#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --mem-per-cpu=16G
#SBATCH -o LOGS/find_tf_motifs.log
#SBATCH -e LOGS/find_tf_motifs.err
#srun source /gpfs/Home/esm5360/miniconda3/envs/my_env

# source /gpfs/Home/esm5360/miniconda3/bin/activate my_env
# cd /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/Homer/bin

# module load parallel
export PATH="./Homer/bin:$PATH"

# Define directories and input file
homer_peak_file="./input/Homer_peaks.txt"
genome="hg38"
atac_data_file="./input/PBMC_ATAC.csv"
rna_data_file="./input/PBMC_RNA.csv"
motif_dir="./output/knownResults/"
output_dir="./output/homer_tf_motif_scores/"
tf_motif_binding_score_file="./output/total_motif_regulatory_scores.tsv"

# Ensure the output directory exists
mkdir -p "$output_dir"

# Format the scATAC-seq peaks into Homer format
echo "Creating Homer peak file"
python3 ./src/python_scripts/create_homer_peak_file.py \
    --atac_data_file "${atac_data_file}"

# Find TF binding motifs in the scATACseq peaks using the Homer genome
echo "Running Homer findMotifsGenome.pl"
touch $homer_peak_file
perl ./Homer/bin/findMotifsGenome.pl \
    "$homer_peak_file" \
    "$genome" \
    ./output \
    -size 200

# Create a table of TF binding motifs in ATAC-seq peaks
echo "Running Homer annotatePeaks.pl"
perl ./Homer/bin/annotatePeaks.pl \
    ./input/Homer_peaks.txt \
    hg38 \
    -m ./output/knownResults/known1.motif > ./output/known_motif_1_motif_to_peak.txt

# Function to process each motif file
process_motif_file() {
    motif_file=$1
    input_file=$2
    genome=$3
    output_dir=$4

    # Extract the motif file name (e.g., known<number>)
    motif_basename=$(basename "$motif_file" .motif)

    # Define the specific output file for this motif
    output_file="$output_dir/${motif_basename}_tf_motifs.txt"

    echo "Processing $motif_basename"
    
    # Run annotatePeaks.pl and save output to the specific file
    if annotatePeaks.pl "$input_file" "$genome" -m "$motif_file" > "$output_file"; then
        echo "Done!"
    else
        echo "Error processing $motif_file" >&2
    fi
}

export -f process_motif_file  # Export the function to make it accessible to parallel
export homer_peak_file genome output_dir  # Export variables to make them accessible to parallel

# Run the function in parallel for each .motif file
find "$motif_dir" -name "*.motif" | parallel -j 32 process_motif_file {} "$homer_peak_file" "$genome" "$output_dir"

echo "All motifs processed in parallel. Individual results saved in $output_dir"

# Parse the motifs in each peak for each TF
python3 src/python_scripts/parse_TF_peak_motifs.py \
    --input_dir "${output_dir}" \
    --output_file "${tf_motif_binding_score_file}" \
    --cpu_count 32

# Calculate a final trans-regulatory potential by scaling the motif
# binding score by the scRNA-seq expression for each TF
python3 src/python_scripts/find_overlapping_TFs.py \
    --rna_data_file "${rna_data_file}" \
    --tf_motif_binding_score_file "${tf_motif_binding_score_file}"


