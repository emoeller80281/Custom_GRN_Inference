#!/bin/bash -l

#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --mem-per-cpu=16G
#SBATCH -o LOGS/find_tf_motifs.log
#SBATCH -e LOGS/find_tf_motifs.err
#srun source /gpfs/Home/esm5360/miniconda3/envs/my_env

source /gpfs/Home/esm5360/miniconda3/bin/activate my_env
cd /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/Homer/bin

module load parallel
export PATH="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/Homer/bin:$PATH"


# Define directories and input file
input_file="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/Homer_peaks.txt"
genome="hg38"
motif_dir="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/knownResults/"
output_dir="/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/homer_tf_motif_scores/"

# Ensure the output directory exists
mkdir -p "$output_dir"

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
export input_file genome output_dir  # Export variables to make them accessible to parallel

# Run the function in parallel for each .motif file
find "$motif_dir" -name "*.motif" | parallel -j 32 process_motif_file {} "$input_file" "$genome" "$output_dir"

echo "All motifs processed in parallel. Individual results saved in $output_dir"
