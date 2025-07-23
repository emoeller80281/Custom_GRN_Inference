#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH -c 64
#SBATCH --mem=256G
#SBATCH -o /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/testing_scripts/map_peaks_to_hic.log
#SBATCH -e /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/testing_scripts/map_peaks_to_hic.err

set -euo pipefail

source activate my_env

python /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/dev/testing_scripts/map_peaks_to_hic.py