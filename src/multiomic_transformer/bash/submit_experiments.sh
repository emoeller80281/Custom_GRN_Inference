#!/bin/bash
#
# Helper script to submit experiments
# Usage: ./submit_experiments.sh [experiment_indices]
#
# Examples:
#   ./submit_experiments.sh           # Submit all experiments (0-4)
#   ./submit_experiments.sh 0         # Submit only experiment 0
#   ./submit_experiments.sh 0,2,4     # Submit experiments 0, 2, and 4
#   ./submit_experiments.sh 0-2       # Submit experiments 0, 1, and 2

set -euo pipefail

cd /gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER

# Create log directory if it doesn't exist
mkdir -p LOGS/transformer_logs/experiments

# Display available experiments
echo ""
echo "=========================================="
echo "     AVAILABLE EXPERIMENTS"
echo "=========================================="
echo ""
echo "  [0] no_filter_to_nearest_gene"
echo "      Dataset: mESC_no_filter_to_nearest_gene"
echo "      Changes: FILTER_TO_NEAREST_GENE=false, HOPS=0"
echo ""
echo "  [1] smaller_window_size"
echo "      Dataset: mESC_smaller_window_size"
echo "      Changes: WINDOW_SIZE=500, HOPS=0"
echo ""
echo "  [2] larger_window_size"
echo "      Dataset: mESC_larger_window_size"
echo "      Changes: WINDOW_SIZE=1500, HOPS=0"
echo ""
echo "  [3] lower_max_peak_dist"
echo "      Dataset: mESC_lower_max_peak_dist"
echo "      Changes: MAX_PEAK_DISTANCE=50000, HOPS=0"
echo ""
echo "  [4] higher_max_peak_dist"
echo "      Dataset: mESC_higher_max_peak_dist"
echo "      Changes: MAX_PEAK_DISTANCE=150000, HOPS=0"
echo ""
echo "=========================================="
echo ""

# Determine which experiments to run
if [ $# -eq 0 ]; then
    # No arguments - submit all experiments
    ARRAY_SPEC="0-4"
    echo "Submitting ALL experiments (array indices: 0-4)"
else
    # Use provided argument as array specification
    ARRAY_SPEC="$1"
    echo "Submitting experiments with array indices: ${ARRAY_SPEC}"
fi

echo ""
read -p "Continue with submission? [y/N] " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Submission cancelled."
    exit 0
fi

# Submit the job array
JOB_ID=$(sbatch --array=${ARRAY_SPEC} \
    src/multiomic_transformer/bash/run_experiments.sh \
    | awk '{print $4}')

echo ""
echo "=========================================="
echo "  Job array submitted successfully!"
echo "  Job ID: ${JOB_ID}"
echo "  Array indices: ${ARRAY_SPEC}"
echo "=========================================="
echo ""
echo "Monitor jobs with:"
echo "  squeue -j ${JOB_ID}"
echo ""
echo "View logs in:"
echo "  LOGS/transformer_logs/experiments/"
echo ""
echo "Cancel jobs with:"
echo "  scancel ${JOB_ID}"
echo ""
