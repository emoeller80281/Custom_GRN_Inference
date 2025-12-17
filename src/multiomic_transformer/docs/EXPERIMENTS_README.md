# Experiment Management Scripts

This directory contains scripts for running multiple GRN inference experiments with different configurations.

## Files

- **`run_experiments.sh`** - Main SLURM job array script that runs experiments with different settings
- **`submit_experiments.sh`** - Helper script to easily submit experiment job arrays
- **`run_pipeline.sh`** - Single-instance pipeline script (can be run standalone)

## Quick Start

### Submit All Experiments

```bash
./src/multiomic_transformer/bash/submit_experiments.sh
```

### Submit Specific Experiments

```bash
# Submit only experiment 0
./src/multiomic_transformer/bash/submit_experiments.sh 0

# Submit experiments 0, 2, and 4
./src/multiomic_transformer/bash/submit_experiments.sh 0,2,4

# Submit experiments 0 through 2
./src/multiomic_transformer/bash/submit_experiments.sh 0-2
```

## Configured Experiments

All experiments start from the same baseline settings and modify specific parameters:

### Baseline Settings
- **MIN_GENES_PER_CELL**: 200
- **MIN_PEAKS_PER_CELL**: 200
- **FILTER_TYPE**: count
- **FILTER_OUT_LOWEST_COUNTS_GENES**: 3
- **FILTER_OUT_LOWEST_COUNTS_PEAKS**: 3
- **NEIGHBORS_K**: 20
- **PCA_COMPONENTS**: 25
- **HOPS**: 1 (default, changed to 0 in all experiments)
- **SELF_WEIGHT**: 1.0
- **WINDOW_SIZE**: 1000
- **DISTANCE_SCALE_FACTOR**: 20000
- **MAX_PEAK_DISTANCE**: 100000
- **DIST_BIAS_MODE**: logsumexp
- **FILTER_TO_NEAREST_GENE**: true

### Experiment 0: no_filter_to_nearest_gene
**Dataset**: `mESC_no_filter_to_nearest_gene`

**Changes**:
- `FILTER_TO_NEAREST_GENE`: true → **false**
- `HOPS`: 1 → **0**

Tests the impact of allowing peaks to be associated with multiple genes instead of only the nearest gene.

### Experiment 1: smaller_window_size
**Dataset**: `mESC_smaller_window_size`

**Changes**:
- `WINDOW_SIZE`: 1000 → **500**
- `HOPS`: 1 → **0**

Tests the effect of smaller genomic windows for aggregating ATAC-seq peaks.

### Experiment 2: larger_window_size
**Dataset**: `mESC_larger_window_size`

**Changes**:
- `WINDOW_SIZE`: 1000 → **1500**
- `HOPS`: 1 → **0**

Tests the effect of larger genomic windows for aggregating ATAC-seq peaks.

### Experiment 3: lower_max_peak_dist
**Dataset**: `mESC_lower_max_peak_dist`

**Changes**:
- `MAX_PEAK_DISTANCE`: 100000 → **50000**
- `HOPS`: 1 → **0**

Tests using a more restrictive peak-to-gene distance threshold (50kb instead of 100kb).

### Experiment 4: higher_max_peak_dist
**Dataset**: `mESC_higher_max_peak_dist`

**Changes**:
- `MAX_PEAK_DISTANCE`: 100000 → **150000**
- `HOPS`: 1 → **0**

Tests using a more permissive peak-to-gene distance threshold (150kb instead of 100kb).

## Adding New Experiments

To add new experiments, edit the `EXPERIMENTS` array in `run_experiments.sh`:

```bash
EXPERIMENTS=(
    "experiment_name|dataset_name|PARAM1=value1;PARAM2=value2"
)
```

**Format**:
- `experiment_name`: Short descriptive name for logging
- `dataset_name`: Dataset name used for output directories
- Parameter overrides: Semicolon-separated list of `PARAMETER=value` pairs

**Example**:
```bash
EXPERIMENTS=(
    # ... existing experiments ...
    "custom_experiment|mESC_custom|WINDOW_SIZE=750;MAX_PEAK_DISTANCE=125000;HOPS=0"
)
```

Then update the `--array` parameter in the SBATCH header and `submit_experiments.sh` to include the new experiment index.

## Monitoring Jobs

### Check job status
```bash
squeue -u $USER
```

### View logs
```bash
# List all experiment logs
ls -lh LOGS/transformer_logs/experiments/

# Follow a specific experiment log
tail -f LOGS/transformer_logs/experiments/grn_experiments_JOBID_ARRAYINDEX.log
```

### Cancel jobs
```bash
# Cancel all jobs in the array
scancel JOBID

# Cancel specific array indices
scancel JOBID_0,2,4
```

## Output Structure

Each experiment creates its own output directory:

```
/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/
└── experiments/
    ├── mESC_no_filter_to_nearest_gene/
    ├── mESC_smaller_window_size/
    ├── mESC_larger_window_size/
    ├── mESC_lower_max_peak_dist/
    └── mESC_higher_max_peak_dist/
```

Each experiment directory contains:
- Trained model checkpoints
- Training logs and metrics
- Model embeddings
- Vocabulary files
- Run parameters (JSON)

## Notes

- All experiments use the same training hyperparameters (epochs, batch size, learning rate, etc.)
- Preprocessing parameters are experiment-specific and defined in the `EXPERIMENTS` array
- The job array will automatically distribute experiments across available nodes
- Each experiment is independent and can fail/succeed without affecting others
