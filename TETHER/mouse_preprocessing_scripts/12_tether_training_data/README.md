# 12 — TETHER training data from the Argelaguet et al. 2022 atlas

Turns the published mouse organogenesis multiome atlas into a TETHER sample, and trains a
TF–TG model on it with a **transcription-factor** split instead of the usual chromosome split.

Sample: `data/sample_input_data/mESC/WT_timecourse_metacells/`

## Why a TF split

The standard TETHER split partitions on the target gene's chromosome (mm10 test = chr18–19),
which costs ~10% of the genome and still lets every TF appear in all three splits. Here the
question is different: *does the model generalise to regulators it has never seen?*

So the partition is on the TF:

| Split | TFs | Definition |
|---|---|---|
| **test** | 52 | TFs the paper's NMP-trajectory GRN implicated in NMP → {spinal cord, somitic mesoderm} |
| **val** | 16 | TF-disjoint 15% holdout carved from the training TFs |
| **train** | 94 | every other TF, across all cell types and **all chromosomes** |

Test TFs are read from the paper's own `global_chip_GRN_coef.txt.gz` — every TF in that table,
not just the ones with significant betas, so nothing the paper evaluated on that trajectory
leaks into training. The list includes T, SOX2, CDX2, LEF1, TCF7L1/2, OLIG2, PAX6, ZIC3,
GATA1/3/4/6, POU5F1.

Target genes are deliberately **shared** across splits (~21.7k genes appear in both train and
test under different TFs). That is the intended generalisation question; holding out genes too
would shrink the evaluation without making the TF question cleaner.

### One caveat on "no leakage"

The frozen TF–DNA submodule (`tf_dna_mm10_3697823`) was trained on **all** of these TFs,
including every test TF. That is leakage of *sequence-binding* preference, not of *regulatory*
relationships — the TF→TG question the model is scored on stays clean — but "unseen TF" is
only true at the regulation layer, not end to end.

## Files

| File | Purpose |
|---|---|
| `export_atac_metacells.R` | Lifts the ArchR metacell PeakMatrix out of R (`SummarizedExperiment` → HDF5). Run with the **`figr_env`** interpreter, not `my_env`. |
| `build_tether_inputs.py` | Writes `RE_pseudobulk.parquet`, `TG_pseudobulk.parquet`, `peak_to_gene_dist.parquet`. |

## Running it

```bash
# 1. ATAC metacells out of R  (~5 min)
/gpfs/Home/esm5360/miniconda3/envs/figr_env/bin/Rscript export_atac_metacells.R

# 2. Pseudobulk matrices + peak-to-gene links  (~2 min)
python3 build_tether_inputs.py

# 3. Cache the TF-TG edge bags with the TF split
sbatch ../../bash_scripts/03a_build_tf_to_tg_cache_nmp_split.sh

# 4. Train
sbatch ../../bash_scripts/03b_train_tf_to_tg_model_nmp_split.sh
```

`config.py` must be set to `species="mm10"`, `cell_type="mESC"`,
`sample_name="WT_timecourse_metacells"` (steps 3–4 read it).

## What the inputs contain

Columns are the paper's SEACells metacells, keyed `<sample>#<barcode>` — the **intersection**
of the RNA and ATAC metacell sets, so every column is a matched set of nuclei.

| | shape | normalisation |
|---|---|---|
| `TG_pseudobulk.parquet` | 32,201 genes × 1,896 metacells | CP10K → log1p → per-gene z-score |
| `RE_pseudobulk.parquet` | 192,251 peaks × 1,896 metacells | TF-IDF → log1p → per-peak z-score |
| `peak_to_gene_dist.parquet` | 644,675 links | peaks within 100 kb of a TSS (mm10 `gene_tss.bed`) |

Metacells come from the 9 wild-type timecourse libraries (E7.5 → E8.75). **Both CRISPR
libraries are excluded**, so the training data is an unperturbed developmental series.

### Normalisation differs from the older mESC samples

The per-library mESC samples (`E7.5_rep1`, …) were built by diffusion-smoothing *single cells*,
which leaves each feature centred but with std well below 1 (≈0.23 ATAC, ≈0.39 RNA) because
averaging over neighbours shrinks the variance. These are real aggregated metacells with no
smoothing step, so features are z-scored to unit variance instead — same centring, cleaner
scale. Don't mix the two conventions in one model.

## Cache sizing

The TF split keeps every chromosome in training, so the ground truth is ~878k edges, ~16×
what the per-library samples had. At `pct_true_edges=1.0` the edge bags would run to ~65 GB,
so true edges are subsampled to 30%; `true_false_ratio` stays at 10.0 to match the existing
runs so AUPRC remains comparable across samples.

## Provenance notes

- `cached_data/mESC_cache/mESC_merged_ground_truth.parquet` was **stale** (70 TFs, 54,784
  edges). Rebuilt from the raw files it is 1.75M edges / 362 TFs, of which 162 have TF
  embeddings. The old file is kept as `.bak_stale70tf`.
- The ATAC metacell matrix has 1,936 WT metacells and the RNA one 2,574; 1,896 are in both.
- `atac_metacell_counts.h5` is an intermediate (raw counts straight out of R) and can be
  deleted once the parquets exist.
