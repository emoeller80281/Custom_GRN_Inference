#!/usr/bin/env Rscript
# Export the ArchR metacell PeakMatrix (SummarizedExperiment, R-only) to HDF5 so the
# Python side of the TETHER pipeline can read it.
#
# Source : Argelaguet et al. 2022 dropbox download,
#          results/atac/archR/metacells/all_cells/PeakMatrix/PeakMatrix_summarized_experiment_metacells.rds
# Output : peak x metacell raw counts + peak IDs + metacell IDs, in one .h5
#
# Run with the figr_env interpreter (R 4.3.3 + SummarizedExperiment + rhdf5):
#   /gpfs/Home/esm5360/miniconda3/envs/figr_env/bin/Rscript export_atac_metacells.R
#
# Only the wild-type timecourse libraries are exported; both CRISPR libraries are
# dropped so the training data is an unperturbed developmental series.

suppressMessages({
  library(SummarizedExperiment)
  library(rhdf5)
})

EXTRACT <- "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/dropbox_data/extracted"
OUTDIR <- "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/data/sample_input_data/mESC/WT_timecourse_metacells"
OUTFILE <- file.path(OUTDIR, "atac_metacell_counts.h5")

WT_SAMPLES <- c(
  "E7.5_rep1", "E7.5_rep2", "E7.75_rep1",
  "E8.0_rep1", "E8.0_rep2",
  "E8.5_rep1", "E8.5_rep2",
  "E8.75_rep1", "E8.75_rep2"
)

se_path <- file.path(
  EXTRACT,
  "results/atac/archR/metacells/all_cells/PeakMatrix/PeakMatrix_summarized_experiment_metacells.rds"
)

cat("Reading", se_path, "\n")
se <- readRDS(se_path)
cat("  full matrix:", nrow(se), "peaks x", ncol(se), "metacells\n")

# Metacell IDs are "<sample>#<barcode>"; keep only the WT timecourse libraries.
metacell_sample <- sub("#.*$", "", colnames(se))
keep <- metacell_sample %in% WT_SAMPLES
se <- se[, keep]
cat("  after dropping CRISPR libraries:", ncol(se), "metacells\n")
print(table(sub("#.*$", "", colnames(se))))

mat <- assay(se, "PeakMatrix")
storage.mode(mat) <- "double"

cat("  counts range:", range(mat), "| pct zero:",
    round(100 * mean(mat == 0), 1), "%\n")

if (file.exists(OUTFILE)) file.remove(OUTFILE)
h5createFile(OUTFILE)

# Chunked so the writer never materialises a second copy of the full matrix.
h5createDataset(OUTFILE, "counts",
                dims = c(nrow(mat), ncol(mat)),
                storage.mode = "double",
                chunk = c(min(8192L, nrow(mat)), min(64L, ncol(mat))),
                level = 4)

chunk <- 128L
for (start in seq(1L, ncol(mat), by = chunk)) {
  stop_i <- min(start + chunk - 1L, ncol(mat))
  h5write(mat[, start:stop_i, drop = FALSE], OUTFILE, "counts",
          index = list(NULL, start:stop_i))
}

h5write(rownames(se), OUTFILE, "peak_ids")
h5write(colnames(se), OUTFILE, "metacell_ids")
H5close()

cat("Wrote", OUTFILE, "\n")
cat("  dims:", nrow(mat), "x", ncol(mat), "\n")
