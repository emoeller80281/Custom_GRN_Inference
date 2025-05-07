library(Seurat)
library(Signac)
library(Matrix)

# === MODIFY THESE PATHS ===
rna_dir <- "/gpfs/Labs/Uzun/DATA/PROJECTS/2025.AIM1_HV/SAMPLE/AML2/AML2_DX_RNA"
atac_dir <- "/gpfs/Labs/Uzun/DATA/PROJECTS/2025.AIM1_HV/SAMPLE/AML2/AML2_DX_ATAC"
output_dir <- "/gpfs/Labs/Uzun/DATA/PROJECTS/2025.AIM1_HV/INPUTS/AML2_DX"

# === RNA MATRIX ===
rna_matrix_file <- list.files(rna_dir, pattern = "matrix\\.mtx$", full.names = TRUE)
rna_genes_file  <- list.files(rna_dir, pattern = "features\\.tsv$", full.names = TRUE)
rna_barcodes_file <- list.files(rna_dir, pattern = "barcodes\\.tsv$", full.names = TRUE)

rna_counts <- readMM(rna_matrix_file)
rna_features <- read.delim(rna_genes_file, header = FALSE)
rna_barcodes <- read.delim(rna_barcodes_file, header = FALSE)

# Ensure row/col counts match
stopifnot(nrow(rna_features) == nrow(rna_counts))
stopifnot(nrow(rna_barcodes) == ncol(rna_counts))

rownames(rna_counts) <- make.unique(rna_features$V1)
colnames(rna_counts) <- rna_barcodes$V1

rna_seurat <- CreateSeuratObject(counts = rna_counts, assay = "RNA")
dense_rna <- as.matrix(GetAssayData(rna_seurat, layer = "counts", assay = "RNA"))
write.csv(dense_rna, file = file.path(output_dir, "dense_RNA_matrix.csv"), quote = FALSE)

# === ATAC MATRIX ===
atac_matrix_file <- list.files(atac_dir, pattern = "peak_matrix\\.mtx$", full.names = TRUE)
atac_peaks_file  <- list.files(atac_dir, pattern = "peaks.noheader\\.tsv$", full.names = TRUE)
atac_barcodes_file <- list.files(atac_dir, pattern = "barcodes\\.tsv$", full.names = TRUE)

atac_counts <- readMM(atac_matrix_file)
atac_features <- read.delim(atac_peaks_file, header = FALSE)
atac_barcodes <- read.delim(atac_barcodes_file, header = FALSE)

# Build unique peak names and validate shape
peak_names <- paste(atac_features$V1, atac_features$V2, atac_features$V3, sep = "-")
stopifnot(length(peak_names) == nrow(atac_counts))
stopifnot(nrow(atac_barcodes) == ncol(atac_counts))

rownames(atac_counts) <- make.unique(peak_names)
colnames(atac_counts) <- atac_barcodes$V1

chrom_assay <- CreateChromatinAssay(counts = atac_counts, sep = c("-", "-"))
atac_seurat <- CreateSeuratObject(counts = chrom_assay, assay = "peaks")

dense_atac <- as.matrix(GetAssayData(atac_seurat, layer = "counts", assay = "peaks"))
write.csv(dense_atac, file = file.path(output_dir, "dense_ATAC_matrix.csv"), quote = FALSE)
