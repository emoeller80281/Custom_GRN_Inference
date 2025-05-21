# Load required libraries
suppressPackageStartupMessages({
  library(cicero)
  # library(monocle3)
  library(Signac)
  library(Seurat)
  library(GenomicRanges)
  library(Matrix)
  library(rtracklayer)
  library(reshape2)
  library(dplyr)
  library(tidyr)
  library(parallel)
  library(htmlwidgets)
  library(profvis)
  library(arrow)
})

# =============================================
# Utility Functions
# =============================================

log_message <- function(message) {
  cat(sprintf("[%s] %s\n", Sys.time(), message))
}

# =============================================
# Command-Line Arguments
# =============================================

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 4) {
  stop("Usage: Rscript script.R <atac_file_path> <output_dir> <chromsize_file_path> <gene_annot_file_path>")
}

atac_file_path <- args[1]
output_dir <- args[2]
chrom_sizes <- args[3]
gene_annot <- args[4]

# atac_file_path <- "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/mESC/mESC_filtered_L2_E7.5_rep2_ATAC.csv"
# output_dir <- "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output/mESC_full_test"
# chrom_sizes <- "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/mESC/mm10.chrom.sizes"
# gene_annot <- "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/mESC/Mus_musculus.GRCm39.113.gtf.gz"

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

log_message("Starting Cicero pipeline...")
log_message(sprintf("ATAC data file: %s", atac_file_path))
log_message(sprintf("Output directory: %s", output_dir))

# =============================================
# Load and Preprocess Data
# =============================================

log_message("Loading ATACseq data from Parquet...")
atac_tbl <- read_parquet(atac_file_path)
atac_data <- as.data.frame(collect(atac_tbl))

# Rename column 1 if needed
if (colnames(atac_data)[1] != "peak_id") {
  colnames(atac_data)[1] <- "peak_id"
}
atac_data$peak_id <- as.character(atac_data$peak_id)

# Ensure there are measure columns
if (ncol(atac_data) <= 1) {
  stop("No measure columns found in ATAC data.")
}

# Sanity check
log_message("Printing column names before melt...")
print(head(colnames(atac_data)))

log_message("Reshaping ATACseq datset to Cicero input format...")
# atac_long <- reshape2::melt(
#   atac_data,
#   id.vars = "peak_id",
#   variable.name = "cell",
#   value.name = "reads"
# ) %>%
#   dplyr::mutate(
#     numeric_reads = suppressWarnings(as.numeric(reads))
#   ) %>%
#   dplyr::filter(!is.na(numeric_reads) & numeric_reads > 0) %>%
#   dplyr::mutate(
#     peak_id = gsub("[:\\-]", "_", peak_id),
#     count = numeric_reads
#   ) %>%
#   dplyr::select(peak_id, cell, count)

log_message("Printing number of columns...")
print(ncol(atac_data))
print(colnames(atac_data)[1:5])

log_message("Checking for non-numeric columns (excluding 'peak_id')...")
non_numeric <- sapply(atac_data[ , -1, drop = FALSE], function(x) !is.numeric(x))
print(which(non_numeric))

if (ncol(atac_data) <= 1) stop("Only one column in data — no columns to melt.")
if (all(non_numeric)) stop("All columns (except peak_id) are non-numeric — nothing to melt.")

# Manually define measure.vars as all columns *except* "peak_id"
cell_columns <- setdiff(colnames(atac_data), "peak_id")

# Double check there are cell columns to melt
stopifnot(length(cell_columns) > 0)

print(str(atac_data[ , 1:5]))
print(summary(sapply(atac_data[ , -1], class)))  # Should be numeric

atac_long <- melt(
  atac_data,
  id_vars = "peak_id",
  variable.name = "cell",
  value.name = "count"
)

length(unique(atac_long$peak_id))
length(unique(atac_long$cell))
sum(is.na(atac_long$count))


log_message("    Done!")
str(atac_long)
log_message("Creating a Cicero cell_data_set (CDS) from ATAC data...")
log_message("    Running dimensionality reduction")
cds <- make_atac_cds(
  atac_long %>% dplyr::select(peak_id, cell, count) %>% as.data.frame() %>% `colnames<-`(NULL),
  binarize = TRUE
)

print(class(cds))  # Should say "cell_data_set"


set.seed(2017)
log_message("    Detecting genes in cds")
cds <- detectGenes(cds)

log_message("    Estimating cds size factor")
cds <- estimateSizeFactors(cds)

log_message("    Reducing dimensions for cds using tSNE")
# input_cds <- preprocessCDS(cds, norm_method = "none")
cds <- reduceDimension(cds, max_components = 2, num_dim=6,
                      reduction_method = 'tSNE', norm_method = "none")

log_message("    Extracting reduced tSNE coordinates")
tsne_coords <- t(reducedDimA(cds))
row.names(tsne_coords) <- row.names(pData(cds))
log_message("        Done!")

log_message("    Making Cicero CDS object")
cicero_obj <- make_cicero_cds(cds, reduced_coordinates = tsne_coords)
log_message("        Done!")

# =============================================
# Run Cicero
# =============================================

log_message("Running Cicero...")

window_size <- 500000
s_val <- 0.75

# Calculate distance penalty parameter for random genomic windows 
# (used to calculate distance_parameter in generate_cicero_models)
log_message("    Estimating distance parameter")
dist_penalty <- estimate_distance_parameter(
  cicero_obj,
  window = window_size, # How many base pairs used to calculate each individual model, furthest distance to compare sites
  sample_num = 100,
  distance_constraint = 50000,
  s = s_val, # Uses "tension globule" polymer model of DNA 
  genomic_coords = chrom_sizes
)

# Generate graphical LASSO models on all sites in a CDS object within overlapping genomic windows
log_message("    Generating Cicero graphical LASSO models")
cicero_models <- generate_cicero_models(
  cicero_obj,
  window = window_size, 
  distance_parameter = mean(dist_penalty), # Distance-based scaling of graphical LASSO regularization parameter
  s = s_val,
  genomic_coords = chrom_sizes
)

# Assembles the connections into a dataframe with cicero co-accessibility scores
log_message("    Assembling co-accessibility scores into a dataframe")
conns <- assemble_connections(
  cicero_models,
  silent = FALSE
)

write.csv(conns, file.path(output_dir, "cicero_peak_to_peak.csv"), row.names = FALSE)

# =============================================
# Process Gene Annotations
# =============================================

log_message("Processing gene annotations...")
gene_anno <- rtracklayer::import(gene_annot) %>% 
  as.data.frame() %>%
  select(seqnames, start, end, strand, gene_id, gene_name, transcript_id) %>%
  mutate(
    chromosome = paste0("chr", seqnames),
    gene = gene_id,
    symbol = gene_name
  )


# Prepare TSS annotations
log_message("Preparing TSS annotations...")
pos <- gene_anno %>% filter(strand == "+") %>%
  arrange(start) %>%
  distinct(gene_id, .keep_all = TRUE) %>%
  mutate(end = start + 1)

neg <- gene_anno %>% filter(strand == "-") %>%
  arrange(desc(start)) %>%
  distinct(gene_id, .keep_all = TRUE) %>%
  mutate(start = end - 1)

gene_annotation_sub <- bind_rows(pos, neg) %>%
  select(chromosome, start, end, gene = symbol)

# Annotate CDS
log_message("Annotating CDS...")
cds <- annotate_cds_by_site(cds, gene_annotation_sub)

# =============================================
# Generate Peak-Gene Associations
# =============================================
log_message("Generating peak-gene associations...")
peak_to_gene <- as.data.frame(fData(cds)) %>%
  select(site_name, gene) %>%
  filter(!is.na(gene)) 

write.csv(peak_to_gene, file.path(output_dir, "cicero_peak_to_gene.csv"), row.names = TRUE)

log_message("Cicero pipeline completed.")

