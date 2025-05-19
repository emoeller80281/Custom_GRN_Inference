# Load required libraries
suppressPackageStartupMessages({
  library(cicero)
  library(monocle3)
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
# Read and collect ATAC matrix from Parquet
atac_tbl <- read_parquet(atac_file_path)
atac_data <- collect(atac_tbl)
atac_data <- as.data.frame(atac_data)

log_message("Reshaping ATACseq datset to Cicero input format...")
atac_long <- reshape2::melt(
  atac_data,
  id.vars = "peak_id",
  variable.name = "cell",
  value.name = "reads"
) %>%
  dplyr::mutate(
    numeric_reads = suppressWarnings(as.numeric(reads))
  ) %>%
  dplyr::filter(!is.na(numeric_reads) & numeric_reads > 0) %>%
  dplyr::mutate(
    peak_id = gsub("[:\\-]", "_", peak_id),
    count = as.integer(numeric_reads)
  ) %>%
  dplyr::select(peak_id, cell, count)

length(unique(atac_long$peak_id))
length(unique(atac_long$cell))
sum(is.na(atac_long$count))


log_message("    Done!")

log_message("Creating a Cicero cell_data_set (CDS) from ATAC data...")
log_message("    Running dimensionality reduction")
cds <- make_atac_cds(atac_long, binarize = TRUE) %>%
  detect_genes() %>%
  estimate_size_factors() %>%
  preprocess_cds(method = "LSI") %>%
  monocle3::reduce_dimension(reduction_method = "UMAP", preprocess_method = "LSI")

umap_coords <- reducedDims(cds)$UMAP
log_message("        Done!")

log_message("    Making Cicero CDS object")
cicero_obj <- make_cicero_cds(cds, reduced_coordinates = umap_coords)
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

spark_disconnect(sc)
