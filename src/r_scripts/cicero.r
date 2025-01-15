# Load required libraries
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

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

log_message("Starting Cicero pipeline...")
log_message(sprintf("ATAC data file: %s", atac_file_path))
log_message(sprintf("Output directory: %s", output_dir))

# =============================================
# Load and Preprocess Data
# =============================================

log_message("Loading and preprocessing ATAC data...")
atac_data <- read.csv(atac_file_path, row.names = 1, check.names = FALSE)

atac_long <- reshape2::melt(as.matrix(atac_data), varnames = c("peak_position", "cell"), value.name = "reads") %>%
  filter(reads > 0) %>%
  mutate(peak_position = gsub("[:\\-]", "_", peak_position))

cds <- make_atac_cds(atac_long, binarize = TRUE) %>%
  detect_genes() %>%
  estimate_size_factors() %>%
  preprocess_cds(method = "LSI") %>%
  reduce_dimension(reduction_method = "UMAP", preprocess_method = "LSI")

umap_coords <- reducedDims(cds)$UMAP
cicero_obj <- make_cicero_cds(cds, reduced_coordinates = umap_coords)

# =============================================
# Run Cicero
# =============================================

log_message("Running Cicero...")
conns <- run_cicero(cicero_obj, chrom_sizes, sample_num = 500, silent = FALSE)

# =============================================
# Process Gene Annotations
# =============================================

log_message("Processing gene annotations...")
gene_anno <- rtracklayer::import(gene_annot) %>% 
  as.data.frame() %>%
  select(seqid, start, end, strand, gene_id, gene_name, transcript_id) %>%
  mutate(
    chromosome = paste0("chr", seqid),
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
peak_to_gene <- fData(cds) %>%
  select(site_name, gene) %>%
  filter(!is.na(gene)) %>%
  as.data.frame()

conns_gene <- conns %>%
  left_join(peak_to_gene, by = c("Peak1" = "site_name")) %>%
  rename(gene1 = gene) %>%
  left_join(peak_to_gene, by = c("Peak2" = "site_name")) %>%
  rename(gene2 = gene)

final_peak_gene <- bind_rows(
  conns_gene %>% filter(!is.na(gene1)) %>% transmute(peak = Peak1, gene = gene1, score = 1),
  conns_gene %>% filter(!is.na(gene2)) %>% transmute(peak = Peak2, gene = gene2, score = 1),
  conns_gene %>% filter(!is.na(gene1) & !is.na(gene2)) %>%
    transmute(peak = Peak1, gene = gene1, score = coaccess)
) %>%
  distinct() %>%
  mutate(score = ifelse(is.na(gene), -1, score), score = (score + 1) / 2)

write.csv(final_peak_gene, file.path(output_dir, "peak_gene_associations.csv"), row.names = FALSE)

log_message("Cicero pipeline completed.")
