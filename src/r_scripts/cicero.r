library(cicero)
library(monocle3)
library(Signac)
library(Seurat)
library(GenomicRanges)
library(Matrix)
library(rtracklayer)

# =============================================
# Functions
# =============================================

log_message <- function(message) {
  cat(sprintf("[%s] %s\n", Sys.time(), message))
}

# =============================================
# Command-Line Arguments
# =============================================

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("Usage: Rscript script.R <atac_file_path> <output_dir>")
}

atac_file_path <- args[1]
output_dir <- args[2]

# Ensure the output directory exists
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

log_message("Starting Cicero pipeline...")
log_message(sprintf("ATAC data file: %s", atac_file_path))
log_message(sprintf("Output directory: %s", output_dir))

# =============================================
# Load Data and Preprocess
# =============================================

# Load the ATAC CSV file
log_message("Reading input ATAC CSV file...\n")
atac_data <- read.csv(atac_file_path, row.names = 1, check.names = FALSE)

# Reshape the data into long format
log_message("Reshaping data to long format...\n")
atac_long <- reshape2::melt(as.matrix(atac_data), varnames = c("peak_position", "cell"), value.name = "reads")

# Filter out rows with zero reads
log_message("Filtering out zero reads...\n")
atac_long <- atac_long[atac_long$reads > 0, ]

# Convert "chr<num>:<start>-<end>" format to "chr<num>_<start>_<end>"
log_message("Converting peak positions to required format...\n")
atac_long$peak_position <- gsub("[:\\-]", "_", atac_long$peak_position)

cds <- make_atac_cds(atac_long, binarize = TRUE)

cds <- detect_genes(cds)
cds <- estimate_size_factors(cds)
cds <- preprocess_cds(cds, method = "LSI")
cds <- reduce_dimension(cds, reduction_method = 'UMAP', preprocess_method = "LSI")

umap_coords <- reducedDims(cds)$UMAP
cicero_obj <- make_cicero_cds(cds, reduced_coordinates = umap_coords)

# =============================================
# Cicero Analysis
# =============================================

log_message("Loading human genome data...")
data("human.hg19.genome")
sample_genome <- human.hg19.genome

log_message("Running Cicero for genome-wide connections...")
conns <- run_cicero(cicero_obj, sample_genome, sample_num = 500, silent = FALSE)

# =============================================
# Gene Annotations
# =============================================

log_message("Downloading and processing gene annotations...")
temp <- tempfile()
download.file("https://ftp.ensembl.org/pub/release-113/gtf/homo_sapiens/Homo_sapiens.GRCh38.113.gtf.gz", temp)
gene_anno <- rtracklayer::readGFF(temp)
unlink(temp)

log_message("Filtering and formatting gene annotations...")
gene_anno <- gene_anno[, c("seqid", "start", "end", "strand", "gene_id", "gene_name", "transcript_id")]
gene_anno$chromosome <- paste0("chr", gene_anno$seqid)
gene_anno$gene <- gene_anno$gene_id
gene_anno$transcript <- gene_anno$transcript_id
gene_anno$symbol <- gene_anno$gene_name
head(gene_anno)

# =============================================
# Annotate and Calculate Gene Activities
# =============================================
pos <- subset(gene_anno, strand == "+")
pos <- pos[order(pos$start),] 
# remove all but the first exons per transcript
pos <- pos[!duplicated(pos$gene_id),] 
# make a 1 base pair marker of the TSS
pos$end <- pos$start + 1 

neg <- subset(gene_anno, strand == "-")
neg <- neg[order(neg$start, decreasing = TRUE),] 
# remove all but the first exons per transcript
neg <- neg[!duplicated(neg$gene_id),] 
neg$start <- neg$end - 1
gene_annotation_sub <- rbind(pos, neg)

# Make a subset of the TSS annotation columns containing just the coordinates 
# and the gene name
gene_annotation_sub <- gene_annotation_sub[,c("chromosome", "start", "end", "symbol")]
head(gene_annotation_sub)
# Rename the gene symbol column to "gene"
names(gene_annotation_sub)[4] <- "gene"

log_message("Annotating CDS with gene annotations...")
colnames(gene_annotation_sub)

cds <- annotate_cds_by_site(cds, gene_annotation_sub)

tail(fData(cds))
tail(conns)

peak_to_gene <- fData(cds)[,c("site_name", "gene")]
peak_to_gene <- subset(peak_to_gene, gene != "NA")
peak_to_gene <- as.data.frame(peak_to_gene)
head(peak_to_gene)
head(conns)

library(dplyr)
library(tidyr)

conns_gene <- conns %>%
  left_join(peak_to_gene, by = c("Peak1" = "site_name")) %>%
  rename(gene1 = gene)
conns_gene <- conns_gene %>%
  left_join(peak_to_gene, by = c("Peak2" = "site_name")) %>%
  rename(gene2 = gene)

# Prepare Peak1 associations
peak1_assoc <- conns_gene %>%
  filter(!is.na(gene1)) %>%
  transmute(
    peak = Peak1,
    gene = gene1,
    score = 1
  )

# Prepare Peak2 associations
peak2_assoc <- conns_gene %>%
  filter(!is.na(gene2)) %>%
  transmute(
    peak = Peak2,
    gene = gene2,
    score = 1
  )

# Prepare Coaccessibility associations
cross_assoc <- conns_gene %>%
  filter(!is.na(gene1) & !is.na(gene2)) %>%
  transmute(
    peak = ifelse(is.na(gene1), Peak1, Peak2),
    gene = ifelse(is.na(gene1), gene2, gene1),
    score = coaccess
  )

# Combine all associations
final_peak_gene <- bind_rows(peak1_assoc, peak2_assoc, cross_assoc) %>%
  distinct()

final_peak_gene <- final_peak_gene %>%
  mutate(score = ifelse(is.na(gene), -1, score))

final_peak_gene$score <- (final_peak_gene$score + 1) / 2

write.csv(final_peak_gene, file.path(output_dir, "peak_gene_associations.csv"), row.names = FALSE)