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

atac_file_path <- "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input/macrophage_buffer1_filtered_ATAC.csv"
output_dir <- "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output"

#atac_file_path <- args[1]
#output_dir <- args[2]

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

  
final_peak_gene <- bind_rows(
  peak_gene_associations$peak1_assoc,
  peak_gene_associations$peak2_assoc,
  peak_gene_associations$cross_assoc
  ) %>%
  distinct()

head(final_peak_gene)
write.csv(final_peak_gene, "peak_gene_associations.csv", row.names = FALSE)



log_message("Building gene activity matrix...")
head(conns)
head(gene_annotation_sub)

unnorm_ga <- build_gene_activity_matrix(cds, conns)

unnorm_ga <- unnorm_ga[!Matrix::rowSums(unnorm_ga) == 0, 
                       !Matrix::colSums(unnorm_ga) == 0]

log_message("Normalizing gene activities...")
num_genes <- pData(cds)$num_genes_expressed
names(num_genes) <- row.names(pData(cds))
cicero_gene_activities <- normalize_gene_activities(unnorm_ga, num_genes)

log_message("Saving gene activity scores...")
write.csv(cicero_gene_activities, file.path(output_dir, "cicero_gene_activity_scores.csv"), row.names = TRUE)


# =============================================
# Map Peaks to Genes
# =============================================

log_message("Validating and formatting peaks for mapping...")
validate_peaks <- function(peaks) {
  if (any(!grepl("chr[0-9XY]+-[0-9]+-[0-9]+", peaks))) {
    stop("Error: Peaks contain improperly formatted entries.")
  }
}
validate_peaks(conns$Peak1)
validate_peaks(conns$Peak2)

log_message("Creating GRanges objects for peaks and genes...")
peak_ranges <- function(peaks) {
  GRanges(
    seqnames = sapply(strsplit(peaks, "-"), `[`, 1),
    ranges = IRanges(
      start = as.numeric(sapply(strsplit(peaks, "-"), `[`, 2)),
      end = as.numeric(sapply(strsplit(peaks, "-"), `[`, 3))
    )
  )
}
peak1_ranges <- peak_ranges(conns$Peak1)
peak2_ranges <- peak_ranges(conns$Peak2)

gene_ranges <- GRanges(
  seqnames = gene_anno$chromosome,
  ranges = IRanges(start = gene_anno$start, end = gene_anno$end),
  strand = gene_anno$strand,
  gene = gene_anno$gene_name
)

log_message("Finding overlaps between peaks and genes...")
map_peaks_to_genes <- function(peak_ranges, gene_ranges) {
  overlaps <- findOverlaps(peak_ranges, gene_ranges)
  mapped_genes <- rep(NA, length(peak_ranges))
  if (length(overlaps) > 0) {
    mapped_genes[queryHits(overlaps)] <- gene_ranges$gene[subjectHits(overlaps)]
  }
  mapped_genes
}
conns$Gene1 <- map_peaks_to_genes(peak1_ranges, gene_ranges)
conns$Gene2 <- map_peaks_to_genes(peak2_ranges, gene_ranges)

log_message("Filtering and saving peak-to-gene links...")
gene_links <- conns[!is.na(conns$Gene1) | !is.na(conns$Gene2), ]
write.csv(gene_links, file.path(output_dir, "peak_to_gene_links.csv"), row.names = FALSE)

log_message("Pipeline completed successfully.")
