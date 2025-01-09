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
gene_anno <- gene_anno[, c("seqid", "start", "end", "strand", "gene_id", "gene_name")]
gene_anno$chromosome <- paste0("chr", gene_anno$seqid)
gene_anno <- gene_anno[!duplicated(gene_anno$gene_id), ]

# =============================================
# Annotate and Calculate Gene Activities
# =============================================

log_message("Annotating CDS with gene annotations...")
cds <- annotate_cds_by_site(cds, gene_anno)

log_message("Building gene activity matrix...")
unnorm_ga <- build_gene_activity_matrix(cds, conns)
unnorm_ga <- unnorm_ga[rowSums(unnorm_ga) > 0, colSums(unnorm_ga) > 0]

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
