library(cicero)
library(monocle3)
library(Signac)
library(Seurat)
library(GenomicRanges)

# Get command-line arguments
args <- commandArgs(trailingOnly = TRUE)

# Get the path to the rds file containing the filtered multiomic data
rds_file_path <- args[1]
output_dir <- args[2]

# Load data and preprocess
multiomic_data <- readRDS(rds_file_path)

expression_matrix <- as.matrix(GetAssayData(multiomic_data, assay = "peaks", layer = "counts"))
cell_metadata <- as.data.frame(multiomic_data@meta.data)
rownames(cell_metadata) <- colnames(expression_matrix)

gene_metadata <- data.frame(
  peak_id = rownames(expression_matrix),
  gene_short_name = rownames(expression_matrix),
  row.names = rownames(expression_matrix)
)

cds <- new_cell_data_set(expression_data = expression_matrix,
                         cell_metadata = cell_metadata,
                         gene_metadata = gene_metadata)

cds <- preprocess_cds(cds, method = "LSI")
cds <- reduce_dimension(cds, reduction_method = 'UMAP', preprocess_method = "LSI")

umap_coords <- reducedDims(cds)$UMAP
cicero_obj <- make_cicero_cds(cds, reduced_coordinates = umap_coords)

# Use entire genome
data("human.hg19.genome")
sample_genome <- human.hg19.genome

# Run Cicero
conns <- run_cicero(cicero_obj, sample_genome, sample_num = 500, silent = FALSE)

# Download and process gene annotations
temp <- tempfile()
download.file("https://ftp.ensembl.org/pub/release-113/gtf/homo_sapiens/Homo_sapiens.GRCh38.113.gtf.gz", temp)
gene_anno <- rtracklayer::readGFF(temp)
unlink(temp)

# Ensure required columns are present
gene_anno <- gene_anno[, c("seqid", "start", "end", "strand", "gene_id", "transcript_id", "gene_name")]

# Add "chr" prefix to chromosome names
gene_anno$chromosome <- paste0("chr", gene_anno$seqid)

# Rename columns for consistency
gene_anno$gene <- gene_anno$gene_id
gene_anno$transcript <- gene_anno$transcript_id
gene_anno$symbol <- gene_anno$gene_name

# Ensure strand column is atomic
gene_anno$strand <- as.character(gene_anno$strand)

# Filter by strand
pos <- subset(gene_anno, strand == "+")
neg <- subset(gene_anno, strand == "-")

# Process positive strand genes
pos <- pos[order(pos$start),]  # Order by start position
pos <- pos[!duplicated(pos$transcript),]  # Remove duplicate transcripts
pos$end <- pos$start + 1  # Mark TSS as 1-bp regions

# Process negative strand genes
neg <- neg[order(neg$start, decreasing = TRUE),]  # Order by start position
neg <- neg[!duplicated(neg$transcript),]  # Remove duplicate transcripts
neg$start <- neg$end - 1  # Mark TSS as 1-bp regions

# Combine processed data
gene_annotation_sub <- rbind(pos, neg)

# Subset to required columns
gene_annotation_sub <- gene_annotation_sub[, c("chromosome", "start", "end", "symbol")]

# Rename the gene symbol column
names(gene_annotation_sub)[4] <- "gene"

# Set the output directory
output_dir <- "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output"

# Annotate CDS and calculate gene activities
cat("Annotating CDS with gene annotations...\n")
cds <- annotate_cds_by_site(cds, gene_anno)

cat("Building gene activity matrix...\n")
unnorm_ga <- build_gene_activity_matrix(cds, conns)

# Remove any rows/columns with all zeroes
unnorm_ga <- unnorm_ga[!Matrix::rowSums(unnorm_ga) == 0, 
                       !Matrix::colSums(unnorm_ga) == 0]

cat("Detecting genes in CDS...\n")
cds <- detect_genes(cds)

# Create a list of the number of genes expressed
cat("Calculating number of genes expressed...\n")
num_genes <- pData(cds)$num_genes_expressed
names(num_genes) <- row.names(pData(cds))

cat("Normalizing gene activities...\n")
cicero_gene_activities <- normalize_gene_activities(unnorm_ga, num_genes)

# Save results
cat("Saving Cicero results...\n")
write.csv(conns, paste0(output_dir, "/cicero_genome_wide_connections.csv"), row.names = FALSE)
write.csv(cicero_gene_activities, paste0(output_dir, "/cicero_gene_activity_scores.csv"), row.names = TRUE)

# Convert gene_anno to a GRanges object
cat("Converting gene annotations to GRanges...\n")
gene_ranges <- GRanges(
  seqnames = gene_anno$chromosome,
  ranges = IRanges(start = gene_anno$start, end = gene_anno$end),
  strand = gene_anno$strand,
  gene = gene_anno$gene,
  gene_name = gene_anno$gene_name
)

# Ensure Peak1 and Peak2 are characters
conns$Peak1 <- as.character(conns$Peak1)
conns$Peak2 <- as.character(conns$Peak2)

# Validate the format of Peak1 and Peak2
if (any(!grepl("chr[0-9XY]+-[0-9]+-[0-9]+", conns$Peak1))) {
  stop("Error: Peak1 contains improperly formatted entries.")
}
if (any(!grepl("chr[0-9XY]+-[0-9]+-[0-9]+", conns$Peak2))) {
  stop("Error: Peak2 contains improperly formatted entries.")
}

# Create GRanges for Peak1 and Peak2
cat("Converting peaks to GRanges...\n")
peak1_ranges <- GRanges(
  seqnames = sapply(strsplit(conns$Peak1, "-"), `[`, 1),
  ranges = IRanges(
    start = as.numeric(sapply(strsplit(conns$Peak1, "-"), `[`, 2)),
    end = as.numeric(sapply(strsplit(conns$Peak1, "-"), `[`, 3))
  )
)

peak2_ranges <- GRanges(
  seqnames = sapply(strsplit(conns$Peak2, "-"), `[`, 1),
  ranges = IRanges(
    start = as.numeric(sapply(strsplit(conns$Peak2, "-"), `[`, 2)),
    end = as.numeric(sapply(strsplit(conns$Peak2, "-"), `[`, 3))
  )
)

# Find overlaps for Peak1 and Peak2 with gene annotations
cat("Finding overlaps between peaks and genes...\n")
peak1_gene_hits <- findOverlaps(peak1_ranges, gene_ranges)
peak2_gene_hits <- findOverlaps(peak2_ranges, gene_ranges)

# Map Peak1 and Peak2 to genes
cat("Mapping peaks to genes...\n")
conns$Gene1 <- NA
conns$Gene2 <- NA

if (length(peak1_gene_hits) > 0) {
  conns$Gene1[queryHits(peak1_gene_hits)] <- gene_ranges$gene[subjectHits(peak1_gene_hits)]
}

if (length(peak2_gene_hits) > 0) {
  conns$Gene2[queryHits(peak2_gene_hits)] <- gene_ranges$gene[subjectHits(peak2_gene_hits)]
}

# Filter connections with at least one gene linked
cat("Filtering peak-to-gene links...\n")
gene_links <- conns[!is.na(conns$Gene1) | !is.na(conns$Gene2), ]

# Save results
cat("Saving results...\n")
output_dir <- "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/output"
write.csv(gene_links, file = paste0(output_dir, "/peak_to_gene_links.csv"), row.names = FALSE)
write.csv(conns, paste0(output_dir, "/cicero_genome_wide_connections.csv"), row.names = FALSE)

cat("Pipeline completed successfully.\n")
