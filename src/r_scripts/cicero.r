library(cicero)
library(monocle3)
library(Signac)
library(Seurat)

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

gene_anno$chromosome <- paste0("chr", gene_anno$seqid)
gene_anno$gene <- gene_anno$gene_id
gene_anno$symbol <- gene_anno$gene_name

# Annotate CDS and calculate gene activities
cds <- annotate_cds_by_site(cds, gene_anno)
unnorm_ga <- build_gene_activity_matrix(cds, conns)
cicero_gene_activities <- normalize_gene_activities(unnorm_ga, num_genes = pData(cds)$num_genes_expressed)

# Save results
write.csv(conns, paste0(output_dir, "/cicero_genome_wide_connections.csv"), row.names = FALSE)
write.csv(cicero_gene_activities, paste0(output_dir, "/cicero_gene_activity_scores.csv"), row.names = TRUE)
