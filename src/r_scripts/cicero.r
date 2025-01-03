library(cicero)
library(monocle3)
library(Seurat)
library(SingleCellExperiment)
library(Signac)
library(ggplot2)
library(patchwork)

data_dir <- '/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/input'

multiomic_data <- readRDS(paste0(data_dir, '/Macrophase_buffer1_filtered.rds'))

# Extract expression matrix for peaks
expression_matrix <- as.matrix(GetAssayData(multiomic_data, assay = "peaks", slot = "counts"))

# Extract cell metadata
cell_metadata <- as.data.frame(multiomic_data@meta.data)
# Ensure rownames of metadata match column names of the expression matrix
rownames(cell_metadata) <- colnames(expression_matrix)

# Extract gene metadata (feature/peak information)
gene_metadata <- data.frame(
  peak_id = rownames(expression_matrix), # Row names of the expression matrix are peaks
  gene_short_name = rownames(expression_matrix),
  row.names = rownames(expression_matrix)
)

cds <- new_cell_data_set(
  expression_data = expression_matrix,
  cell_metadata = cell_metadata,
  gene_metadata = gene_metadata
)

set.seed(2017)
cds <- detect_genes(cds)
cds <- estimate_size_factors(cds)
cds <- preprocess_cds(cds, method = "LSI")
cds <- reduce_dimension(cds, reduction_method = 'UMAP',
                        preprocess_method = "LSI")
cds <- cluster_cells(cds)
cds <- learn_graph(cds)
cds <- order_cells(cds)

plot_cells(cds)

umap_coords <- reducedDims(cds)$UMAP

cicero_obj <- make_cicero_cds(cds, reduced_coordinates = umap_coords)

data("human.hg19.genome")

sample_genome <- subset(human.hg19.genome, V1 == "chr2")
sample_genome$V2[1] <- 1000000

conns <- run_cicero(cicero_obj, sample_genome, sample_num = 2)
head(conns)

options(timeout = 600)

# download and unzip
temp <- tempfile()
download.file("https://ftp.ensembl.org/pub/release-113/gtf/homo_sapiens/Homo_sapiens.GRCh38.113.gtf.gz", temp)
gene_anno <- rtracklayer::readGFF(temp)
unlink(temp)

# rename some columns to match requirements
gene_anno$chromosome <- paste0("chr", gene_anno$seqid)
gene_anno$gene <- gene_anno$gene_id
gene_anno$transcript <- gene_anno$transcript_id
gene_anno$symbol <- gene_anno$gene_name

plot_connections(conns, "chr2", 548598, 1048598,
                 gene_model = gene_anno, 
                 coaccess_cutoff = 0.15, 
                 connection_width = .5, 
                 collapseTranscripts = "longest" )

pos <- subset(gene_anno, strand == "+")
pos <- pos[order(pos$start),] 
# remove all but the first exons per transcript
pos <- pos[!duplicated(pos$transcript),] 
# make a 1 base pair marker of the TSS
pos$end <- pos$start + 1 

neg <- subset(gene_anno, strand == "-")
neg <- neg[order(neg$start, decreasing = TRUE),] 
# remove all but the first exons per transcript
neg <- neg[!duplicated(neg$transcript),] 
neg$start <- neg$end - 1

gene_annotation_sub <- rbind(pos, neg)

# Make a subset of the TSS annotation columns containing just the coordinates 
# and the gene name
gene_annotation_sub <- gene_annotation_sub[,c("chromosome", "start", "end", "symbol")]

# Rename the gene symbol column to "gene"
names(gene_annotation_sub)[4] <- "gene"

cds <- annotate_cds_by_site(cds, gene_annotation_sub)

#### Generate gene activity scores ####
# generate unnormalized gene activity matrix
unnorm_ga <- build_gene_activity_matrix(cds, conns)

# remove any rows/columns with all zeroes
unnorm_ga <- unnorm_ga[!Matrix::rowSums(unnorm_ga) == 0, 
                       !Matrix::colSums(unnorm_ga) == 0]

# make a list of num_genes_expressed
num_genes <- pData(cds)$num_genes_expressed
names(num_genes) <- row.names(pData(cds))

# normalize
cicero_gene_activities <- normalize_gene_activities(unnorm_ga, num_genes)

