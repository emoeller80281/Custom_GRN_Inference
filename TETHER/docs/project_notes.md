# Preprocessing

### Muon Filtering and QC

The table below specifies the thresholds used for filtering each dataset.

| Sample | Min Cells per Gene | Min Genes per Cell | Max Genes per Cell | Min Total Counts | Max Total Counts | Max Pct MT | Min Cells per Peak | Min Peaks per Cell | Max Peaks per Cell | Min Total Peak Counts | Max Total Peak Counts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mESC 1 | 20 | 1500 | 6000 | 1000 | 25000 | 20 | 1 | 500 | 25000 | 1000 | 60000 |
| mESC 2 | 20 | 1500 | 6000 | 1000 | 25000 | 20 | 1 | 500 | 25000 | 1000 | 60000 |
| Macrophage 1 | 3 | 1500 | 15000 | 1000 | 60000 | 35 | 1 | 100 | 18000 | 1000 | 100000 |
| Macrophage 2 | 3 | 1500 | 7000 | 1000 | 60000 | 35 | 1 | 100 | 30000 | 1000 | 100000 |
| K562 | 3 | 1000 | 10000 | 500 | 50000 | 20 | 1 | 100 | 20000 | 1000 | 50000 |
| Mouse Hepatocytes 1 | 20 | 1000 | 6500 | 1000 | 50000 | 20 | 1 | 500 | 25000 | 1000 | 100000 |
| Mouse Hepatocytes 3 | 20 | 1000 | 6500 | 1000 | 50000 | 20 | 1 | 500 | 25000 | 1000 | 100000 |

These thresholds were used to filter the RNA and ATAC:

#### RNA Filtering
```python
mu.pp.filter_var(rna, 'n_cells_by_counts', lambda x: (x >= MIN_CELLS_PER_GENE))
mu.pp.filter_obs(rna, 'n_genes_by_counts', lambda x: (x >= MIN_GENES_PER_CELL) & (x <= MAX_GENES_PER_CELL))
mu.pp.filter_obs(rna, 'total_counts', lambda x: (x >= MIN_TOTAL_COUNTS) & (x <= MAX_TOTAL_COUNTS))
mu.pp.filter_obs(rna, 'pct_counts_mt', lambda x: x <= MAX_PCT_COUNTS_MT)
```

### ATAC Filtering
```python
mu.pp.filter_var(atac, 'n_cells_by_counts', lambda x: x >= MIN_CELLS_PER_PEAK)
mu.pp.filter_obs(atac, 'n_genes_by_counts', lambda x: (x >= MIN_PEAKS_PER_CELL) & (x <= MAX_PEAKS_PER_CELL))
mu.pp.filter_obs(atac, 'total_counts', lambda x: (x >= MIN_TOTAL_PEAK_COUNTS) & (x <= MAX_TOTAL_PEAK_COUNTS))
```

The RNA and ATAC counts per cell were normalized to $1e^{4}$, log1p normalized, and scaled. After QC and normalization, a joint RNA-ATAC embedding was generated using [MOFA](https://pmc.ncbi.nlm.nih.gov/articles/PMC6010767/). This was used to calculate the nearest neighbors of each cell in the joint embedding space. Each cell's ATAC and RNA values were combined with its nearest neighbor's values two hops away using multi-hop diffusion. The resulting RNA and ATAC profile of each cell is therefore a weighted average of the molecular profile of cells within two steps of the cell-cell connectivity graph.

This reduces the effect of technical dropout in the datasets and homogenizes the cell gene expression. Technical dropout introduces noise and sparsity into the dataset. This can hurt the ability for a predictive model to learn the effects of a low gene expression or peak accessibility value if the value is not consistently measured. Aggregating data from cells with similar expression and accessibility profiles to create metacells smooths the effects of dropout events.

Let

$$C \in \mathbb{R}^{N \times N}$$

Represent the square connectivity matrix for the cells in the dataset, where $C_{ij}$ represents the connectivity between cells $i$ and $j$.

We add self-loops to ensure that each cell contributes to its own profile in addition to those of its neighbors.

$$\tilde{W} = \mathcal{N}\left(C + I \right)$$

where $I$ is the $N \times N$ identity matrix. 

We row normalize the connectivity matrix to ensure that each each row adds up to 1, so each row of $W$ defines a weighted distribution over the cell and its neighbors. This is done so the weights of the neighbors are always a weight less than 1, but the total weight from all neighbors plus the cell itself adds up to one.

We use a diffusion method to calculate the two-hop weighted diffusion between the cell and its neighbor-of-neighbor cells. 

Where the total weight between cell $i$ and $j$ with $k$ as the intermediate step ($i \rightarrow k \rightarrow j$) is defined as

$$W^{(2)}_{ij} = \sum^{N}_{k=1}W_{ik}W_{kj}$$

Let the RNA expression matrix be

$$X_{\text{RNA}} \in \mathbb{R}^{N \times G}$$

And the ATAC accessibility matrix be

$$X_{\text{ATAC}} \in \mathbb{R}^{N \times P}$$

Both matrices are multiplied by the two-hop connectivity weight matrix $W^{(2)}$

$$X^{\text{soft}}_{\text{RNA}} = W^{(2)}X_{\text{RNA}}$$
$$X^{\text{soft}}_{\text{ATAC}} = W^{(2)}X_{\text{ATAC}}$$

To get the final pseudobulk profiles for each cell. This gently blends RNA expression and ATAC accessibility between the nearest neighbors, helping to mask sparsity due to dropout events during sequencing. Importantly, this method does not reduce the number of cells by aggregating multiple cells, but rather smooths out each cell's expression with that of its neighbors.









<br>
<br>

# Model Architecture

## TF-DNA Binding Model

<p align="center">
  <img src="../plots/model_architecture/TF_DNA_model.png" width="650">
</p>




## TF-TG Regulation Model

<p align="center">
  <img src="../plots/model_architecture/TF_TG_model.png" width="650">
</p>

<br>
<br>

# Running Model Training

## Step 1: Generating TF Embeddings

The training data consists of all TF-DNA interactions for an organism, downloaded from ChIP-Atlas. Once the TF-DNA interactions are downloaded, the TF amino acid sequence FASTA file is downloaded for each TF. Next, a [Foldseek](https://github.com/steineggerlab/foldseek) database is generated and used to create 3Di protein structural information sequences using the [ProsT5](https://academic.oup.com/nargab/article/6/4/lqae150/7901286) protein language model for each TF in the [AlphaFoldDB](https://alphafold.ebi.ac.uk/) using these FASTA sequences.

The TF amino acid sequence and 3Di structural sequences are passed into the pretrained ProsT5 T5Tokenizer and T5EncoderModel to generate TF embeddings that contain both sequence and 3D structural information. 

## Step 2: Training the TF-DNA Binding Model

### Building the Training Data
The labels for the TF-DNA edges are created using the binding data for ChIP-Atlas. False edges are generated by randomly shuffling the ChIP-Atlas edges, ensuring no overlap with the True edges.

Next, we created a one-hot encoding for the DNA sequence of the TF binding location (with a nucleotide order of ACGT). We ensured that the one-hot encodings are the same shape by encoding the DNA sequence 128bp upstream and 128bp downstream (set by the `flank_size` parameter) from the center of the ChIP-seq peak.

The training, validation, and test datasets were stratified by chromosome to ensure that the model never sees TF binding locations on the excluded chromosomes during training. 

| Organism | Train Chroms | Val Chroms | Test Chroms |
|:---------|:------------:|:----------:|:-----------:|
|  Human   |    1 - 17    |  18 - 19   |   20 - 22   |
|  Mouse   |    1 - 15    |  16 - 17   |   18 - 19   |

### Training the Model

## Step 3: Training the TF-TG Regulation Model

### Building the Training Data

One-hot encodings of the ATAC peak sequences are generated using the ATAC-seq pseudobulk dataset. The RNA-seq pseudobulk dataset is used to create a set of candidate TFs and TGs, using TFs and TGs that overlap with the combined ground truth dataset for the cell type. The full universe of potential TF-TG edge combinations are generated, then labeled based on whether the edge is in the ground truth dataset. The ratio of True:False edges is randomly sampled to a maximum of 1:10 to reduce the sparsity of True edges in the dataset. 

The training, validation, and test datasets are created by stratifing the TGs and peaks by chromosome, using the same split as the TF-DNA binding model.

### Training the Model

<br>
<br>

# Ground Truth Datasets

| System      | Dataset Name                  | PMID        | TFs | TGs    | Edges      |
|-------------|-------------------------------|-------------|-----|--------|------------|
| mESC        | ChIP-Atlas                    | 38749504    | 131 | 24,821 | 7,734,466  |
| mESC        | RN111 – ChIP-seq              | 31907445    | 247 | 25,700 | 977,841    |
| mESC        | RN112 – LOGOF                 | 31907445    | 55  | 16,522 | 104,201    |
| mESC        | RN114 – ChIPX                 | 23794736    | 47  | 21,240 | 107,980    |
| mESC        | RN116 – ChIPX and LOGOF       | 23794736    | 21  | 4,542  | 8,170      |
| K562        | ChIP-Atlas                    | 38749504    | 565 | 40,153 | 17,417,550 |
| K562        | RN117 – ChIP-seq              | 37486787    | 150 | 27,761 | 1,435,720  |
| Macrophage  | ChIP-Atlas                    | 38749504    | 24  | 37,851 | 2,500,303  |
| Hepatocytes | ChIP-Atlas                    | 38749504    | 24  | 22,142 | 134,599    |

A combined ground truth set was created for each cell type using the intersection of TFs and TGs from its individual ground truth datasets.

The ratio of True to False edges was set as a maximum of **1:10 True:False edges**, where a True edge is defined as a TF $\rightarrow$ TG edge from the combined ground truth dataset. 

<br>
<br>

# Inferred GRN Sizes by Method

The number of unique TFs, TGs, and edges per GRN inference method (before restricting to only ground truth edges)

<table>
  <tr>
    <td><img src="../plots/grn_sizes_by_method/num_tfs_by_method_boxplot_auprc.png" width="350"></td>
    <td><img src="../plots/grn_sizes_by_method/num_tgs_by_method_boxplot_auprc.png" width="350"></td>
    <td><img src="../plots/grn_sizes_by_method/num_edges_by_method_boxplot_auprc.png" width="350"></td>
  </tr>
</table>

<p align="center">
  <img src="../plots/grn_sizes_by_method/num_edges_jitter_auprc.png" width="650">
</p>

<br>
<br>

# Performance Metrics vs Other GRN Inference Methods

The preprocessed datasets were used as the input for each GRN inference method. GRNs were generated using each method, then evaluated against each sample's combined ground truth datasets.

Each boxplot represents the performance of each method across all samples.

<table>
  <tr>
    <td><img src="../plots/model_vs_other_method_boxplots/roc_by_method_boxplot.png" width="400"></td>
    <td><img src="../plots/model_vs_other_method_boxplots/prc_by_method_boxplot.png" width="400"></td>
  </tr>
  <tr>
    <td><img src="../plots/model_vs_other_method_boxplots/f1_by_method_boxplot.png" width="400"></td>
    <td><img src="../plots/model_vs_other_method_boxplots/early precision rate_by_method_boxplot.png" width="400"></td>
  </tr>
  <tr>
    <td><img src="../plots/model_vs_other_method_boxplots/fpr_by_method_boxplot.png" width="400"></td>
    <td><img src="../plots/model_vs_other_method_boxplots/predictable tf fraction_by_method_boxplot.png" width="400"></td>
  </tr>
</table>

<br>
<br>

# Stability

We assessed how well each method designates the same edges as important given different training datasets.
We first created a 1000 cell subset of each sample. From this, we created 10x subsamples of 700 cells and generated
GRNs for each subsample for each method. For our method, we trained new models on each of the 10 subsamples. The pairwise
Jaccard Index of the top 10% of edges were calculated between each of the subsamples. Higher Jaccard Index values 
mean that the method consistently ranks the same edges as important when given different cell expression and accessibility
values as input.

<img src="../plots/stability/stability_by_method_boxplot.png" width="450">

<img src="../plots/stability/stability_by_method_heatmap.png" width="600">

<br>
<br>

# Scalability

The resources required to generate a GRN using each GRN inference method were recorded for each sample. Our method scales with the number of edges, with an average speed of **329 edges per second**.

<table>
  <tr>
    <td><img src="../plots/scalability/wall_time_boxplot.png" width="350"></td>
    <td><img src="../plots/scalability/user_time_boxplot.png"width="350"></td>
    <td><img src="../plots/scalability/max_ram_gb_boxplot.png"width="350"></td>
  </tr>
</table>

> NOTE: Our method uses one NVIDIA V100 GPU for model inference, while the other methods use only CPUs. 