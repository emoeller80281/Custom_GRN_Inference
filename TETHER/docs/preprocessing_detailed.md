# Preprocessing

## Muon Filtering and QC

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

### RNA Filtering
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

The RNA and ATAC counts per cell were normalized to $1e^{4}$, log1p normalized, and scaled. 

## Pseudobulk Generation

After QC and normalization, a joint RNA-ATAC embedding was generated using [MOFA](https://pmc.ncbi.nlm.nih.gov/articles/PMC6010767/). This was used to calculate the nearest neighbors of each cell in the joint embedding space. Each cell's ATAC and RNA values were combined with its nearest neighbor's values two hops away using multi-hop diffusion. The resulting RNA and ATAC profile of each cell is therefore a weighted average of the molecular profile of cells within two steps of the cell-cell connectivity graph.

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
