# Preprocessing

The samples were processed independently using Muon.

| Sample | Min Cells per Gene | Min Genes per Cell | Max Genes per Cell | Min Total Counts | Max Total Counts | Max Pct MT | Min Cells per Peak | Min Peaks per Cell | Max Peaks per Cell | Min Total Peak Counts | Max Total Peak Counts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mESC 1 | 20 | 1500 | 6000 | 1000 | 25000 | 20 | 1 | 500 | 25000 | 1000 | 60000 |
| mESC 2 | 20 | 1500 | 6000 | 1000 | 25000 | 20 | 1 | 500 | 25000 | 1000 | 60000 |
| Macrophage 1 | 3 | 1500 | 15000 | 1000 | 60000 | 35 | 1 | 100 | 18000 | 1000 | 100000 |
| Macrophage 2 | 3 | 1500 | 7000 | 1000 | 60000 | 35 | 1 | 100 | 30000 | 1000 | 100000 |
| K562 | 3 | 1000 | 10000 | 500 | 50000 | 20 | 1 | 100 | 20000 | 1000 | 50000 |
| Mouse Hepatocytes 1 | 20 | 1000 | 6500 | 1000 | 50000 | 20 | 1 | 500 | 25000 | 1000 | 100000 |
| Mouse Hepatocytes 3 | 20 | 1000 | 6500 | 1000 | 50000 | 20 | 1 | 500 | 25000 | 1000 | 100000 |

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