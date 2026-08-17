# Project Overview

Eukaryotic cells control the activation and suppression of gene expression through complicated and condition-specific interactions between transcription factors, intermediate DNA binding proteins, and cis-regulatory elements such as enhancers and promoters. Transcription factors bind to these cis-regulatory elements to trigger transcription via the formation of the transcription pre-initiation complex [1](https://www.nature.com/articles/nrg3207). These TF to TG regulatory relationships can be represented by gene regulatory networks, where nodes represent genes and directed edges represent the regulatory relationships between them.

Gene regulatory networks (GRNs) define how cells sense, interpret, and respond to biological signals. These networks are composed of directed edges between transcription factors (TFs) and the target genes (TGs) that they regulate. Changes in patterns of TF-TG regulation can alter cellular identity, developmental programs, and disease-associated transcriptional states. Although experimental approaches such as ChIP-seq, TF knockout screens, and perturb-seq can identify regulatory relationships, these methods are costly, labor-intensive, and often specific to a particular organism, cell type, or experimental condition. As a result, comprehensive and context-specific GRN mapping remains a major challenge.

Single-cell multiomic RNA and ATAC sequencing provides a powerful opportunity to address this challenge by jointly measuring gene expression and chromatin accessibility in individual cells. However, existing computational GRN inference methods often struggle to distinguish true TF-TG regulatory relationships from indirect associations. To improve TF-TG prediction, we developed a new deep learning framework that integrates TF protein structure, DNA sequence, chromatin accessibility, peak-to-gene distance, and TF/TG expression to infer cell type- and context-specific regulatory interactions.

Our framework consists of two connected models. First, we use a modified version of the [TFBindFormer](https://www.biorxiv.org/content/10.64898/2026.04.09.717563v2) TF-DNA binding model to predict whether a TF will bind to a region of DNA. We generate rich TF structure and sequence embeddings using the [ProsT5](https://academic.oup.com/nargab/article/6/4/lqae150/7901286) protein language model encoder. The TF-DNA binding model integrates these TF embeddings with DNA sequence representations using bidirectional cross-attention to predict TF-DNA binding at open chromatin regions near target genes. Second, a TF-TG regulation model combines predicted TF-DNA binding with single-cell chromatin accessibility, TF expression, TG expression, and peak-to-gene distance across batches of cells to predict whether a TF regulates a target gene.

We trained and evaluated this framework across seven mouse and human single-cell multiomic datasets, including mouse embryonic stem cells, mouse hepatocytes, human macrophages, and K562 cells. To assess genomic generalization, models were trained and tested on held-out chromosome splits. The TF-TG model achieved strong performance across datasets, outperformed leading GRN inference methods in average AUROC, accuracy, and early precision rankings, and retained predictive ability when evaluated across samples, cell types, and organisms. These results suggest that the model learns both cell-type-specific and generalizable principles of TF-TG regulation. Overall, this framework provides a scalable approach for inferring regulatory networks from single-cell multiomic data and studying transcriptional state changes across development, disease, and cellular context.

# Outline
### Introduction
1. Gene regulatory networks
    - 
    - What are GRNs?
        - GRNs are directed networks showing the regulatory relationships between TFs and TGs, where the nodes are the genes and the edges are the regulatory relationships between them.
    - Why are they important?
        - Understanding the logical structure of GRNs allows us to make predictions about how perturbations to a gene or set of genes will affect how the cell functions.
        - The ability to accurately model GRNs will allow researchers to test hypotheses *in-silico* to narrow down candidate drug targets, rather than running large and expensive drug screens.
        - The ability to accurately model protein folding with AlphaFold2 has demonstrated the usefulness of predictive models in biology for aiding researchers in designing their experiments.
    - Why is it challenging to make them?
        - Direct effects are difficult to distinguish from indirect effects
        - Limited amount of data from individual cell types and tissues
        - Limited and poor-quality cell type-specific ground truth datasets
        - Non-linear relationships that change gene regulation
            - TFs interact with each other to regulate genes
            - TFs change their behavior between different cell types
        - Bias in which genes are studied; some have more experimental evidence than others
        - Experimental methods
            - Experimental results from one cell type and condition are not necessarily generalizable to other conditions and cell types.
            - Costly and time-consuming
        - Computational methods
            - Bulk sequencing doesn't capture cell-cell heterogeneity
            - Level of RNA expression does not directly translate to the level of protein expression
            - Sparsity causes high noise in single-cell data
        - Directed vs Undirected relationships
            - Co-expression methods don't show direction
        - What have other people tried?
            - Co-expression networks
                - Does not show which genes are regulators and which are targets
                - Does not include information about the cis-regulatory elements that TFs are using to regulate their TGs
            - Motif scanning
                - DNA surrounding motifs alters TF binding affinity
    - What would be the benefits of a tool that could produce accurate GRNs?
2. Single-cell Multiomics
    - What is it?
        - Measure both the gene expression and chromatin accessibility from the same individual cells
    - Why is it better than scRNAseq or scATACseq alone? Why does it help us with predicting GRNs?
        - Shows which regions of the chromatin are accessible around a target gene AND the gene expression
        - Allows us to model the biology of a TF binding to regulatory regions of open chromatin to control TG expression
        - scRNA-seq alone does not capture the 
    - What are its limitations?
        - Large number of cells per individual, but each cell is not an independent observation.
3. What are some other single-cell multiomic GRN inference methods?
    - 
    - What have they tried?
    - How have they fallen short?
4. What is my project?
    - How is my project unique?
    - What did we find?
    - How does this expand the field?

### Methods
1. Dataset sources
2. Preprocessing
3. Ground truth sources
4. Model architectures
    - TF-DNA binding model
    - TF-TG regulation model
5. Training data generation
6. Model performance evaluation metrics
7. Evaluating biological questions using the model
    - Identifying genes related to mouse embryogenesis

### Results
1. TF-DNA binding prediction performance
2. TF-TG regulation prediction performance
3. TF-TG model generalization between cell types
4. Performance vs other inference methods
5. Performance of the model when answering biological questions

### Conclusions
1. How does our method perform compared to other methods?
2. How does our method contribute to the field?
3. Why should other people use our method over existing methods?
4. What are the limitations of our method?


# Preprocessing

## Muon Filtering and QC

The datasets were filtered using a standard Muon filtering and QC pipeline. The QC thresholds for filtering cells, genes, and peaks were selected separately for each dataset. After filtering, a joint RNA-ATAC embedding was generated for each cell using [MOFA](https://pmc.ncbi.nlm.nih.gov/articles/PMC6010767/). The chromatin accessibility and gene expression of each cell were blended with it's closest two-hop neighbors in the joint embedding space to generate metacells, a method used by [LINGER](https://www.nature.com/articles/s41587-024-02182-7). 

For more detailed information on the preprocessing pipeline and metacell generation method, see [preprocessing_detailed.md](./preprocessing_detailed.md)

<br>
<br>

# Model Architecture

For a detailed review of the model architectures, see [model_architectures.md](./model_architectures.md)

## TF-DNA Binding Model

<p align="center">
  <img src="../plots/model_architecture/TF_DNA_model.png" width="650">
</p>

The TF-DNA binding model combines a rich sequence and structure embedding from the ProsT5 protein language model with a DNA-sequence to predict the likelihood that the TF will bind to the DNA sequence.

The training data consists of ChIP-seq TF-DNA interactions for an organism from ChIP-Atlas. A [Foldseek](https://github.com/steineggerlab/foldseek) database is generated to create 3Di protein structural information sequences for each TF. The [ProsT5](https://academic.oup.com/nargab/article/6/4/lqae150/7901286) protein language model encoder then combines the 3Di sequences with the amino acid sequences for each TF in the [AlphaFoldDB](https://alphafold.ebi.ac.uk/), generating TF embeddings that capture sequence and 3D structural information.

The labels for the TF-DNA edges are created using the binding data for ChIP-Atlas. False edges are generated by randomly shuffling the ChIP-Atlas edges, ensuring no overlap with the True edges.

Next, we created a one-hot encoding for the DNA sequence of the TF binding location (with a nucleotide order of ACGT). We ensured that the one-hot encodings are the same shape by encoding the DNA sequence 128bp upstream and 128bp downstream (set by the `flank_size` parameter) from the center of the ChIP-seq peak.

The training, validation, and test datasets were stratified by chromosome to ensure that the model never sees TF binding locations on the excluded chromosomes during training. 

| Organism | Train Chroms | Val Chroms | Test Chroms |
|:---------|:------------:|:----------:|:-----------:|
|  Human   |    1 - 17    |  18 - 19   |   20 - 22   |
|  Mouse   |    1 - 15    |  16 - 17   |   18 - 19   |

## TF-TG Regulation Model

<p align="center">
  <img src="../plots/model_architecture/TF_TG_model.png" width="650">
</p>

The TF regulation model combines information about the regulatory landscape surrounding a TG with the gene expression to predict the likelihood that a TF regulates the TG. The TF-DNA model first makes a prediction on whether the TF will bind to the ATAC peaks near the TG for a cell. The binding likelihood for each peak is combined with the peak's accessibility and distance from the TG TSS. This provides the model with context about whether the chromatin is accessible in a given cell along with relative positional information about where the TF can bind. A gene expression to peak information cross attention layer allows the model to focus on important patterns in the accessible regions of the DNA where a TF can bind around the TG to influence the model predictions. The TF binding information from the chromatin landscape around the TG is combined with the TF and TG gene expression values to predict the the probability that the TF regulates the TG.

The model is trained using a mixture of cell type-specific ground truth datasets for each cell type, including ChIP-seq, LOGOF, and ChIPX. The majority of the ground truth datasets are from ChIP-Atlas, which contains a large number of cell type-specific ChIP-seq datasets. The training, validation, and test datasets are created by stratifing the TGs and peaks by chromosome, using the same split as the TF-DNA binding model.

### Ground Truth Datasets

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



One-hot encodings of the ATAC peaks sequences were generated using the ATAC-seq pseudobulk dataset. The RNA-seq pseudobulk dataset was used to create a set of candidate TFs and TGs that overlapped with the combined ground truth dataset for the cell type. The full universe of all potential TF-TG edge combinations were generated and labeled based on whether the edge was in the ground truth. For each True edge, 10 False edges were randomly sampled to generate a labeled training dataset. Peaks associated with each TG were selected based on whether the center of the peak was located within 100 kb from the transcription start site (TSS) of the TG. TGs with no associated ATAC peaks and ATAC peaks with no associated TGs were filtered out of the dataset. 



### Training the Model

<br>
<br>



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
    <td><img src="../plots/model_vs_other_method_boxplots/early precision rate_by_method_boxplot.png" width="400"></td>
    <td><img src="../plots/model_vs_other_method_boxplots/recall_by_method_boxplot.png" width="400"></td>
  </tr>
  <tr>
    <td><img src="../plots/model_vs_other_method_boxplots/fpr_by_method_boxplot.png" width="400"></td>
    <td><img src="../plots/model_vs_other_method_boxplots/f1_by_method_boxplot.png" width="400"></td>
  </tr>
  <tr>
    <td><img src="../plots/model_vs_other_method_boxplots/perint_by_method_boxplot.png" width="400"></td>
    <td><img src="../plots/model_vs_other_method_boxplots/predictable tf fraction_by_method_boxplot.png" width="400"></td>
  </tr>
</table>

### AUROC Calculation

### AUPRC Calculation

### Early Precision Rate Calculation

$$\text{Early precision}=
\frac{\text{true edges among top }k}{k}$$

$$EPR=
\frac{\text{early precision}}
{\text{positive prevalence}}$$

### Early Recall Calculation

$$\text{Recall}
=
\frac{TP}{TP+FN}
=
\frac{|A\cap B|}{|B|}$$

### Early False Positive Rate Calculation

$$FPR=
\frac{FP}{FP+TN}
=
\frac{|A\setminus B|}
{|U\setminus B|}$$

### F1 Score Calculation

$$\text{F1} = 2\times\frac{\text{Precision}\times\text{Recall}}
{\text{Precision}+\text{Recall}}$$

### Percent Interaction Calculation

$$\text{PerInt}=
100\times
\frac{|A\cap B|}
{\min(|A|,|B|)}$$

### Predictable TF Fraction Calculation

$$\text{PTF fraction}
=
\frac{\text{predictable TFs}}
{\text{evaluable TFs}}$$

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

## References

<p style="padding-left: 2em; text-indent: -2em;">
  1. Spitz, F., Furlong, E. Transcription factors: from enhancer binding to developmental control. Nat Rev Genet 13, 613–626 (2012). https://doi.org/10.1038/nrg3207
</p>

