# Dataset selection criteria

This guide defines how to appraise a new single-cell multiome dataset before it enters the TF-TG
pipeline. The goal is to add data without adding confounding variables. Each criterion comes from
how the models use the data or from a failure that this project has already measured.

Measured values are from 2026-10-30 unless a section gives another date. They cover the 17 samples
in `TETHER/bash_scripts/03c_build_celltype_data.sh`.

## Terms

- **Slice:** one cell type in one sample, with its own ground truth file.
- **Bag:** the random cells that the model scores for one edge. The holdout models use 25 cells
  (`--max_cells_per_pair 25`).
- **Tier:** the `proxy_tier` of a ChIP source in `<dataset>_cell_type_map.tsv`. Tier 1 is the same
  cell type. Tier 2 is the same lineage. Tier 3 is a distant derivative, a cell line, a tumor or
  another organ.
- **Detected TF:** a TF with a nonzero RNA value in at least 5% of the cells of the slice.
- **Scoreable slice:** a slice that passes every gate in [Stage C](#stage-c-slice-gates).

## Rules for every appraisal

1. Finish the appraisal before you run any model on the dataset. Model performance is never a
   selection criterion, because selecting on it makes the comparison with other methods invalid.
2. Do the stages in order. Stages A and B need no download. They reject most unsuitable datasets
   at no compute cost.
3. Do not change a threshold for one dataset. If a threshold changes, apply the new value to all
   datasets and record the change.
4. Save a record for each appraised dataset, including rejected ones (see
   [Appraisal record](#appraisal-record)).

### Verdicts

| Verdict | Condition | Use |
|---|---|---|
| Accept | All gates pass | Training and evaluation |
| Restricted | All gates pass except one condition marked "Restricted" | As given in the gate |
| Reject | Any other gate fails | None |

### Threshold sources

Each threshold has one of three sources. No threshold was tuned against model performance.

- **Code:** the pipeline enforces this value.
- **Measured:** a recorded failure in this project sets this value.
- **Judgment:** a proposed value, set with the current samples as a reference. Change it only under
  rule 3.

## Stage A: Study gates (no download)

| ID | Gate | Pass condition | If it fails | Why |
|---|---|---|---|---|
| A1 | Paired assay | RNA and ATAC come from the same cell, with the same barcode | Reject | The model joins RNA and ATAC per cell for each bag. Computationally integrated data has no true pairing. |
| A2 | Assay type | 10x Multiome | Restricted: independent test only, not training | Assay changes depth and peak shape (see [Reference values](#reference-values)). Both SHARE-seq samples fail the depth gates. |
| A3 | Organism and reference | Mouse, with counts on mm10 or with FASTQ files to realign | Reject | The TF-DNA model scores mm10 sequence, and the ground truth uses mm10 TSS positions. |
| A4 | Raw counts | Per-cell raw counts for both modalities, with peak coordinates. A fragment file is preferred. | Reject | Normalized, scaled or corrected matrices cannot give the common input scale (C-gates below). |
| A5 | Biological state | Unperturbed cells: no knockout, drug, disease or stimulation | Reject | Perturbation changes regulation, and the ChIP labels do not describe that state. The control arm of such a study is eligible if its controls received no active treatment. |
| A6 | Genetic background | Inbred C57BL/6 | Reject | Other strains and F1 hybrids have variants that change read mapping and the sequence at a peak. The TF-DNA model still reads mm10. |
| A7 | Replicates | At least 2 independent libraries, from different animals or litters, per condition | Restricted: independent test only, not training | Technical replicates of one library do not count. Donors in one pooled library count only if the data identifies them. |
| A8 | Age | One age or stage per library. An aged sample needs a young control from the same study. | Reject | Without the control, age and batch are the same variable. |
| A9 | Independence | No library is already in the project, in any form | Reject | Compare GSM IDs. Then compare barcodes after the download (D3). Re-analyses and atlases often reuse public libraries. |
| A10 | Sample identity | The study states what the cells are, and the data confirms it in C1 | Reject | The samples filed as `mESC` are gastrulation embryos, not cultured ESCs, and that was found only by annotation. |

### Cell lines and cultures

A cultured sample passes only if its karyotype is normal and the ChIP comes from the same culture
conditions. Aneuploid and tumor lines are rejected for training, because copy number changes RNA
and accessibility together. This effect has no relation to regulation.

### Human datasets

Human datasets are outside this guide until a human holdout scheme exists. Do not pool them with
mouse data. The pipeline guide covers mm10 only, and 63 hg38 TFs with `XP_` protein accessions have
no embedding.

## Stage B: Ground truth gates (no download)

Labels are the largest source of noise that we can control. Complete these gates before you
download the data. Write `data/ground_truth_files/cell_type_specific/<dataset>_cell_type_map.tsv`
from the cell types that the paper reports.

| ID | Gate | Pass condition | If it fails | Why |
|---|---|---|---|---|
| B1 | Expected coverage | At least 3 expected cell types each have a tier 1 or 2 source with at least 10 TFs in the frozen TF table | Reject | The model must separate cell types. Most kidney slices have 1 to 6 TFs. |
| B2 | Native binding | No enabled source term has forced expression of a TF | Disable the term | Overexpressed TF binding is not the binding of that cell. Example: most ReMap `fibroblast` TFs are overexpressed. |
| B3 | Correct antigen | The experiment titles agree with the antigen, for each enabled term | Disable the term | Examples: `Bld/B_cells` "Cebpb" files are anti-cJUN. One `Lung_endothelial_cells` "Nr3c1" file is the IgG control. |
| B4 | Correct cell type | The term contains only the stated cell type | Disable the term, or use opt-in | Examples: ChIP-Atlas `Neu/Cerebellar_Cortex` holds cerebral cortex experiments, and `CD34+` pools several populations. |
| B5 | Whole tissue | Whole-tissue terms stay disabled, also for a TF that is restricted to one cell type | Keep as opt-in | The builder takes all antigens in a term, so the binding of all cell types goes to one slice. |
| B6 | Context match | ChIP stage and treatment are near the sample state, or the difference is in the `note` column | Lower the tier | A stimulated or induced ChIP does not describe resting cells. |
| B7 | No substitute | A cell type without a source has no slice | Do not substitute a source | Example: no mouse IC, TAL or thin-limb ChIP exists. A borrowed source gives labels that do not separate cell types (D2). |
| B8 | Explicit exclusions | A term that a name search would select, but that is wrong, has an `EXCLUDED` row | Add the row | An omission is invisible. `verify_ground_truth.py` fails on an `EXCLUDED` source only if the row exists. |

How to check B2 to B4:

- Read the experiment titles in ChIP-Atlas `experimentList.tab` (columns 9 to 12).
- The builder has no per-(term, TF) filter. If one bad file is in a term, you must disable the whole
  term or add that filter.
- TFs outside the frozen table (`TETHER/cached_data/mm10/tf_dna_cache/tf_name_to_idx.csv`, 443 TFs)
  drop out. A new dataset does not justify a change to the table, because every model would have
  to be rebuilt.

Know the limit of these gates. Two ChIP experiments for one TF in the same slice share a target
Jaccard of only 0.27 to 0.32 (measured 2026-09-11). Most of the disagreement between cell types is
already present within one cell type. Prefer TFs with at least 2 independent experiments, and do
not expect label selection to remove this noise.

## Stage C: Data gates (after standard preprocessing)

Download the data and run the standard preprocessing first. Use the same steps as the current
samples, so that a measured difference comes from the data and not from the processing.

### Preprocessing rules

- Start from raw counts. Keep them in `layers["counts"]`.
- Both modalities: `normalize_total` to 1e4, then `log1p`, stored as sparse float32 in `X`. Do not
  scale. The model uses these values as they are. [preprocessing_detailed.md](preprocessing_detailed.md)
  still says "scaled", which is the older convention.
- Keep the same filter floors for all samples: `Min Cells per Gene` 3 and `Min Cells per Peak` 1.
  Choose only the upper limits per sample, from its distribution. Current settings are not uniform:
  `E8.5_rep1` uses 20 and a 20% mitochondrial limit, and the other mESC samples use 3 and 40%.
- Add the thresholds to `data/qc_filtering_settings.tsv`.
- Peaks: use the Cell Ranger ARC peak set, or re-call peaks from fragments with one method. Do not
  use a fixed-width peak set from a deposit.
- Do the same doublet step for all samples.

### Stage C: Sample gates

| ID | Gate | Pass condition | Source | Why |
|---|---|---|---|---|
| C1 | Sample identity | Marker genes confirm the cell types that the study reports | Measured | Confirms A10. |
| C2 | Pairing | At least 95% of barcodes are in both modalities, and the RNA and ATAC cluster ARI is at least 0.2 (shuffled control near 0) | Judgment | A wrong pairing removes the signal without an error. Skin: 0.37, against 0.00 when shuffled. |
| C3 | Input scale | `expm1(X) / counts` is constant within each cell, in both modalities. `X` has no negative values. | Measured | A mismatched scale gave AUROC 0.530 where rank-mapping the inputs gave 0.675. All 17 samples now have sparse, non-negative `X`. |
| C4 | RNA depth | Median of at least 800 detected genes per cell | Judgment | Low depth removes TFs from the input. Skin has 480. |
| C5 | ATAC density | At least 2% of the matrix is nonzero | Judgment | Skin has 0.98% and SHARE-seq brain has 0.76%. All 10x samples have at least 3.27%. |
| C6 | Peak coverage | The median peak is open in at least 1% of cells, which is 0.25 open cells per 25-cell bag | Measured | Skin has 0.10 open cells per bag, and it has the lowest stability: top-10% Jaccard 0.49. |
| C7 | Label origin | Labels come from marker scores or from published labels joined by barcode. They must not come from a map of cluster numbers. | Measured | A liver re-run renumbered the clusters: 4,564 hepatocyte-marker cells became Endothelial. Skin had the same drift. |
| C8 | Unlabeled cells | At most 10% of cells have no label | Judgment | Skin has 29%, because its map drifted from the clusters. |
| C9 | Gene names | Review the upper-case collision warning from the cache builder | Measured | Skin had both `PISD` and `Pisd`. |

After any re-run of preprocessing, compare the `celltype` counts with the previous file before you
build caches. No pipeline step checks the labels.

### Stage C: Slice gates

Calculate these after you filter the ground truth to TFs in the frozen table and genes in the sample.

| ID | Gate | Pass condition | Source | Why |
|---|---|---|---|---|
| S1 | Cells | At least 100 cells | Code: 50 (`2 × max_cells_per_pair`). Judgment: 100 | At 50 cells, two bags use all of the cells. |
| S2 | TF breadth | At least 10 detected TFs | Code: 5 TFs. Judgment: 10 detected | Per-TF metrics over few TFs are noisy, and a TF with no mRNA in the slice is a poor positive label. Kidney `B cells` has 44 TFs, but only 16 to 27 are detected. |
| S3 | Label density | Positive edges are at most 60% of the TF × TG box | Code | Negatives come only from that box. A density above 0.33 cannot supply 1:2 positives to negatives. Report the density. |
| S4 | Source quality | At most 50% of edges are tier 3 | Judgment | The labels must describe this cell type. Kidney `Endothelial` is 100% tier 3. |
| S5 | TF balance | One TF holds at most 50% of the edges | Judgment | Otherwise the slice measures one TF. Example: `Gut` has 84% from one TF. |

### Stage C: Dataset gates

| ID | Gate | Pass condition | If it fails | Why |
|---|---|---|---|---|
| C10 | Distinct slices | At least 3 scoreable slices with different labels (D2) | Reject. Restricted if the dataset is one purified population with at least 20 detected TFs: use it for training and for TF-level evaluation, but not for cell-type specificity claims. | The project's claim is cell-type-specific regulation. |
| C11 | Large slice | At least one slice has 1,000 cells or more | Exclude from stability analysis only | Stability needs repeated subsamples. Current subsamples use 290 to 9,060 cells. |

## Stage D: Label and sample overlap checks

Do these checks before you add the dataset to a training run.

| ID | Check | Action |
|---|---|---|
| D1 | Labels shared with other tissues | Hash the edge set of each new ground truth file and compare it with all existing files. Treat an edge Jaccard above 0.5 as the same labels (Judgment). Add these slices to the holdout list. |
| D2 | Labels shared within the dataset | Slices with the same labels cannot test cell-type specificity. Merge them in `<tissue>_label_map.tsv`, or keep only the slice with the best source. |
| D3 | Duplicate libraries | Compare barcodes and count totals with all existing samples. `GSE246464_HSC` and `GSE246464_HSC_10x_raw` show that a second copy can exist under another name. |
| D4 | Study overlap | Record the study of each library. If one study supplies both training and held-out tissues, do not call the result "unseen study". SHARE-seq brain and skin are this case. |
| D5 | Folder names | `data/processed/<tissue>/` and `data/ground_truth_files/cell_type_specific/mm10/<tissue>/` must have the same name, because evaluation finds the ground truth by tissue folder. |

### Why D1 is necessary

`--auto_holdout_celltypes` compares cell type **names**. Eleven groups of current ground truth files
have identical edges under the same or different names. For example, these files are identical:

- `Skin_endothelial` and `Lung_endothelial` (`GSE140203_shareseq`)
- `Endothelial` (`kidney`, `mouse_liver`, `10x_E18_mouse_brain`)

The run configs of the no-kidney and no-liver models hold out `Endothelial` but list
`Skin_endothelial` for training. The no-skin model holds out no cell type, so it trains on
`Endothelial` and is tested on `Skin_endothelial`. These models therefore see their test labels in
training for these slices. Until the holdout compares label content, pass each such slice with
`--holdout_celltype`.

Slices with identical labels within one tissue:

| Tissue | Slices |
|---|---|
| `kidney` | `Distal_tubule`, `Thick_ascending_limb` |
| `kidney` | `Collecting_duct_IC`, `Collecting_duct_PC` |
| `GSE140203_shareseq` | `Hair_follicle_stem_cells`, `Hair_matrix_TACs`, `Outer_root_sheath` |
| `GSE140203_shareseq` | `OPCs`, `Oligodendrocytes` |
| `GSE140203_shareseq` | `Skin_endothelial`, `Lung_endothelial` |

## Selection priority

Apply this ranking only to datasets that pass all gates. Rank from highest to lowest:

1. **New label content.** The dataset adds scoreable slices with labels that no current tissue has.
   More libraries of a current tissue add cell states but no labels, because all samples of a tissue
   share one ground truth file.
2. **A second independent study of a current tissue.** Today, tissue and study are almost the same
   variable, so a tissue holdout is also a study holdout. A second 10x study with the same cell
   types separates the two effects.
3. **Matched validation.** The same study or cell state has ChIP-seq, CUT&RUN or a perturbation
   result that can give tier 1 labels or an independent test.
4. **More depth.** Higher C4 to C6 values reduce noise from bag sampling.

Keep tissue weight in view: mESC already supplies 6 of the 17 samples and 126 of the 182 sample ×
slice rows. A new library of a large tissue also moves the shared input scaler toward that tissue.

## Metadata to record

These variables are not gates, but they can confound a result. Record them so that you can
stratify by them later.

| Variable | Values to record |
|---|---|
| Study | Accession, PMID, laboratory |
| Library | GSM ID, animal or litter, sequencing batch |
| Biology | Tissue, stage or age, sex, strain and substrain |
| Sample handling | Fresh or frozen, dissociation or nuclei protocol, cell sorting or enrichment |
| Assay | Kit version, Cell Ranger ARC version, reference version, whether the counts include introns |
| Processing | Doublet method, ambient RNA correction (none in the input data), QC thresholds |
| Labels | Source (published, marker score, label transfer), reference atlas |

Do not use a deposit whose RNA counts exclude intronic reads unless you can recount it. Multiome
RNA comes from nuclei, so an exon-only matrix has a different gene profile.

## Reference values

Values on `X` after preprocessing. Ranges cover the libraries of each dataset.

| Dataset | Assay | Libraries | Cells per library | Median genes per cell | ATAC nonzero % | Median peak open % | Open cells per 25-cell bag | Median peak width (bp) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mESC E7.5–E8.5 | 10x | 6 | 3,654–8,199 | 2,846–5,014 | 4.4–7.6 | 2.1–3.6 | 0.53–0.89 | 863–867 |
| Kidney | 10x | 3 | 5,423–9,666 | 927–1,919 | 3.3–7.0 | 1.6–3.7 | 0.41–0.92 | 866–876 |
| E18 brain | 10x | 1 | 3,998 | 3,280 | 5.3 | 2.3 | 0.58 | 875 |
| HSC | 10x | 4 | 1,214–3,658 | 2,632–2,939 | 10.2–14.4 | 5.1–8.4 | 1.26–2.10 | 880–883 |
| Liver (6 mice) | 10x | 1 file | 19,691 | 3,268 | 6.1 | 2.4 | 0.59 | 875 |
| Skin | SHARE-seq | 1 | 32,416 | 480 | 0.98 | 0.40 | 0.10 | 300 |
| Brain | SHARE-seq | 1 | 2,139 | 1,632 | 0.76 | 0.47 | 0.12 | 499 |

The TF-DNA model reads 256 bp at the peak center. This window covers almost all of a 300 bp peak
but less than a third of an 870 bp peak, which is one more reason to keep one peak method.

## Current samples under these gates

This table tests the thresholds. It covers the measured gates only. Strain, sex, sample handling and
chemistry (A6, A8 and the metadata table) were not audited, and neither were the source terms of the
current maps (B2 to B6). The table does not by itself remove any current sample.

| Dataset | Sample × slice rows | Pass code gates | Pass slice gates | Distinct scoreable slices | Result |
|---|---:|---:|---:|---:|---|
| mESC | 126 | 70 | 26 | 7 | Accept |
| Liver | 6 | 5 | 3 | 3 | Accept |
| HSC | 4 | 4 | 4 | 1 | Restricted (one purified population) |
| E18 brain | 6 | 4 | 2 | 2 | Reject (C10, also has one library) |
| Kidney | 24 | 12 | 2 | 1 | Reject (C10) |
| SHARE-seq skin and brain | 16 | 11 | 0 | 0 | Reject (C4–C6, C8, C10) |

The S2 gate has the largest effect: 118 of the 182 rows have fewer than 10 detected TFs. This
reflects the available ChIP, not only sample quality.

## After acceptance

Follow [pipeline_guide.md](pipeline_guide.md). These items also change when a dataset is added:

- Add the sample to `DATASETS` in `03c_build_celltype_data.sh`, and update `--array`.
- Run `run_build_gt.sh`, then [verify_ground_truth.py](../scripts/generate_celltype_ground_truth/verify_ground_truth.py).
- Read the slice report that the cache builder prints. Compare it with this appraisal.
- Add the sample to the holdout commands (step 6), with the slices from D1.
- Run CellOracle, FigR, LINGER, Pando and SCENIC+ on the same preprocessed file. If a method fails,
  record the failure. Do not drop the dataset to hide it.
- Retrain all holdout models. The input scaler comes from all training samples, so every model
  changes. Compare old and new models only on the same evaluation set.

## Reporting rules

- Report per-TF AUROC next to pooled AUROC. In July GRNs, per-TF AUROC fell from about 0.57 to about
  0.50 after removal of each TG's mean score, so check this control too.
- Split the metrics into TFs that the TF-TG model saw in training and TFs that it did not see.
- Give the shuffled-score baseline for each metric.
- Stratify the metrics by tier.
- Say "unseen cell type", not "unseen TF", unless the TF is also absent from TF-DNA training. The
  TF-DNA model trains on ChIP-Atlas peaks of all cell types, and many of the same experiments
  supply the TF-TG labels.

## Appraisal record

Save one copy per dataset in `TETHER/docs/dataset_appraisals/<dataset>.md`.

```markdown
# <dataset name>

- Accession, PMID:
- Appraised by, date:
- Verdict: Accept | Restricted (use: ) | Reject (gates: )

## Stage A
A1:   A2:   A3:   A4:   A5:   A6:   A7:   A8:   A9:   A10:

## Stage B
Map file:
Expected slices with at least 10 tier 1 or 2 TFs:
Terms disabled or excluded, with reason:

## Stage C
| Library | Cells | Median genes | ATAC nonzero % | Median peak open % | Unlabeled % | ARI | C3 |
|---|---|---|---|---|---|---|---|

| Library | Slice | Cells | Detected TFs | Density | Tier 3 % | Top TF % | Scoreable |
|---|---|---|---|---|---|---|---|

C10:   C11:

## Stage D
Shared labels (D1, D2):
Duplicate check (D3):
Study overlap (D4):

## Metadata
(table from "Metadata to record")
```
