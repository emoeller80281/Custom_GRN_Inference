# Dataset selection criteria

This guide defines how to appraise a new single-cell multiome dataset before it enters the TF-TG
pipeline. The goal is to add data without adding confounding variables. Each criterion comes from
how the models use the data or from a failure that this project has already measured.

Measured values are from 2026-10-30 unless a section gives another date. They cover the 17 mouse
samples in `TETHER/bash_scripts/03c_build_celltype_data.sh`. No human sample has been through these
stages yet. The gates apply to mouse and human data, with the human conditions in
[Human datasets](#human-datasets).

## Terms

- **Slice:** one cell type in one sample, with its own ground truth file.
- **Bag:** the random cells that the model scores for one edge. The holdout models use 25 cells
  (`--max_cells_per_pair 25`).
- **Tier:** the `proxy_tier` of a ChIP source in `<dataset>_cell_type_map.tsv`. Tier 1 is the same
  cell type. Tier 2 is the same lineage. Tier 3 is a distant derivative, a cell line, a tumor or
  another organ.
- **Detected TF:** a TF with a nonzero RNA value in at least 5% of the cells of the slice.
- **Donor:** one person who gave cells or tissue.
- **TF table:** `TETHER/cached_data/<species>/tf_dna_cache/tf_name_to_idx.csv`. Each species has its
  own table.

## Rules for every appraisal

1. Finish the appraisal before you run any model on the dataset. Model performance is never a
   selection criterion, because selecting on it makes the comparison with other methods invalid.
2. Do the stages in order. Stages A and B need no download. They reject most unsuitable datasets at no compute cost.
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

| ID  | Gate                   | Pass condition | If it fails | Why |
| --- | ---------------------- | -------------- | ----------- | --- |
| A1  | Paired assay           | RNA and ATAC come from the same cell, with the same barcode | Reject | The model joins RNA and ATAC per cell for each bag. Computationally integrated data has no true pairing. |
| A2  | Assay type             | 10x Multiome | Restricted: independent test only, not training | Assay changes depth and peak shape (see [Reference values](#reference-values)). Both SHARE-seq samples fail the depth gates. |
| A3  | Organism and reference | Mouse with counts on mm10, or human with counts on GRCh38 (hg38), or FASTQ files to realign. One library holds one organism, so a mixed-species (barnyard) library fails. | Reject | Each species' TF-DNA model scores that genome's sequence, and its ground truth uses that genome's TSS positions. Realign other assemblies (hg19, mm9). Do not lift over their peaks. |
| A4  | Raw counts             | Per-cell raw counts for both modalities, with peak coordinates. A fragment file is preferred. | Reject | Normalized, scaled or corrected matrices cannot give the common input scale. |
| A5  | Biological state       | Unperturbed cells: no knockout, drug, disease or stimulation. Human: tissue from donors with no reported disease of that organ. | Reject. Restricted for human tumor-adjacent tissue: independent test only, not training. | Perturbation changes regulation, and the ChIP labels do not describe that state. The control arm of such a study is eligible if its controls received no active treatment. |
| A6  | Genetic background     | Mouse: inbred C57BL/6. Human: the study identifies the donor of each library, and no donor has a reported abnormal karyotype. | Reject | Other mouse strains and F1 hybrids have variants that change read mapping and the sequence at a peak. The TF-DNA model still reads the reference. Human donors also have these variants, and no study can avoid them, so the requirement is to know the donor. |
| A7  | Age                    | One age or stage per library. An aged mouse sample needs a young control from the same study. Human: fetal, child and adult donors are separate conditions. Adult donors can form one condition if each donor's age is recorded. | Reject | Without the control, age and batch are the same variable. Donor age cannot be fixed by design, so a human study must allow stratification by age. |
| A8  | Independence           | No library is already in the project, in any form | Reject | Compare GSM IDs. Then compare barcodes after the download. Re-analyses and atlases often reuse public libraries. |
| A9  | Sample identity        | The study states what the cells are. | Reject | The samples filed as `mESC` are gastrulation embryos, not cultured ESCs, and that was found only by annotation. |
| A10 | Data access            | The terms of a controlled-access deposit (dbGaP, EGA) permit this use, and approval is in place before the download | Reject until approved | Most human FASTQ files are controlled. Record the access terms, and keep controlled data out of shared folders and public results. |

### Cell lines and cultures

A cultured sample passes only if its karyotype is normal and the ChIP comes from the same culture
conditions. Aneuploid and tumor lines are rejected for training, because copy number changes RNA and accessibility together. This effect has no relation to regulation. This rule includes the human lines that hold most human ChIP, such as K-562, Hep G2 and MCF-7.

### Human datasets

Human datasets use the same gates and the same thresholds as mouse datasets (rule 3). Do not pool human and mouse data in one TF-DNA or TF-TG model.

#### Human ChIP coverage

Human ChIP is mostly in cell lines, which are tier 3 unless the dataset is that line. The gates
will therefore remove most primary-tissue slices. Do not change a tier or a threshold to keep them.

Number of TFs per ChIP-Atlas term, before filtering to the TF table (from
`chip_atlas_human_num_tfs_by_celltype.tsv`, 2026-09-03):

| Group                       | Terms and TF counts |
| --------------------------- | ------------------- |
| Tumor and transformed lines | Hep G2 665, K-562 565, 293 419, MCF-7 262, LoVo 244, HeLa 199, HCT 116 185, GM12878 177, A549 131 |
| Pluripotent cells           | hESC H1 96, iPS cells 76, hESC H9 60, hESC HUES64 36 |
| Stem-cell derived           | iPSC derived neural cells 26, iPSC derived cardiac cells 19 |
| Blood                       | CD4+ T cells 48, B cells 26, Macrophages 25, Erythroblasts 24, CD34+ 22, Monocytes 15, CD34 Hematopoietic stem cells 15, T cells 12, Dendritic Cells 11 |
| Skin and vessels            | HUVEC 41, Keratinocytes 38, Dermal fibroblast 23 |
| Other primary               | Adipocytes 11, Bronchial epithelial cells 9, Hepatocytes 5, Pancreatic islets 5, Kidney Cortex 5, Brain 3 |

## Stage B: Ground truth gates (no download)

Labels are the largest source of noise that we can control. Complete these gates before you
download the data. Write `data/ground_truth_files/cell_type_specific/<dataset>_cell_type_map.tsv`
from the cell types that the paper reports.

| ID  | Gate                | Pass condition | If it fails | Why |
| --- | ------------------- | -------------- | ----------- | --- |
| B1  | Native binding      | No enabled source term has forced expression of a TF | Disable the term | Overexpressed TF binding is not the binding of that cell. Examples: most mouse ReMap `fibroblast` TFs are overexpressed. In human ChIP-Atlas `293`, at least 26% of TF experiments have a tag or overexpression word in the title. |
| B2  | Correct antigen     | The experiment titles agree with the antigen, for each enabled term | Disable the term | Examples: `Bld/B_cells` "Cebpb" files are anti-cJUN. One `Lung_endothelial_cells` "Nr3c1" file is the IgG control. |
| B3  | Whole tissue        | Whole-tissue terms stay disabled, also for a TF that is restricted to one cell type | Keep as opt-in | The builder takes all antigens in a term, so the binding of all cell types goes to one slice. |
| B4  | Context match       | ChIP stage and treatment are near the sample state, or the difference is in the `note` column | Lower the tier | A stimulated or induced ChIP does not describe resting cells. |
| B5  | No substitute       | A cell type without a source has no slice | Do not substitute a source | Example: no mouse IC, TAL or thin-limb ChIP exists. A borrowed source gives labels that do not separate cell types. |
| B6  | Explicit exclusions | A term that a name search would select, but that is wrong, has an `EXCLUDED` row | Add the row | An omission is invisible. `verify_ground_truth.py` fails on an `EXCLUDED` source only if the row exists. |

## Metadata to record

These variables are not gates, but they can confound a result. Record them so that you can
stratify by them later.

| Variable | Values to record |
|---|---|
| Study | Accession, PMID, laboratory |
| Library | GSM ID, animal or litter, sequencing batch |
| Biology | Tissue, stage or age, sex. Mouse: strain and substrain. Human: donor ID, ancestry if reported, source of the tissue (organ donor, surgery, biopsy), medication if reported |
| Access | Open or controlled, repository, access terms |
| Sample handling | Fresh or frozen, dissociation or nuclei protocol, cell sorting or enrichment |
| Assay | Kit version, Cell Ranger ARC version, reference version, whether the counts include introns |
| Processing | Doublet method, ambient RNA correction (none in the input data), QC thresholds |
| Labels | Source (published, marker score, label transfer), reference atlas |

Do not use a deposit whose RNA counts exclude intronic reads unless you can recount it. Multiome
RNA comes from nuclei, so an exon-only matrix has a different gene profile.

## Appraisal record

Save one copy per dataset in `TETHER/docs/dataset_appraisals/<dataset>.md`.

```markdown
# <dataset name>

- Accession, PMID:
- Organism and reference:
- Appraised by, date:
- Verdict: Accept | Restricted (use: ) | Reject (gates: )

## Stage A
A1:   A2:   A3:   A4:   A5:   A6:   A7:   A8:   A9:   A10:

## Stage B
Map file:
Expected slices with at least 10 tier 1 or 2 TFs:
Terms disabled or excluded, with reason:

## Metadata
(table from "Metadata to record")
```