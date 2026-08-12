# Handoff — mouse gastrulation preprocessing & annotation

**Status as of 2026-08-12.** Written for a new session picking this up cold.

Everything below concerns the mouse/mESC gastrulation Multiome samples, which feed the
TETHER GRN model. It is separate from the GRN model code in `TETHER/scripts/` and
`TETHER/bash_scripts/` — **do not add preprocessing scripts back into those directories.**

---

## 1. What the data actually is

Nine wild-type 10x Multiome samples spanning mouse gastrulation, **not cultured ESCs**
despite the `mESC` directory name. Timepoints E7.5 → E8.75:

| Sample | Cells | | Sample | Cells |
|---|---|---|---|---|
| E7.5_rep1 | 5,952 | | E8.5_rep1 | 7,438 |
| E7.5_rep2 | 1,778 | | E8.5_rep2 | 8,772 |
| E7.75_rep1 | 4,826 | | E8.75_rep1 | 3,498 |
| E8.0_rep1 | 5,613 | | E8.75_rep2 | 4,908 |
| E8.0_rep2 | 4,846 | | **Integrated total** | **47,631** |

Two further samples, `E8.5_CRISPR_T_KO` and `E8.5_CRISPR_T_WT`, exist in
`data/raw/mESC_10x_data` and are **deliberately excluded** from the integrated object: they
are a genotype contrast, not a timepoint, and co-integrating them would let Harmony regress
away the very difference that experiment exists to measure.

**Important constraint on the raw data:** `data/raw/mESC_10x_data/<sample>/` holds only
GEO-deposited **filtered** matrices (~9–11k barcodes each), plus ATAC fragments. There is
**no raw/unfiltered droplet matrix and no BAM/FASTQ.** This blocks CellBender and SoupX's
`autoEstCont` — see §6.

---

## 2. Where things live

```
TETHER/mouse_preprocessing_scripts/
├── README.md                  <- run order, how to read the labels
├── handoff.md                 <- this file
├── common/mudata_utils.py     <- sanitize_for_h5, run_harmony (shared by 04/05/07)
├── 01_qc_scan/                <- QC distributions -> data/qc_scan/
├── 02_per_sample_annotation/  <- per-sample QC, clustering, marker annotation
├── 03_per_sample_reports/     <- one HTML report per sample
├── 04_combine_samples/        <- consensus peaks, Harmony, WNN -> mESC_combined.h5mu
├── 05_atlas_annotation/       <- three annotation methods (only one adopted)
├── 06_label_validation/       <- marker validation, disputed clusters, sub-clustering
├── 07_final_labels/           <- writes celltype_final into the h5mu
└── 08_reporting/              <- HTML report for integration + annotation
```

Step directories start with digits, so they are not importable package names. Cross-step
imports go through the workflow root:

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.mudata_utils import sanitize_for_h5
```

**Main object:** `data/processed/mESC/combined/mESC_combined.h5mu` (14 GB).
**Report:** https://claude.ai/code/artifact/f00036b9-84b7-4a80-b3c2-aa6e20782e54

---

## 3. Reading the labels

```python
mdata["rna"].obs["celltype_final"]          # the label to use
mdata["rna"].obs["celltype_final_support"]  # how it is justified
```

| `celltype_final_support` | Clusters | Cells | Meaning |
|---|---|---|---|
| `centroid+marker` | 18 | 37,338 (78.4%) | centroid call confirmed by its own canonical markers |
| `marker_override` | 3 | 5,669 (11.9%) | markers outvoted the correlation; evidence in `final_atlas_labels.csv` |
| `unresolved` | 2 | 4,624 (9.7%) | deliberately unlabelled — **not a cell type, exclude downstream** |

90.3% confidently labelled across 18 atlas populations. Also present: `leiden` (23 RNA
clusters), `leiden_wnn` (36 joint clusters), `celltype_atlas_*` (pre-reconciliation
centroid labels), `celltype_pijuansala*` (rejected marker-scoring labels, kept for
provenance).

---

## 4. The annotation history — read this before re-annotating

Three methods were run. **Two produced confident, wrong answers.** Both failure scripts are
kept in `05_atlas_annotation/` on purpose, because the failures are easy to repeat.

1. **Marker-signature scoring** (`annotate_combined_pijuansala.py`) — **rejected.** Called
   2,366 cells PGC while `Dppa3`/`Nanos3` were flat zero, driven by non-specific
   `Prdm1`/`Ifitm3`, and flagged those clusters *confident*. Lesson: a large margin means
   the winner beat the runner-up, not that specific genes drove it.

2. **kNN transfer in a joint Harmony embedding** (`label_transfer_pijuansala.py`) —
   **rejected.** Put 32.6% of cells into "Pharyngeal mesoderm", the consensus for 8 of 23
   clusters, including clusters independently verified as Pax6+ neurectoderm, Meox1+
   somitic and Pou5f1-high epiblast, at 0.78–0.89 confidence. The atlas labels were checked
   directly against expression and are correct — the *shared embedding* is at fault. Query
   is single-nucleus Multiome, reference is whole-cell scRNA-seq, and Harmony correcting
   that technology gap collapses query cells onto the reference's dense, non-distinctive
   mesodermal centre. Outputs quarantined in
   `data/processed/mESC/combined/rejected_knn_transfer/` with a README.

3. **Cluster-centroid Spearman correlation** (`centroid_label_transfer.py`) — **adopted.**
   No shared embedding, so nothing to over-correct; rank correlation is insensitive to the
   nuclear-vs-whole-cell scale difference. Crucially, its confidence flag *tracks
   correctness*: self-flagged "distinctive" clusters were 14/15 marker-supported (93%),
   self-flagged uncertain ones 4/8 (50%).

**The reference is stage-matched** (Pijuan-Sala et al., Nature 566:490–495, 2019;
doi:10.1038/s41586-019-0933-9), at
`/gpfs/Labs/Uzun/DATA/.../REFERENCE/pijuan_sala_atlas/atlas_reference.h5ad`
(116,312 cells, 37 populations). This matters: the annotation guide routes developmental
tissue to a stage-matched reference specifically because adult references produce
systematic errors. CellTypist is **not** applicable — no mouse gastrulation model exists.

---

## 5. Validation performed

**Marker validation** (`06_label_validation/validate_centroid_labels.py`) — for each
cluster, is the canonical panel of its *assigned* type enriched there relative to all other
clusters? Calibrated on controls first: correct labels scored 6/6 supported, deliberately
rotated labels 0/6 (recovering the true label each time), so the test has real power.

**Disputed clusters** (`inspect_unresolved_clusters.py`) — the five failures were resolved
on raw per-gene ratios, producing three overrides and two unresolved calls. All recorded
with evidence in `07_final_labels/reconcile_atlas_labels.py`.

**Free biological check:** no timepoint information entered the annotation, yet the labels
reproduce gastrulation timing — cardiomyocytes absent until E8.0, neural crest and spinal
cord rising monotonically, Primitive Streak confined to E7.5/E7.75, ExE ectoderm declining
14.5% → 0%.

**Cluster 1 deep validation** (`validate_cluster_annotation.py`, per the annotation guide's
Tier-1 protocol — positive *and* negative markers, co-expression, cell cycle, composition):
Parietal endoderm **SUPPORTED**, 9/9 positives (8 at rank 1 of 23; `Thbd` 53.6×, `Sox7`
23.3×), 0/13 negatives violated, 100% co-expression of ≥2 positives. DE against the
runner-up cluster 11 returns `Nid1`, `Thbd`, `Nog` — Reichert's membrane components.

---

## 6. Open issues — the important part

### 6a. Ambient RNA is uncorrected and is now the limiting factor

Measured, not suspected. Classic soup genes sit at near-identical detection inside and
outside any given cluster:

| Gene | % in cluster 1 | % elsewhere |
|---|---|---|
| Ttr | 86.4 | 87.3 |
| Apoa1 | 82.7 | 84.3 |
| Hbb-bh1 | 71.1 | 69.9 |
| Afp | 53.0 | 54.3 |

A uniform floor of the highest-expressing genes across every cluster is the ambient
signature. It did not corrupt cluster 1 (ratios ≈1.0 cancel in any comparison) but it
**dominates the low-depth cells**, which is why cluster 0 could not be annotated.

**No decontamination was ever run.** Because only filtered matrices exist (§1):
- **CellBender** — needs the raw h5 with empty droplets. Not possible.
- **SoupX `autoEstCont`** — profiles the soup from empty droplets. Not possible by default;
  usable only with a manually supplied soup profile via `setSoupProfile()`.
- **decontX** (R/Bioconductor `celda`) — estimates contamination from cluster structure
  alone, no empty droplets required. **This is the viable option** and is not installed
  (`celda`, `decontx`, `scar` all absent from `my_env`).

`06_label_validation/estimate_ambient_contamination.py` was **written but never run** — it
estimates the contamination fraction ρ from lineage-exclusive genes as a proxy for the
missing empty droplets. It is **untested**; run it before trusting it. Knowing whether ρ is
~5% or ~30% decides whether re-running with decontX is worth it.

### 6b. Cluster 0 (4,468 cells) is a mixture, and mostly soup

Sub-clustered (`subcluster_unresolved.py`, positive control verified to recover a planted
population) into 4 subclusters, then tested for ambient with `verify_rescued_subclusters.py`.
Pure ambient puts every gene at ~1.0× its own library's floor:

| Sub | n | Fold over library floor | Verdict |
|---|---|---|---|
| 0 | 1,812 | 0.64–1.11 | ambient |
| 1 | 1,689 | 0.82–1.53 | **not a distinct population** |
| 2 | 873 | no markers | unresolvable |
| 3 | 94 | 1.64–**11.90** (`Ttn`, `Myh6`) | **genuine low-depth cardiomyocytes** |

**Only ~94 cells (2%) are recoverable.** Sub 1 looked like epiblast (`Pou5f1` 77%/17%) but
that was an artifact of comparing against a different sample — against its *own* library
`Pou5f1` is only 1.43×. E7.75_rep1 is genuinely epiblast-rich, so its floor is already 60%
of the real Primitive Streak level.

**Method warning for whoever continues this:** cross-sample marker frequency is not
sufficient evidence in a sample-confounded cluster. Use the absolute-expression test
against the same-library floor.

### 6c. Cell-cycle scoring is not usable as run

`sc.tl.score_genes_cell_cycle` puts 94.8% of the *entire dataset* in non-G1 (G1 = 5.2%).
Gastrulation embryos are fast-cycling but 95% is not credible — G1 is assigned only when
both S and G2M scores are ≤0, and these scores skew positive here. Cell-cycle confounding
therefore **cannot currently be assessed**. Recalibrate before using `phase`.

### 6d. Parietal endoderm (cluster 1) abundance is a dissection artifact

Its timepoint profile is bimodal (E7.5 1.5%, E8.0 0.3%, E8.5 2.3%, E8.75 0.04%). It spans
four samples so it is not batch-driven; parietal endoderm forms **Reichert's membrane**, an
extraembryonic membrane whose recovery depends on dissection. **Exclude cluster 1 from
composition-over-time analyses.**

---

## 7. Environment gotchas (each of these cost a failed job)

- **`set -euo pipefail` before `source activate my_env` kills every job in ~8s** with
  `MKL_INTERFACE_LAYER: unbound variable` — the MKL activate.d hook reads it unguarded. All
  batch scripts use `set -eo pipefail` → activate → `set -u`. Preserve that order.
- **harmonypy 2.0 breaks `sc.external.pp.harmony_integrate`** — the wrapper transposes
  `Z_corr` for the old (PCs × cells) convention, but 2.0 returns (cells × PCs). Use
  `common.mudata_utils.run_harmony`, which checks orientation.
- **muon 0.1.9 WNN** stores `use_rep` as a per-modality dict in `uns['wnn']['params']`,
  which scanpy's `_choose_representation` chokes on (`unhashable type: 'dict'`). Pop that
  key before `sc.tl.umap`. The parameter is `neighbor_keys`, not `use_rep`.
- **Consensus peaks must accumulate per chromosome.** A global
  `np.maximum.accumulate(end)` lets chr1's ~195 Mb coordinates dominate every later
  chromosome, collapsing them into single intervals the width filter then deletes —
  producing a *plausible-looking* ATAC modality containing essentially only chr1. There is
  now a hard collapse guard. A single-chromosome unit test will not catch this.
- **h5mu writes fail late** if block assignment (`df[[a,b,c]] = arr`) makes columns object
  dtype. Always run `sanitize_for_h5` before `.write()`.
- **`scikit-misc` IS installed** — `seurat_v3` HVG works (needs `layer="counts"`). Older
  notes claiming a silent fallback to `seurat` are obsolete.
- Long jobs use `-p memory --mem=700G`; the h5mu is 14 GB and loading it dominates runtime.

---

## 8. Suggested next steps

1. **Run `estimate_ambient_contamination.py`** (untested — verify it) to get ρ. This is the
   highest-value next action; it decides everything below.
2. If ρ is material, **install `celda` and re-run with decontX** from step 02 onward. This
   is the change most likely to rescue the 4,374 unresolved cells; sub-clustering alone
   cannot.
3. Optionally merge sub 3's 94 cardiomyocytes into cluster 22 in `celltype_final`. At 0.2%
   of the object it will not move any downstream result — leaving them flagged is equally
   defensible.
4. Recalibrate cell-cycle scoring before using `phase` for anything.
5. The CRISPR T-KO/T-WT pair still needs its own analysis, separate from this trajectory.
6. Nothing here has been committed. `git status` shows the whole workflow directory as new
   or renamed; the moves used `git mv` where files were already tracked.
