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

Three methods were run. **Two produced confident, wrong answers.** Both failure scripts were
kept in `05_atlas_annotation/` for a while on purpose, since the failures are easy to repeat —
as of the 2026-08-12 review pass (§9) they've been **deleted**; nothing downstream ever read
their output, and the write-up below plus in `README.md` is what needs to survive, not the code.

1. **Marker-signature scoring** (`annotate_combined_pijuansala.py`, removed) — **rejected.**
   Called 2,366 cells PGC while `Dppa3`/`Nanos3` were flat zero, driven by non-specific
   `Prdm1`/`Ifitm3`, and flagged those clusters *confident*. Lesson: a large margin means
   the winner beat the runner-up, not that specific genes drove it.

2. **kNN transfer in a joint Harmony embedding** (`label_transfer_pijuansala.py`, removed) —
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

1. **Decide whether this pipeline's output is meant to feed the GRN model at all.** Confirmed
   by grep (2026-08-12): zero references anywhere in `TETHER/scripts/`, `TETHER/utils.py`,
   `TETHER/config.py`, `TETHER/muon_preprocessing.py`, or any notebook to `mESC_combined.h5mu`,
   `celltype_final`, or `mouse_preprocessing_scripts` itself. The GRN model's training cache
   (`TETHER/cached_data/mESC_cache/`) is already populated from the same raw gastrulation
   samples via the older, uncoordinated `muon_preprocessing.py` + `config.py` route (see
   `TETHER/docs/preprocessing_detailed.md`), which has no cell-type annotation step at all.
   `CLAUDE.md` doesn't mention this pipeline exists. So the entire validated annotation effort
   here — three methods tried, one calibrated and adopted, marker-validated, cross-checked
   against a "free" biological timing signal — currently has **no consumer**. Whether/how to
   wire `celltype_final` into GRN model training, or whether the two pipelines are
   intentionally meant to stay separate, is a call only the PI can make; at minimum, `CLAUDE.md`
   should mention this pipeline exists so a future session can discover it.
2. Ambient RNA correction is unmeasured and, as of 2026-08-12, **deliberately deferred**. No
   raw/unfiltered droplet matrix exists anywhere for these samples — confirmed directly this
   session: the `mESC_10x_raw/` directory every script points at as "raw" has zero barcodes
   below 2,256 UMIs (median 32,752), i.e. it's already a called-cell matrix, not a true raw
   matrix with an empty-droplet population. That rules out CellBender, SoupX's `autoEstCont`,
   and any empty-droplet-derived decontX/scAR profile. `scar` (Python-native; installed in
   `my_env` this session) and R/Bioconductor `celda`'s decontX (not installed — `my_env` has no
   R at all) both remain technically available, but either would need a manually-supplied
   ambient profile instead — e.g. cluster 0, already characterized in §6b as ~pure ambient.
   Revisit only if a real raw matrix becomes available, or the PI decides the manual-profile
   compromise is worth it.
3. If ambient correction is later pursued and shows material contamination, re-run from step 02
   onward. This is the change most likely to rescue the 4,374 unresolved cells; sub-clustering
   alone cannot.
4. Optionally merge sub 3's 94 cardiomyocytes into cluster 22 in `celltype_final`. At 0.2%
   of the object it will not move any downstream result — leaving them flagged is equally
   defensible.
5. Recalibrate cell-cycle scoring before using `phase` for anything.
6. The CRISPR T-KO/T-WT pair still needs its own analysis, separate from this trajectory.
7. The workflow described in §1–§7 (as of the previous session) was in fact committed, in
   `4c2845c`/`c1db006` — this file previously said otherwise. Today's review/cleanup pass (§9)
   is not yet committed.

---

## 9. Code review & cleanup — 2026-08-12

A follow-up session reviewed this workflow end-to-end against newly-added SciAgent-Skills
single-cell/multiomics guidance (`scanpy-scrna-seq`, `muon-multiomics-singlecell`,
`harmony-batch-correction`, `single-cell-annotation-guide`, `anndata-data-structure`), checking
claims against the actual on-disk outputs (`combine_summary.json`, `final_atlas_labels.csv`,
`centroid_annotation_rna_leiden.csv`, `marker_genes_top25.csv`) rather than the code alone.

**Net verdict: the pipeline held up well.** The three-method annotation trial plus a rotated-label
calibration control (6/6 correct labels supported, 0/6 rotated labels supported) is a clean
execution of the "use multiple independent methods, validate with a control" principle. The
adopted centroid-correlation method is methodologically convergent with SingleR (Aran et al. 2019,
*Nat Immunol*) — a peer-reviewed, technology-robust reference-annotation approach — which is
independent validation that the *design*, not just the calibration outcome, is sound. The
`ad.concat(join="inner")` reasoning (a gene absent from one sample was filtered, not observed as
zero) and the explicit scale-then-restore-from-`.layers["counts"]` pattern are both more careful
than the default scanpy idiom. Several previously-documented library-bug workarounds (harmonypy
orientation, muon WNN `use_rep` dict, per-chromosome peak-consensus accumulation) were re-verified
in place and are still doing their job.

Changes made this pass:

- **Removed `annotate_combined_pijuansala.py` and `label_transfer_pijuansala.py`** (plus their
  `.sh` wrappers) from `05_atlas_annotation/` — see the updated §4 above and `README.md`. Verified
  first that nothing downstream reads their output and that neither file self-identified as
  rejected in-file (a future session opening either script cold, or running its `.sh` wrapper,
  would previously have had no signal it was discouraged).
- **Closed an `OVERRIDES` staleness gap in `07_final_labels/reconcile_atlas_labels.py`.** The five
  hand-adjudicated cluster overrides were a pure string-keyed lookup on cluster ID, checked
  *before* the marker-revalidation branch every other cluster gets — a Leiden renumbering on a
  future re-run could have silently applied an old override (label + justification) to whatever
  cluster now holds that ID. Each override now also carries the expected `n_cells` and expected
  centroid label captured at write time; a mismatch at lookup time raises `RuntimeError` instead
  of emitting a silent label, consistent with this file's existing fail-loud handling of the
  sibling case (an unadjudicated cluster).
- **Closed a per-sample confidence-caveat gap in `03_per_sample_reports/build_scrna_report.py`.**
  `02`'s per-sample `score_marker_panels()` uses the identical statistical pattern — `score_genes`
  → z-score across clusters → argmax → margin-as-confidence — as the rejected atlas-level
  `annotate_combined_pijuansala.py`, on a literal subset of the same PGC marker panel. The report's
  only confidence caveat (margin < 0.5) fires in the *opposite* direction from the documented
  failure (which was high-margin, confident, and wrong). Added a check that intersects each
  cluster's assigned panel against its own top-25 DE genes; an empty intersection is now flagged
  (both a per-row table flag and a caveat) regardless of margin. Per-sample `cell_type` was and
  remains a QC-dashboard artifact only — nothing downstream of `02` treats it as ground truth for
  `celltype_final` — but the report can now catch the specific failure mode it already had the
  data to catch.
- Added a one-line staleness caveat to `inspect_unresolved_clusters.py`'s `DISPUTES` dict and
  `verify_rescued_subclusters.py`'s `CANDIDATES` dict — both hardcode cluster IDs from this same
  Leiden run. Lower severity than `OVERRIDES` (documentation only, no code guard): both are one-off
  adjudication tools a human reads once, not steps re-executed on every pipeline run, and
  `verify_rescued_subclusters.py`'s genuine/ambient verdict doesn't actually depend on the
  possibly-stale `reference_cluster` ID, only on the candidate's own same-library floor.
- Documented in `README.md` that `06_label_validation/validate_cluster_annotation.py` must run
  *after* `07_final_labels/` despite its directory number — it reads `celltype_final`, and its own
  `.sh` wrapper never passes the `--label` override that would avoid needing it.
- Removed confirmed-unused imports/variables: `pandas` (`01_qc_scan/qc_scan.py`), `anndata`
  (`06_label_validation/estimate_ambient_contamination.py`), `numpy`
  (`07_final_labels/reconcile_atlas_labels.py`), an unused parsed-but-unread `combine_summary.json`
  load (`08_reporting/build_annotation_report.py`).
- Reworded a stale comment in `02_per_sample_annotation/annotate_scrna_celltypes.py` claiming its
  argparse QC defaults mirrored a specific sample's row in `data/qc_filtering_settings.tsv` — they
  no longer do (4 of 6 values drifted), and are moot in production anyway since the TSV is always
  found. Now documented as an arbitrary last-resort fallback instead of tied to one sample.

Not resolved this pass, and not something a code review should resolve unilaterally — see §8 item 1
above.
