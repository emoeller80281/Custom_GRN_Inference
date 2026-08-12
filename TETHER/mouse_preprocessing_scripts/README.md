# Mouse gastrulation preprocessing & annotation workflow

End-to-end pipeline taking the raw 10x Multiome mESC/gastrulation samples in
`data/raw/mESC_10x_data` to a single Harmony-integrated, atlas-annotated MuData at
`data/processed/mESC/combined/mESC_combined.h5mu`.

Every step is a SLURM batch script (`sbatch <script>.sh`) that activates `my_env`. Numbered
directories are the run order; each holds the Python entry point and the batch script that
submits it.

| Step | Directory | Produces |
|---|---|---|
| 01 | `01_qc_scan/` | Per-sample QC distributions, quantiles, MAD bounds and survival curves → `data/qc_scan/`. The evidence base for choosing thresholds. |
| 02 | `02_per_sample_annotation/` | Per-sample QC filtering, doublet removal, clustering, marker annotation → `data/processed/mESC/<sample>/` |
| 03 | `03_per_sample_reports/` | One self-contained HTML report per sample |
| 04 | `04_combine_samples/` | Consensus peaks, gene intersection, Harmony (RNA + ATAC), WNN → `combined/mESC_combined.h5mu` |
| 05 | `05_atlas_annotation/` | Annotation against the Pijuan-Sala 2019 atlas (three methods; see below) |
| 06 | `06_label_validation/` | Independent canonical-marker check of the labels, plus raw evidence for disputed clusters |
| 07 | `07_final_labels/` | Reconciled `celltype_final` + `celltype_final_support` written to the h5mu |
| 08 | `08_reporting/` | HTML report for the integration and annotation |

`common/` holds helpers needed by more than one step (`sanitize_for_h5`, `run_harmony`).
Import it from a step directory with:

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.mudata_utils import sanitize_for_h5
```

## Step 05 contains three annotation methods, only one of which is used

Kept together deliberately — the two failures are the reason the third is trusted, and both
failed in ways that are easy to repeat.

- `annotate_combined_pijuansala.py` — **rejected.** Marker-signature scoring with z-score
  argmax. Called 2,366 cells PGC while `Dppa3`/`Nanos3` were flat zero, driven by
  non-specific `Prdm1`/`Ifitm3`, and flagged those clusters *confident*. A large margin only
  means the winner beat the runner-up, not that specific genes drove it.
- `label_transfer_pijuansala.py` — **rejected.** kNN vote in a joint Harmony embedding. Put
  32.6% of cells into "Pharyngeal mesoderm", including verified Pax6+ neurectoderm and
  Meox1+ somitic mesoderm, at high confidence. The atlas labels were verified correct; the
  shared embedding is at fault, because the query is single-nucleus and the reference is
  whole-cell, and correcting that gap collapses query cells onto a non-distinctive
  reference centroid.
- `centroid_label_transfer.py` — **adopted.** Spearman correlation of cluster centroids
  against atlas cell-type centroids over reference-informative genes. No shared embedding,
  and rank correlation is insensitive to the nuclear/whole-cell scale difference.

Outputs of the rejected runs are kept under
`data/processed/mESC/combined/rejected_knn_transfer/` with a README recording why they fail.

## Reading the final labels

```python
mdata["rna"].obs["celltype_final"]           # the label
mdata["rna"].obs["celltype_final_support"]   # how it is justified
```

`celltype_final_support` is one of:

- `centroid+marker` — centroid call confirmed by its own canonical markers (78.4% of nuclei)
- `marker_override` — markers outvoted the correlation; see `final_atlas_labels.csv` for the
  per-gene evidence (11.9%)
- `unresolved` — deliberately unlabelled (9.7%). Clusters 0 and 13 are low-complexity
  (median 1,439 and 1,291 genes against 3,159–5,166 elsewhere) and are **not** a cell type.
  Cluster 0 is largely one library (46% E7.75_rep1, 31% E8.5_rep2). Exclude or sub-cluster
  these before downstream use; do not treat them as a population.

Reference: Pijuan-Sala et al., *A single-cell molecular map of mouse gastrulation and early
organogenesis*, Nature 566:490–495 (2019), doi:10.1038/s41586-019-0933-9.
