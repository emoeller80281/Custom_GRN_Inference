# `test_tf_tg_predictions.ipynb` — Statistics & Plots

Reference for how each statistic in the main analysis notebook is calculated and which figures the
notebook produces. Cells are `# %%`-delimited under Markdown `##`/`###` headers; search the header
text or a function name to jump to a section. (A code-only export is kept at
`test_tf_tg_predictions.py` when the notebook itself is too large to open.)

## Common inputs

Most statistics operate on a **labeled edge table** with columns `Source` (TF), `Target` (TG),
`Score`, and often `_in_gt` (1 if the edge is in the ground-truth set, else 0). These come from:

- `auprc_all_method_dfs[sample][method]` — loaded by `load_auprc_grns_all_methods` from
  `testing_results/labeled_auprc_grns/{cell_type}/{sample}/{method}.tsv` (built by
  `plot_auprc_all_methods.py`).
- `generalizability_df` — concatenation of
  `testing_results/model_generalizability/comparison_metric_files/*_10000.csv` (built by
  `model_generalizability.py`). This is a **precomputed metrics table**, one row per
  (Model, Test Set); the notebook only aggregates and plots it.
- Per-subsample GRNs under `testing_results/stability_evaluation/…` for the stability analysis.

When a table lacks `_in_gt`, labels are derived on the fly: `label = 1 if "{Source}\t{Target}" in
gt_pairs else 0` (`compute_metrics`, `get_labels_and_scores_for_roc`). All name matching is
upper-cased.

## Statistic definitions

### Threshold-based classification metrics (`compute_metrics`)
Binary predictions are `pred = (Score >= score_threshold)`. From `labels` and `pred`:
- **accuracy / precision / recall / f1** — `sklearn` `accuracy_score` / `precision_score` /
  `recall_score` / `f1_score` (all with `zero_division=0`).
- Returned per (method, sample) as one-row DataFrames, concatenated into the method-comparison table.

### AUROC
- `roc_auc_score(labels, Score)`. Requires both classes present, else `NaN`.
- **Random baseline `rand_auroc`**: `roc_auc_score(labels, np.random.rand(n))` — AUROC of uniform
  random scores on the same labels (≈0.5).

### AUPRC (average precision)
- `average_precision_score(labels, Score)` — used in `plot_sample_prc_curves`,
  `plot_model_vs_test_set_auprc`, and `load_or_generate_tftg_predictions` (which writes
  `auprc`/`rand_auprc` into its metrics rows). The PR curve itself is `precision_recall_curve`.
- **Random baseline `rand_auprc`**: `average_precision_score(labels, np.random.rand(n))`; on a
  balanced-by-construction set this ≈ the positive prevalence.

### AUPRC lift — two different definitions, don't conflate them
- **Subtractive** (`generalizability_df`, `agg_results`): `auprc_lift = auprc − rand_auprc`.
- **Ratio** (`plot_model_vs_test_set_auprc_lift`): `auprc_lift = auprc / rand_auprc`, with the random
  baseline drawn at `1.0`.
- **Curve lift AUC** (`plot_model_vs_test_set_auprc_lift`): interpolate precision and random-precision
  onto a shared recall grid (`interpolate_precision_at_recall`), take the ratio
  `precision / rand_precision` per recall point, then `curve_lift_auc = auc(recall_grid,
  precision_lift)` — area under the precision-lift-vs-recall curve.

### Early precision / precision@k (`precision_at_k`)
- Fraction of true positives among the **top-k highest-scoring** edges: sort by descending `Score`
  (stable), take the top `k`, return `mean(label)`. `k` is capped at `len(y_true)`.
- **`early_precision`** in the metrics tables is `precision_at_k(k=10_000)` (top 10k edges).

### `interpolate_precision_at_recall`
Helper that maps a method's `(recall, precision)` curve onto target recall values via `np.interp`
(sorts by recall, averages duplicate-recall precisions). Used to put every method's PR curve on a
common recall grid for lift curves and cross-method comparison.

### Method ranking (`metric_df_to_rank_df`)
1. Take the median of `metric_col` per (experiment/sample, method).
2. Within each experiment, `rank(method="min", ascending=not higher_is_better)` — best method = rank 1
   (ties share the lower rank).
3. Aggregate across experiments per method: `avg_rank = mean(rank)`, `median_rank = median(rank)`,
   `mean_metric = mean(metric_value)`; sorted by `avg_rank` then `median_rank`.
Returns `(all_ranks_df, rank_df)` — the per-experiment ranks and the across-experiment summary.

### Stability / reproducibility (Jaccard, `calculate_jaccard_index`)
- For two subsample GRNs, take the **top 10% of edges by `Score`** in each, form edge sets
  `{(Source, Target)}`, and compute Jaccard `|X ∩ Y| / |X ∪ Y|`.
- Computed **pairwise across all 10 subsamples** of a sample (`sample_jaccard_indices`), then
  summarized per sample (median).
- **Random baseline**: the same computation after shuffling each subsample's `Score` column
  (`random_state=42`) — Jaccard of two randomly-ranked edge lists.

### GRN size summary (`create_grn_size_summary_df`)
Per (sample, method), after dropping `Score == 0` rows: `num_unique_tfs` (`Source.nunique()`),
`num_unique_tgs`, `num_edges` (row count), `num_true_edges` (`_in_gt.sum()`),
`num_false_edges = num_edges − num_true_edges`.
- **Percent of edge combinations** (`add_percent_edges_vs_own`):
  `100 * num_edges / (own-model num_edges for that sample)` — network size relative to the TF-TG
  own-model GRN on the same sample.

### Generalizability grouping (`## Generalizability` load cell + `agg_results`)
`generalizability_df` rows are sliced by how far the test set is from the model's training data,
using `org_dict` (sample → (organism, cell_type)):
- **own_test_set**: `Model == Test Set`.
- **same_cell_type**: same cell type, different sample.
- **different_cell_type**: same organism, different cell type.
- **different_organism**: different organism.
`agg_results` then groups by Model Cell Type and reports the **mean** of auroc, auprc, auprc_lift,
accuracy, precision, early_precision, recall, f1 (rounded to 3 dp), each also saved as a CSV under
`testing_results/model_generalizability/`.

## Plots produced

Output directories are all defined in the notebook's `## Paths` cell (under `TETHER/plots/…`). Colors
and display names come from `method_color_dict` / `sample_color_map` / `sample_rename_map`.

| Section (search header) | Function | Figure |
|---|---|---|
| `### Generalizability summary figure` | `plot_performance_across_evaluation_sets` | mean metric across own / same-cell / diff-cell / diff-organism evaluation tiers |
| `### Own test set ROC - one graph per sample` | `plot_model_vs_test_set_auroc_curves` | grid of per-sample ROC curves (model vs its own test set) |
| `### Own test set ROC - All samples one fig` | `plot_model_vs_test_set_auroc_combined` | one ROC axis, one curve per sample |
| `### Own test set PRC - All samples one fig` | `plot_model_vs_test_set_auprc` | one PRC axis, one curve per sample |
| `### Own test set AUPRC Lift - All samples in one fig` | `plot_model_vs_test_set_auprc_lift` | precision-lift-vs-recall curves + ratio-lift legend |
| `### Metric performance by method boxplots` | `plot_method_box_and_whisker` | per-metric boxplots (auroc/auprc/accuracy/early_precision/precision/recall/f1) across samples, by method |
| `### ROC curves` | `plot_sample_roc_curves` | model-vs-baseline-method ROC, per sample |
| `### PRC Curves` | `plot_sample_prc_curves` | model-vs-baseline-method PRC, per sample |
| `### PRC lift curves` | (uses PRC + random baseline) | precision-over-random lift curves per sample |
| `### AUC and PRC Lift` | `lift_by_method_boxplot` | AUROC/AUPRC lift boxplots by method |
| `### Rank Plots` | `avg_rank_by_method_plot`, `avg_rank_by_method_lollipop_plot`, `experiment_by_method_rank_heatmap`, `rank_by_method_boxplot` | method-ranking as bar / lollipop / (sample×method) heatmap / boxplot |
| `## GIF of True/False histogram across training` | (inline) | animation of the true/false score histogram over training epochs |
| `## Evaluate TF-DNA Model Performance` | `tf_dna_binding_roc_plot`, `tf_dna_binding_prc_plot` | stage-1 TF–DNA binding ROC/PRC |
| `### Network size box and whisker plots` | `plot_grn_size_boxplot` | GRN size (edges / unique TFs / TGs) boxplots by method |
| `### Network size jitter plots` | `plot_grn_size_jitter` | GRN size jitter/strip plots by method |
| `### Percent of edge combination box and whisker plots` | `add_percent_edges_vs_own` + `plot_grn_size_boxplot` | edge count as % of own-model GRN, by method |
| `## Stability` | `plot_method_stability_by_sample_boxplot` | pairwise top-10%-edge Jaccard boxplots per sample (10 subsamples) |
| `### Other Inference Method Stability` | `label_df` + Jaccard | stability boxplots extended to the baseline methods |
| `## Feature ablation` (mESC / Hepatocyte / K562 / Macrophage) | `plot_feature_ablation_boxplot` | metric comparison across `simplified_models/` ablations (no_binding / no_expr_info / no_peak_info / no_peak_tg_distance vs full) |

Figures are saved with `fig.savefig(..., dpi=300, bbox_inches="tight")` to the corresponding
`plots/…` directory from the `## Paths` cell.
