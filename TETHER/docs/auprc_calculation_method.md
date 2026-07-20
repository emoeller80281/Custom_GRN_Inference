# AUPRC Calculation Method

Method outline for `plot_auprc_all_methods.py` — evaluates the TF→TG regulation
model (on its own held-out test set, and cross-trained on another sample) against
external GRN inference methods (SCENIC+, LINGER, CellOracle, Pando, FigR) using a
chromosome-held-out ground truth test set, and plots precision-recall curves with
AUPRC.

Run as: `python plot_auprc_all_methods.py --species <mm10|hg38> --cell_type <...>
--sample_name <...> --cross_model_cell_type <...> --cross_model_sample_name <...>
[--force_reload]`

## 1. Inputs

| Input | Source | Notes |
|---|---|---|
| ATAC pseudobulk | `sample_input_data_dir/RE_pseudobulk.parquet` | filtered to peaks on `test_chroms` only |
| RNA pseudobulk | `sample_input_data_dir/TG_pseudobulk.parquet` | gene index upper-cased |
| Peak→gene distance | `sample_input_data_dir/peak_to_gene_dist.parquet` | TSS distance per peak/target pair |
| Merged ground truth | `cached_data/{cell_type}_cache/{cell_type}_merged_ground_truth.parquet` | pre-built upstream by `scripts/build_tf_to_tg_train_data.py` (`utils.load_ground_truth_files`); this script only reads the cache |
| TF name → index | `cached_data/{cell_type}_cache/tf_name_to_idx.csv` | defines which TFs have TF-DNA embeddings |
| TF-TG training metadata | `cached_data/{cell_type}_cache/tf_tg_training_cache/{sample_name}/metadata.json` | `tf_name_to_idx`, `tg_id_to_idx` used for model inference |
| Gene reference GTF | `genome_annotation/{species}/*.gtf.gz` | used only to assign genes to chromosomes for the train/val/test split |
| Genome FASTA + chrom.sizes | `reference_genome/{species}/` | used to one-hot encode peak sequences |
| TF embeddings / masks | `cached_data/{cell_type}_cache/tf_embeddings.pt`, `tf_masks.pt` | precomputed TF-DNA embedding lookup tensors |
| Model checkpoints | `TF_TG_MODEL_CHECKPOINTS[cell_type][sample_name]` (own), `[cross_model_cell_type][cross_model_sample_name]` (cross), `config.tf_dna_model_checkpoints[cell_type]` (frozen TF-DNA submodule) | dict literal in the script, `plot_auprc_all_methods.py:60-88` |
| Baseline method GRNs | `OTHER_METHOD_MUON_DIR/{Method}_muon/{method}_{cell_type}_{sample_name}.tsv` | SCENIC+, LINGER, CellOracle, Pando, FigR |

## 2. Pipeline

### Stage 1 — Load sample data (`:324-344`)
Load ATAC pseudobulk and restrict to peaks on `test_chroms` (species-specific,
see Stage 3). Load RNA pseudobulk with an upper-cased gene index. Load
peak→gene distances. `common_cells` = sorted intersection of RNA and ATAC
pseudobulk columns.

### Stage 2 — Filter ground truth to what the model can score (`:346-389`)
1. Load `{cell_type}_merged_ground_truth.parquet`.
2. Keep only edges whose Source (TF) and Target (TG) are both present in the
   RNA pseudobulk index.
3. Load `tf_name_to_idx.csv` and keep only edges whose TF also has a
   precomputed TF-DNA embedding (i.e. was seen during TF-DNA model training).
   This second filter is TF-only — it does not re-check the target gene.

### Stage 3 — Chromosome-based train/val/test split (`:391-417`)
- `split_genes_by_chromosome` assigns every gene in the GTF to train/val/test
  by chromosome:
  - mm10: train = chr 1-15, val = chr 16-17, test = chr 18-19
  - hg38: train = chr 1-17, val = chr 18-19, test = chr 20-22
- `create_train_val_test_splits` buckets the filtered ground truth by the
  **target gene's** chromosome only (TF chromosome is irrelevant to the
  split).
- `gt_test_df` is upper-cased and reduced to `gt_tfs`, `gt_tgs`, `gt_pairs`
  (`Source\tTarget` strings) — this defines `gt_lookup`, the ground truth
  universe used everywhere downstream.

### Stage 4 — Build the candidate universe and sample negatives (`:419-438`)
`full_universe` = every `(TF, TG)` combination from `gt_tfs × gt_tgs`
(`pd.MultiIndex.from_product`), labeled `_in_gt` by membership in `gt_pairs`.

`sample_auprc_10x_negatives(full_universe, random_state=42)` is then called
**immediately**, before any inference happens: it keeps all positives, samples
negatives at 10x the positive count (`replace=False`, or all negatives if
fewer than 10x exist), and shuffles — producing `full_universe_10x_negatives`.
`true_df`/`false_df` (and the `true_interactions`/`false_interactions`
generators built from them) come from this **sampled** universe, not the
dense `gt_tfs × gt_tgs` grid — so Stage 5's inference only scores the edges
actually needed for evaluation. Because the sample only depends on
`full_universe`/ground truth (not on any method's scores) and uses a fixed
`random_state`, it is also the exact same negative sample reused for every
baseline method in Stage 7 — computed once per run, not per method.

### Stage 5 — Generate own-model and cross-model predictions (`:440-598`)
Gated entirely by cache: if both
`testing_results/full_test_grns/{cell_type}/{sample_name}/tf_tg_predictions_{cell_type}_{sample_name}.tsv`
and `tf_tg_cross_model_predictions_{cross_model_cell_type}_{cross_model_sample_name}.tsv`
already exist and `--force_reload` is not set, this entire stage is skipped and both are
read straight from disk. Otherwise:

1. `convert_labeled_dataframe_to_indices` maps the sampled `true_interactions`/
   `false_interactions` TF/TG names to `tf_name_to_idx`/`tg_id_to_idx` indices
   (from the training metadata), and drops any pair without a mapping.
2. `utils.create_centered_peak_onehot_array` one-hot encodes a ±128bp window
   (flank_size=128) around every test-chromosome peak from the genome FASTA
   (10 parallel workers, chunk size 10,000). This is sized by test-chromosome
   peak count, not by the sampled edge count, so it doesn't shrink with Stage 4's sampling.
3. `utils.prepare_tftg_lookup_tables` finds, for every target gene, all peaks
   within 100kb of its TSS (capped to the 8 closest, `max_precompute_peaks=8`),
   sorted by distance; also builds dense `atac_mat`/`rna_mat` matrices indexed
   by `common_cells`.
4. `utils.build_tftg_inputs` assembles one compact "edge bag" per labeled
   TF-TG pair: up to `max_peaks_per_tg=8` nearby peaks (padded/masked to the
   observed max), `max_cells_per_pair=16` sampled cells, per-cell peak
   accessibility, and TF/TG expression (`seed=125`).
5. `tf_embeddings.pt` / `tf_masks.pt` are loaded as the TF-DNA embedding
   lookup tensors.
6. `TFTGEdgeBagDataset` wraps the compact inputs; peak DNA sequences are
   fetched on the fly from the one-hot tensor via `peak_indices` at
   `__getitem__` time (not stored per-edge). `DataLoader(batch_size=512,
   num_workers=8, collate_fn=collate_tftg_edge_bags)`.
7. `utils.load_tf_tg_regulation_model` reconstructs the frozen TF-DNA binding
   submodule + TF-TG regulation head from checkpoints (stripping any
   `torch.compile` `_orig_mod.` prefixes), optionally `torch.compile`s it.
   Whether the **own** model and the **cross** model are actually loaded and
   run is each independently gated by whether *their own* cache file exists
   — so if only one of the two prediction files is missing, only that model
   runs, even though the shared setup in steps 1-6 above always runs once
   both files aren't already present.
8. `generate_model_predictions` runs the forward pass per batch (bf16
   autocast on CUDA, `pooling_mode="lse"`, `pooling_temperature=1.0`),
   sigmoids the edge logits to `[0,1]` scores, then **aggregates duplicate
   `(Source, Target)` rows by median** (a TF-TG pair can appear more than once
   across sampled cell-bags) to produce `prediction_df`.
9. Both prediction DataFrames are cached to
   `testing_results/full_test_grns/{cell_type}/{sample_name}/*.tsv`.

The TF-TG regression model architecture (`models/tf_to_tg.py:14`,
`TFTGRegulationModel`): a frozen pretrained TF-DNA binding model scores each
(TF, peak) pair; those scores are projected together with accessibility and
distance features, attended over peaks per TF-TG-cell triple, then pooled
across the sampled cells (log-sum-exp by default — a soft-max-like pooling
controlled by `pooling_temperature`) to produce one logit per TF-TG edge.

### Stage 6 — Load and standardize baseline method GRNs (`:600-630`)
For SCENIC+, LINGER, CellOracle, Pando, FigR: `load_and_standardize_method`
reads each method's TSV/CSV, renames its TF/target/score columns to
`Source`/`Target`/`Score`, upper-cases names, and filters to edges where both
Source and Target are in `gt_tfs`/`gt_tgs`. These are only as dense as
whatever edges each external tool actually output. The own-model and
cross-model `prediction_df`s are added to the same `standardized_method_dfs`
dict under `"TF-TG Model (own test set)"` / `"TF-TG Model (cross-trained)"`.

### Stage 7 — Build the per-method labeled AUPRC set (`:632-664`)
For each method's standardized DataFrame:
- If `testing_results/labeled_auprc_grns/{cell_type}/{sample_name}/{method}.tsv`
  exists and `--force_reload` isn't set, load it directly.
- Otherwise: `create_ground_truth_comparison_df` subsets the method's
  scores to edges within `gt_tfs`/`gt_tgs` and labels `_in_gt` by exact
  `(Source, Target)` membership in `gt_pairs`; this is then left-merged
  onto `full_universe_10x_negatives[Source, Target, _in_gt]` (built in
  Stage 4) on `(Source, Target)`. **Any edge the method didn't score gets
  `Score = 0`** (`fillna(0)`) — a missing prediction is treated as maximal
  non-confidence, not dropped from the evaluation set. For the own/cross
  model this fill rarely triggers now, since Stage 4/5 already scored
  exactly this sampled edge set; for the sparser baseline methods it's the
  normal case.
- Cache to `testing_results/labeled_auprc_grns/{cell_type}/{sample_name}/{method}.tsv`.

This exact logic (full-universe construction + 10x negative sampling +
per-method score merge with 0-fill) has been condensed into
`calculate_auprc.create_labeled_auprc_grn(score_df, gt_df)` for reuse
outside this script.

### Stage 8 — Compute AUPRC and PR curve per method (`:700-758`)
For each method present in `method_color_dict` (`:668-676`: `TF-TG Model (own
test set)`, `TF-TG Model (cross-trained)`, `LINGER`, `CellOracle`, `Pando`,
`SCENIC+`, `FigR`, `GRaNIE`) — methods computed in Stage 7 but absent from
this dict are silently skipped here even though their labeled GRN was still
built and cached:
- Skip (without recording a metric) if the labeled set has fewer than 2
  distinct `_in_gt` classes.
- `average_precision_score(y, s)` → AUPRC; `precision_recall_curve(y, s)` →
  the plotted curve.
- Random baseline: `plotting_utils._create_random_distribution(s)` draws
  scores **uniformly between `s.min()` and `s.max()`** (not a permutation of
  `s`) with `seed=42`, then computes `rand_auprc`/`rand_prec`/`rand_rec` the
  same way.

### Stage 9 — Plot (`:681-855`)
Single 6×6 square `Axes`. Per method: a solid step precision-recall curve
(linewidth 3 for the two TF-TG model methods, 2 for baselines) plus a dashed,
semi-transparent random-baseline curve in the same color. A manual legend
(color swatch + `"{method} = {auprc:.3f}"`, built with matplotlib
`AnchoredOffsetbox`/`VPacker`/`HPacker`/`DrawingArea`) is sorted by descending
AUPRC and placed below the axes. Title comes from `sample_to_title_map`
(falls back to the raw `sample_name`). Saved to
`plots/auprc_plots/{sample_name}_auprc.png` at 300 dpi.

### Stage 10 — Save the metrics table (`:857-866`)
`auprc_metrics_df` (`sample_name`, `method`, `auprc`, `rand_auprc`) is
tab-separated and written to `plots/auprc_metrics/{sample_name}_auprc_metrics.tsv`.

## 3. Output artifacts

| Path | Contents |
|---|---|
| `testing_results/full_test_grns/{cell_type}/{sample_name}/tf_tg_predictions_{cell_type}_{sample_name}.tsv` | own-model scores over the Stage 4 sampled TF×TG evaluation set |
| `testing_results/full_test_grns/{cell_type}/{sample_name}/tf_tg_cross_model_predictions_{cross_model_cell_type}_{cross_model_sample_name}.tsv` | cross-trained-model scores over the same sampled set |
| `testing_results/labeled_auprc_grns/{cell_type}/{sample_name}/{method}.tsv` | per-method labeled `(Source, Target, Score, _in_gt)` evaluation set (shared negative sample) |
| `plots/auprc_plots/{sample_name}_auprc.png` | combined precision-recall plot |
| `plots/auprc_metrics/{sample_name}_auprc_metrics.tsv` | AUPRC / random-AUPRC per method |

## 4. Key functions

| Function | File | Purpose |
|---|---|---|
| `generate_model_predictions` | `plot_auprc_all_methods.py:90` | run TF-TG model over a DataLoader, sigmoid + median-aggregate to `Source/Target/Score` |
| `create_ground_truth_comparison_df` | `plot_auprc_all_methods.py:152` | subset a method's scores to the GT TF/TG universe and label `_in_gt` |
| `load_and_standardize_method` | `plot_auprc_all_methods.py:177` | normalize an external method's GRN file to `Source/Target/Score` |
| `convert_labeled_dataframe_to_indices` | `plot_auprc_all_methods.py:211` | map TF/TG names to model input indices, dropping unmapped pairs |
| `sample_auprc_10x_negatives` | `plot_auprc_all_methods.py:234` | 10x-negative-sampled AUPRC evaluation universe from the full GT grid; now called before inference (Stage 4), not after |
| `split_genes_by_chromosome` | `scripts/build_tf_to_tg_train_data.py:24` | assign genes to train/val/test by chromosome |
| `create_train_val_test_splits` | `scripts/build_tf_to_tg_train_data.py:55` | bucket GT edges by target gene's split |
| `utils.create_centered_peak_onehot_array` | `utils.py:303` | one-hot encode DNA windows around peaks |
| `utils.prepare_tftg_lookup_tables` | `utils.py:1488` | per-TG nearby-peak lookup + dense ATAC/RNA matrices |
| `utils.build_tftg_inputs` | `utils.py:1552` | assemble compact per-edge model input tensors |
| `utils.load_tf_tg_regulation_model` | `utils.py:1055` | reconstruct TF-DNA + TF-TG models from checkpoints |
| `TFTGEdgeBagDataset` / `collate_tftg_edge_bags` | `scripts/train_tf_to_tg_model.py:125,170` | dataset/collation for edge-bag inference |
| `TFTGRegulationModel.forward` | `models/tf_to_tg.py:109` | bag-level TF→TG regulation forward pass |
| `plotting_utils._create_random_distribution` | `plotting_utils.py:14` | uniform random-score baseline for AUPRC comparison |
| `calculate_auprc.create_labeled_auprc_grn` | `calculate_auprc.py` | reusable condensation of Stage 4 (universe + sampling) + Stage 7 (per-method merge) |

## 5. Non-obvious details / caveats

- **Inference runs on the sampled evaluation set, not the dense grid**:
  `sample_auprc_10x_negatives` was moved up to Stage 4 (`:419-438`) so
  `true_df`/`false_df` — and therefore everything Stage 5 builds and scores
  — come from the 10x-negative-sampled universe, not the full
  `gt_tfs × gt_tgs` grid. The one-hot peak encoding step (Stage 5.2) is
  unaffected by this, since it's sized by test-chromosome peak count, not
  by the number of TF-TG edges.
- **Same random seed reused for two different things**: `random_state=42` /
  `seed=42` is used both for the 10x-negative sample (`sample_auprc_10x_negatives`)
  and for the uniform random-baseline scores (`_create_random_distribution`).
  These are independent RNG draws (different call sites), not the same
  random sequence reused.
- Caching is keyed only by `cell_type`/`sample_name` (and cross-model names)
  — upstream data or ground-truth changes won't be picked up unless
  `--force_reload` is passed or the cached files are deleted manually. This
  now also means: if a cached `full_test_grns` prediction file from before
  the Stage 4 sampling change is reused, it may still reflect the old dense
  grid — delete stale caches (or pass `--force_reload`) after this change to
  regenerate them against the sampled universe.
- All name matching (TF, TG, method columns) is done on **upper-cased**
  strings throughout, to avoid case-mismatch join failures between sources.
- The cross-trained model is evaluated on `cell_type`/`sample_name`'s test
  set — i.e. it reuses the *same* DataLoader/edge bags as the own model, just
  scored with a checkpoint trained on a different sample.
