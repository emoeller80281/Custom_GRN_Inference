"""Section: model generalizability across samples, cell types, and organisms.

Refactored from ``test_tf_tg_predictions.ipynb`` (unmodified). This module
defines one child class of :class:`.base.TFTGBase`; the plotting helpers are
methods and the notebook's driver cells live in :meth:`run`.
"""
from .base import *  # noqa: F401,F403  (config, shared funcs, notebook imports)
from .base import TFTGBase

# Section-level constants (referenced by the methods below)
celltype_order = ["mESC", "mouse_hepatocytes", "Macrophage", "K562", "iPSC"]


class Generalizability(TFTGBase):
    """Section: model generalizability across samples, cell types, and organisms."""

    def agg_results(self, df):
        df_grouped = df.groupby(["Model Cell Type"]).agg(
            auroc_mean=("auroc", "mean"),
            auprc_mean=("auprc", "mean"),
            auprc_lift_mean=("auprc_lift", "mean"),
            accuracy_mean=("accuracy", "mean"),
            precision_mean=("precision", "mean"),
            early_precision_mean=("early_precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean")
        ).reset_index()

        df_grouped = df_grouped.sort_values("Model Cell Type", key=lambda x: x.map({celltype: i for i, celltype in enumerate(celltype_order)}))

        # Round to 3 decimal places for better readability
        df_grouped = df_grouped.round(3)

        return df_grouped

    def plot_performance_across_evaluation_sets(self, summary_df, metrics, evaluation_order, save_path):
        # NOTE(refactor): in the notebook `x` was a module-global set by a glue cell
        # (`x = np.arange(len(evaluation_order))`). It is computed here so the method
        # is self-contained; behaviour is identical.
        x = np.arange(len(evaluation_order))
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

        for metric_col, label in metrics.items():
            ax.plot(
                x,
                summary_df[metric_col],
                marker="o",
                markersize=9,
                linewidth=3,
                label=label
            )

        ax.set_xticks(x)
        ax.set_xticklabels(evaluation_order, rotation=20, ha="right")

        # Pad x-axis so labels/points do not hit the edge
        ax.set_xlim(-0.5, len(evaluation_order) - 0.5)

        ax.set_title("Mean Performance Across\nDifferent Evaluation Sets", pad=20)
        # ax.set_xlabel("Evaluation Set", labelpad=12)
        ax.set_ylabel("Score", labelpad=12)
        ax.set_ylim(0, 1)

        ax.legend(
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0
        )

        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        return fig, ax

    def plot_model_vs_test_set_auroc_curves(self, models_to_plot, auroc_models_vs_test_set_individual_plot_path):
        fig, ax = plt.subplots(1, 7, figsize=(18, 6))

        for i, sample_name in enumerate(models_to_plot):

            ax[i].set_aspect("equal")

            score_label_df = load_generalizability_df(sample_name, sample_name)
            scores = score_label_df["Score"].values
            labels = score_label_df["Label"].values

            rng = np.random.default_rng(42)
            rand_scores = rng.permutation(scores)

            fpr, tpr, thresholds = roc_curve(labels, scores)
            rand_fpr, rand_tpr, rand_thresholds = roc_curve(labels, rand_scores)
            auroc = roc_auc_score(labels, scores)

            ax[i].plot(
                fpr,
                tpr,
                lw=4,
                color="#4195df",
                label=f"AUROC = {auroc:.3f}",
                zorder=2,
            )

            ax[i].plot(
                rand_fpr,
                rand_tpr,
                color="black",
                linestyle=":",
                lw=2,
                alpha=0.6,
                zorder=1,
            )

            ax[i].set_title(
                sample_rename_map.get(sample_name, sample_name),
                fontsize=16,
            )

            ax[i].set_xlim(0, 1)
            ax[i].set_ylim(0, 1)

            ax[i].text(
                0.15,
                0.05,
                f"AUROC = {auroc:.3f}",
                transform=ax[i].transAxes,
                fontsize=16,
                bbox=dict(facecolor="none", edgecolor="none"),
            )

            ax[i].tick_params(bottom=False, left=False)
            ax[i].set_xticklabels([])
            ax[i].set_yticklabels([])

        fig.suptitle("Model vs Test Set AUROC Curves", fontsize=25, y=0.75)

        fig.text(
            0.5,
            0.18,
            "False Positive Rate",
            ha="center",
            fontsize=20,
        )

        fig.text(
            0.00,
            0.42,
            "True Positive Rate",
            va="center",
            rotation="vertical",
            fontsize=20,
        )

        fig.subplots_adjust(
            left=0.02,
            right=0.99,
            bottom=0.01,
            top=0.85,
            wspace=0.08,
        )

        return fig, ax

    def plot_model_vs_test_set_auroc_combined(self, models_to_plot, auroc_models_vs_test_set_all_samples_plot_path):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_aspect("equal")

        rng = np.random.default_rng(42)
        random_curve_plotted = False

        for sample_name in models_to_plot:

            score_label_df = load_generalizability_df(sample_name, sample_name)
            scores = score_label_df["Score"].values
            labels = score_label_df["Label"].values

            labels = pd.to_numeric(pd.Series(labels), errors="coerce").to_numpy()
            scores = pd.to_numeric(pd.Series(scores), errors="coerce").to_numpy()

            valid_mask = ~np.isnan(labels) & ~np.isnan(scores)
            labels = labels[valid_mask].astype(int)
            scores = scores[valid_mask]

            fpr, tpr, _ = roc_curve(labels, scores)
            auroc = roc_auc_score(labels, scores)

            ax.plot(
                fpr,
                tpr,
                lw=3,
                color=sample_color_map.get(sample_name, None),
                label=f"{sample_rename_map.get(sample_name, sample_name)} = {auroc:.3f}",
            )

            # Plot one shuffled/random baseline only
            if not random_curve_plotted:
                rand_scores = rng.permutation(scores)
                rand_fpr, rand_tpr, _ = roc_curve(labels, rand_scores)

                ax.plot(
                    rand_fpr,
                    rand_tpr,
                    color="black",
                    linestyle="--",
                    lw=2,
                    alpha=0.6,
                    zorder=1,
                )

                random_curve_plotted = True


        ax.set_title("AUROC", fontsize=30)
        ax.set_xlabel("False Positive Rate", fontsize=20)
        ax.set_ylabel("True Positive Rate", fontsize=20)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.tick_params(axis="both", labelsize=16)

        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=20,
        )

        fig.subplots_adjust(
            left=0.10,
            right=0.72,
            bottom=0.10,
            top=0.90,
        )

        return fig, ax

    def plot_model_vs_test_set_auprc(self, auprc_all_method_dfs, models_to_plot, tf_tg_method_name):
        fig, ax = plt.subplots(figsize=(7, 6))

        rng = np.random.default_rng(42)

        def downsample_curve(x, y, max_points=500):
            x = np.asarray(x)
            y = np.asarray(y)

            if len(x) <= max_points:
                return x, y

            idx = np.linspace(0, len(x) - 1, max_points, dtype=int)
            idx = np.unique(idx)

            return x[idx], y[idx]

        for sample_name in models_to_plot:
            if sample_name not in auprc_all_method_dfs:
                logging.warning(
                    f"Sample {sample_name} was not found in labeled AUPRC files. Skipping."
                )
                continue

            if tf_tg_method_name not in auprc_all_method_dfs[sample_name]:
                logging.warning(
                    f"Method {tf_tg_method_name!r} not found for sample {sample_name}. "
                    f"Available methods: {sorted(auprc_all_method_dfs[sample_name].keys())}"
                )
                continue

            auprc_df = auprc_all_method_dfs[sample_name][tf_tg_method_name].copy()

            labels = pd.to_numeric(auprc_df["_in_gt"], errors="coerce").to_numpy()
            scores = pd.to_numeric(auprc_df["Score"], errors="coerce").to_numpy()

            valid_mask = ~np.isnan(labels) & ~np.isnan(scores)
            labels = labels[valid_mask].astype(int)
            scores = scores[valid_mask]

            if len(np.unique(labels)) < 2:
                logging.warning(
                    f"Sample {sample_name} has only one label class after filtering. Skipping."
                )
                continue

            precision, recall, _ = precision_recall_curve(labels, scores)
            auprc = average_precision_score(labels, scores)

            rand_scores = rng.permutation(scores)
            rand_precision, rand_recall, _ = precision_recall_curve(labels, rand_scores)
            rand_auprc = average_precision_score(labels, rand_scores)

            # sklearn returns recall in descending order, so reverse for cleaner left-to-right plotting
            plot_recall = recall[::-1]
            plot_precision = precision[::-1]

            plot_recall, plot_precision = downsample_curve(plot_recall, plot_precision)

            ax.step(
                plot_recall,
                plot_precision,
                where="post",
                lw=3,
                color=sample_color_map.get(sample_name, None),
                label=f"{sample_rename_map.get(sample_name, sample_name)} = {auprc:.3f}\n  (Random: {rand_auprc:.3f})",
                zorder=3,
            )

            # if not random_curve_plotted:
            ax.step(
                rand_recall[::-1],
                rand_precision[::-1],
                where="post",
                linestyle="--",
                lw=1,
                alpha=0.4,
                color=sample_color_map.get(sample_name, None),
                zorder=1,
            )

                # random_curve_plotted = True

        ax.set_title("AUPRC", fontsize=30)
        ax.set_xlabel("Recall", fontsize=20)
        ax.set_ylabel("Precision", fontsize=20)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")

        ax.tick_params(axis="both", labelsize=16)

        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=20,
        )

        fig.subplots_adjust(
            left=0.10,
            right=0.72,
            bottom=0.10,
            top=0.90,
        )

        return fig, ax

    def plot_model_vs_test_set_auprc_lift(self, auprc_all_method_dfs, models_to_plot, tf_tg_method_name):
        fig, ax = plt.subplots(figsize=(7, 6))

        ax.set_box_aspect(1)

        rng = np.random.default_rng(42)

        cell_type_method_auprc_lift = {
            "sample_name": [],
            "method": [],
            "auprc": [],
            "rand_auprc": [],
            "auprc_lift": [],
            "curve_lift_auc": [],
        }

        recall_grid = np.linspace(0, 1, 1000)

        for sample_name in models_to_plot:

            if sample_name not in auprc_all_method_dfs:
                logging.warning(f"Sample {sample_name} not found in auprc_all_method_dfs. Skipping.")
                continue

            if tf_tg_method_name not in auprc_all_method_dfs[sample_name]:
                logging.warning(
                    f"Method {tf_tg_method_name} not found for sample {sample_name}. Skipping."
                )
                continue

            auprc_df = auprc_all_method_dfs[sample_name][tf_tg_method_name].copy()

            labels = pd.to_numeric(auprc_df["_in_gt"], errors="coerce").to_numpy()
            scores = pd.to_numeric(auprc_df["Score"], errors="coerce").to_numpy()

            valid_mask = ~np.isnan(labels) & ~np.isnan(scores)
            labels = labels[valid_mask].astype(int)
            scores = scores[valid_mask]

            if len(np.unique(labels)) < 2:
                logging.warning(
                    f"Sample {sample_name} has only one label class after filtering. Skipping."
                )
                continue

            precision, recall, _ = precision_recall_curve(labels, scores)
            auprc = average_precision_score(labels, scores)

            rand_scores = rng.permutation(scores)
            rand_precision, rand_recall, _ = precision_recall_curve(labels, rand_scores)
            rand_auprc = average_precision_score(labels, rand_scores)

            # precision_recall_curve returns recall in descending order,
            # so reverse before interpolation.
            precision_interp = np.interp(
                recall_grid,
                recall[::-1],
                precision[::-1],
            )

            rand_precision_interp = np.interp(
                recall_grid,
                rand_recall[::-1],
                rand_precision[::-1],
            )

            precision_lift = precision_interp / rand_precision_interp

            rand_auprc_lift_baseline = 1.0  # The baseline for random AUPRC lift is always 1.0

            auprc_lift = auprc / rand_auprc
            curve_lift_auc = auc(recall_grid, precision_lift)

            cell_type_method_auprc_lift["sample_name"].append(sample_name)
            cell_type_method_auprc_lift["method"].append(tf_tg_method_name)
            cell_type_method_auprc_lift["auprc"].append(auprc)
            cell_type_method_auprc_lift["rand_auprc"].append(rand_auprc)
            cell_type_method_auprc_lift["auprc_lift"].append(auprc_lift)
            cell_type_method_auprc_lift["curve_lift_auc"].append(curve_lift_auc)

            min_recall_to_plot = 0.01

            plot_mask = recall_grid >= min_recall_to_plot

            ax.plot(
                recall_grid[plot_mask],
                precision_lift[plot_mask],
                lw=3,
                color=sample_color_map.get(sample_name, None),
                label=(
                    f"{sample_rename_map.get(sample_name, sample_name)} "
                    f"= {auprc_lift:.3f}"
                ),
                zorder=3,
            )


        ax.axhline(
            rand_auprc_lift_baseline,
            color="black",
            linestyle="--",
            lw=2,
            alpha=0.6,
            zorder=1,
        )

        ax.set_title("AUPRC Lift", fontsize=30)
        ax.set_xlabel("Recall", fontsize=20)
        ax.set_ylabel("Precision / Baseline", fontsize=20)

        ax.set_xlim(0, 1)
        # ax.set_ylim(-1, 1)

        ax.tick_params(axis="both", labelsize=16)

        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=21,
        )

        fig.subplots_adjust(
            left=0.15,
            right=0.72,
            bottom=0.12,
            top=0.90,
        )

        return fig, ax

    def run(self):
        """Execute this section end-to-end (data generation, plotting, saving)."""
        all_comparison_dfs = []
        individual_comparison_dir = RESULT_DIR / "model_generalizability" / "comparison_metric_files"
        for comparison_file in individual_comparison_dir.glob("*10000.csv"):
            comparison_df = pd.read_csv(comparison_file)
            all_comparison_dfs.append(comparison_df)

        num_comparisons = len(all_comparison_dfs)
        print(f"Loaded {num_comparisons} comparisons")

        generalizability_df = pd.concat(all_comparison_dfs, ignore_index=True)
        generalizability_df.to_csv(RESULT_DIR / "model_generalizability" / "model_generalizability_results.csv", index=False)


        generalizability_df["Model Organism"] = generalizability_df["Model"].map(lambda x: org_dict.get(x, ("Unknown", "Unknown"))[0])
        generalizability_df["Model Cell Type"] = generalizability_df["Model"].map(lambda x: org_dict.get(x, ("Unknown", "Unknown"))[1])
        generalizability_df["Test Set Organism"] = generalizability_df["Test Set"].map(lambda x: org_dict.get(x, ("Unknown", "Unknown"))[0])
        generalizability_df["Test Set Cell Type"] = generalizability_df["Test Set"].map(lambda x: org_dict.get(x, ("Unknown", "Unknown"))[1])

        generalizability_df["auprc_lift"] = generalizability_df["auprc"] - generalizability_df["rand_auprc"]

        own_test_set = generalizability_df[generalizability_df["Model"] == generalizability_df["Test Set"]]
        same_cell_type = generalizability_df[(generalizability_df["Model Cell Type"] == generalizability_df["Test Set Cell Type"]) & (generalizability_df["Model"] != generalizability_df["Test Set"])]
        different_cell_type = generalizability_df[(generalizability_df["Model Cell Type"] != generalizability_df["Test Set Cell Type"]) & (generalizability_df["Model Organism"] == generalizability_df["Test Set Organism"])]
        different_organism = generalizability_df[generalizability_df["Model Organism"] != generalizability_df["Test Set Organism"]]


        own_test_set_group = self.agg_results(own_test_set)
        same_cell_type_group = self.agg_results(same_cell_type)
        different_cell_type_group = self.agg_results(different_cell_type)
        different_organism_group = self.agg_results(different_organism)

        own_test_set_group.to_csv(RESULT_DIR / "model_generalizability" / "own_test_set_metrics.csv", index=False)
        same_cell_type_group.to_csv(RESULT_DIR / "model_generalizability" / "different_sample_metrics.csv", index=False)
        different_cell_type_group.to_csv(RESULT_DIR / "model_generalizability" / "different_cell_type_metrics.csv", index=False)
        different_organism_group.to_csv(RESULT_DIR / "model_generalizability" / "different_organism_metrics.csv", index=False)

        evaluation_order = [
            "Own Test Set",
            "Other Samples",
            "Different Cell Type",
            "Different Organism"
        ]

        evaluation_sets = {
            "Own Test Set": own_test_set_group,
            "Other Samples": same_cell_type_group,
            "Different Cell Type": different_cell_type_group,
            "Different Organism": different_organism_group
        }

        metrics = {
            "auroc_mean": "AUROC",
            # "auprc_mean": "AUPRC",
            "accuracy_mean": "Accuracy",
            "precision_mean": "Precision",
            "recall_mean": "Recall",
            "f1_mean": "F1 Score"
        }

        summary_rows = []

        for eval_name in evaluation_order:
            df = evaluation_sets[eval_name]
            row = {"Evaluation Set": eval_name}

            for metric_col in metrics:
                row[metric_col] = df[metric_col].mean()

            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)

        x = np.arange(len(evaluation_order))

        # Poster-friendly global settings
        plt.rcParams.update({
            "font.size": 18,
            "axes.titlesize": 26,
            "axes.labelsize": 22,
            "xtick.labelsize": 16,
            "ytick.labelsize": 18,
            "legend.fontsize": 18,
        })


        fig, ax = self.plot_performance_across_evaluation_sets(summary_df, metrics, evaluation_order, performance_across_evaluation_sets_plot_path)

        fig.savefig(performance_across_evaluation_sets_plot_path, dpi=300, bbox_inches="tight")

        plt.show()


        all_auroc_curve_fig, all_auroc_curve_ax = self.plot_model_vs_test_set_auroc_curves(
            models_to_plot, auroc_models_vs_test_set_individual_plot_path
            )

        all_auroc_curve_fig.savefig(auroc_models_vs_test_set_individual_plot_path, dpi=300, bbox_inches="tight")

        plt.show()


        all_auroc_combined_fig, all_auroc_combined_ax = self.plot_model_vs_test_set_auroc_combined(
            models_to_plot, auroc_models_vs_test_set_all_samples_plot_path
            )

        all_auroc_combined_fig.savefig(
            auroc_models_vs_test_set_all_samples_plot_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()


        auprc_all_method_dfs = load_auprc_grns_all_methods(sample_list=models_to_plot)

        auprc_combined_vs_test_set_fig, auprc_combined_vs_test_set_ax = self.plot_model_vs_test_set_auprc(
            auprc_all_method_dfs, models_to_plot, OWN_MODEL_METHOD
            )

        fig.savefig(
            auprc_models_vs_test_set_all_samples_plot_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()


        auprc_lift_combined_vs_test_set_fig, auprc_lift_combined_vs_test_set_ax = self.plot_model_vs_test_set_auprc_lift(
            auprc_all_method_dfs, models_to_plot, OWN_MODEL_METHOD
            )

        auprc_lift_combined_vs_test_set_fig.savefig(
            auprc_lift_models_vs_test_set_all_samples_plot_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()
