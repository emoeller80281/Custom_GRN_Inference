"""Section: run a trained TF-TG model against held-out test sets.

Refactored from ``test_tf_tg_predictions.ipynb`` (unmodified). This module
defines one child class of :class:`.base.TFTGBase`; the plotting helpers are
methods and the notebook's driver cells live in :meth:`run`.
"""
from .base import *  # noqa: F401,F403  (config, shared funcs, notebook imports)
from .base import TFTGBase


class ModelVsTestSet(TFTGBase):
    """Section: run a trained TF-TG model against held-out test sets."""

    def run(self):
        """Execute this section end-to-end (data generation, plotting, saving)."""
        # all_comparison_df_list = []

        # evaluations = [
        #     ("mESC", "E7.5_rep1", "mESC", "E7.5_rep1"),
        #     # ("mESC", "E8.5_rep1", "mESC", "E8.5_rep1"),

        #     # ("mouse_hepatocytes", "hepatocytes_1", "mouse_hepatocytes", "hepatocytes_1"),
        #     # ("mouse_hepatocytes", "hepatocytes_3", "mouse_hepatocytes", "hepatocytes_3"),

        #     # ("Macrophage", "buffer_1", "Macrophage", "buffer_1"),
        #     # ("Macrophage", "buffer_2", "Macrophage", "buffer_2"),

        #     # ("K562", "sample_1", "K562", "sample_1")

        # ]

        # all_plot_data = {}

        # subset_size = 10_000
        # # for model_cell_type, model_training_sample, test_set_cell_type, evaluation_sample in tqdm(evaluations, desc="Evaluating all model vs test set combinations", ncols=100):
        # for model_cell_type, model_training_sample, test_set_cell_type, evaluation_sample in evaluations:
        #     logging.info(f"Evaluating {model_cell_type} {model_training_sample} Model → {test_set_cell_type} {evaluation_sample} Test Set")

        #     dataset_split_type = "test"

        #     comparison_result = run_prediction_vs_test_set(
        #         tf_tg_model_checkpoints=tf_tg_model_checkpoints,
        #         model_cell_type=model_cell_type,
        #         model_training_sample=model_training_sample,
        #         test_set_cell_type=test_set_cell_type,
        #         evaluation_sample=evaluation_sample,
        #         dataset_split_type=dataset_split_type,
        #         subset_size=subset_size,
        #         show_progress_bar=True,
        #         compile_model=True
        #     )

        #     metric_df = comparison_result["metric_df"]
        #     plot_data = comparison_result["plot_data"]

        #     all_labels_flat = plot_data[0]
        #     all_scores_flat = plot_data[1]

        #     title = comparison_result["title"]

        #     all_plot_data[title] = (all_labels_flat, all_scores_flat)

        #     all_comparison_df_list.append(metric_df)

        # full_comparison_df = pd.concat(all_comparison_df_list, ignore_index=True)

        # display(full_comparison_df.T)

        # for title in all_plot_data.keys():
        #     all_labels_flat = all_plot_data[title][0]
        #     all_scores_flat = all_plot_data[title][1]

        #     histogram_fig = plotting_utils.plot_score_histograms(
        #         labels=all_labels_flat,
        #         scores=all_scores_flat,
        #         n_bins=50,
        #         y_log=False,
        #         panel_kind="kde",
        #         density=False,
        #         title = title
        #     )
        #     histogram_fig.show()

        #     model_sample, test_set_sample = title.split(" → ")
        #     model_sample = model_sample.split("Model")[0].split()[1].strip()
        #     test_set_sample = test_set_sample.split("Test Set")[0].split()[1].strip()
        #     title = f"{model_sample} Model\n{test_set_sample} Test Set"

        # auroc_auprc_fig = plotting_utils.plot_auroc_auprc(
        #     labels=all_labels_flat,
        #     scores=all_scores_flat,
        #     title = title,
        #     plot_type = "roc"
        # )
        # auroc_auprc_fig.show()
        pass
