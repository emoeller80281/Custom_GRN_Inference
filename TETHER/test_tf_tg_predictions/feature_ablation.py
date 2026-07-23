"""Section: feature-ablation performance comparison.

Refactored from ``test_tf_tg_predictions.ipynb`` (unmodified). This module
defines one child class of :class:`.base.TFTGBase`; the plotting helpers are
methods and the notebook's driver cells live in :meth:`run`.
"""
from .base import *  # noqa: F401,F403  (config, shared funcs, notebook imports)
from .base import TFTGBase

# Section-specific imports (hoisted from the notebook cells)
import pandas as pd
import wandb


class FeatureAblation(TFTGBase):
    """Section: feature-ablation performance comparison."""

    def plot_feature_ablation_boxplot(self, full_run_df):
        """
        Create a box plot of validation AUROC for different model variants.
        """

        fig = plt.figure(figsize=(8, 6))

        sns.boxplot(
            data=full_run_df,
            x="model_variant",
            y="val/auroc",
            order=["normal", "no_peak_tg_distance", "no_expr_info", "no_tf_dna_binding", "no_peak_info"],
        )

        xtick_name_map = {
            "normal": "Full Model",
            "no_peak_tg_distance": "No Peak-TG Distance",
            "no_expr_info": "No Expression Info",
            "no_tf_dna_binding": "No TF-DNA Binding",
            "no_peak_info": "No Peak Info",
        }

        # Set the x-tick labels to the more descriptive names
        plt.xticks(
            ticks=range(len(xtick_name_map)), 
            labels=[
                xtick_name_map[x] for x in [
                    "normal", 
                    "no_peak_tg_distance", 
                    "no_expr_info", 
                    "no_tf_dna_binding", 
                    "no_peak_info"]
                ], 
            rotation=45, 
            ha="right", 
            fontsize=12
            )

        plt.title("Feature Ablation\nValidation AUROC", fontsize=18)
        plt.ylabel("Validation AUROC", fontsize=16)
        plt.xlabel("Model Variant", fontsize=16)
        plt.xticks(rotation=45, ha="right", fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim(0.5, 1.0)

        return fig

    def run(self):
        """Execute this section end-to-end (data generation, plotting, saving)."""
        api = wandb.Api()

        # Project is specified by <entity/project-name>
        runs = api.runs("luminarada-penn-state-health/tf_tg_feature_ablation")

        summary_list, config_list, name_list = [], [], []
        for run in runs:
            # .summary contains the output keys/values for metrics like accuracy.
            #  We call ._json_dict to omit large files
            summary_list.append(run.summary._json_dict)

            # .config contains the hyperparameters.
            #  We remove special values that start with _.
            config_list.append(
                {k: v for k,v in run.config.items()
                  if not k.startswith('_')})

            # .name is the human-readable name of the run.
            name_list.append(run.name)

        runs_df = pd.DataFrame({
            "summary": summary_list,
            "config": config_list,
            "name": name_list
            })

        run_summary_df = runs_df["summary"].apply(pd.Series)
        run_summary_df["name"] = runs_df["name"]

        run_config_df = runs_df["config"].apply(pd.Series)

        # Add the run config columns to the summary dataframe
        full_run_df = pd.concat([run_summary_df, run_config_df], axis=1)
        full_run_df.to_csv(RESULT_DIR / "feature_ablation_wandb_data.csv", index=False)

        full_run_df


        fig = self.plot_feature_ablation_boxplot(full_run_df)
        fig.savefig(feature_ablation_plot_dir / "feature_ablation_boxplot.png", dpi=300, bbox_inches="tight", facecolor="white")

        mESC_full_run_df = full_run_df[full_run_df["cell_type"] == "mESC"].copy()

        mESC_feature_ablation_fig = self.plot_feature_ablation_boxplot(mESC_full_run_df)
        mESC_feature_ablation_fig.savefig(feature_ablation_plot_dir / "mESC_feature_ablation_boxplot.png", dpi=300, bbox_inches="tight", facecolor="white")

        hepatocytes_full_run_df = full_run_df[full_run_df["cell_type"] == "mouse_hepatocytes"].copy()

        hepatocytes_feature_ablation_fig = self.plot_feature_ablation_boxplot(hepatocytes_full_run_df)
        hepatocytes_feature_ablation_fig.savefig(feature_ablation_plot_dir / "hepatocytes_feature_ablation_boxplot.png", dpi=300, bbox_inches="tight", facecolor="white")

        K562_full_run_df = full_run_df[full_run_df["cell_type"] == "K562"].copy()

        K562_feature_ablation_fig = self.plot_feature_ablation_boxplot(K562_full_run_df)
        K562_feature_ablation_fig.savefig(feature_ablation_plot_dir / "K562_feature_ablation_boxplot.png", dpi=300, bbox_inches="tight", facecolor="white")

        macrophage_full_run_df = full_run_df[full_run_df["cell_type"] == "Macrophage"].copy()

        macrophage_feature_ablation_fig = self.plot_feature_ablation_boxplot(macrophage_full_run_df)
        macrophage_feature_ablation_fig.savefig(feature_ablation_plot_dir / "macrophage_feature_ablation_boxplot.png", dpi=300, bbox_inches="tight", facecolor="white")
