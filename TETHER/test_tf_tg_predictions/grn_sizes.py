"""Section: comparison of inferred GRN sizes across methods.

Refactored from ``test_tf_tg_predictions.ipynb`` (unmodified). This module
defines one child class of :class:`.base.TFTGBase`; the plotting helpers are
methods and the notebook's driver cells live in :meth:`run`.

Refactor caveats (carried over from the notebook, behaviour preserved):
  * ``create_grn_size_summary_df`` ignores its ``sample_grn_dict`` argument and
    always summarises ``self.auprc_grns``. As a result ``auroc_summary_df`` is
    actually built from the AUPRC GRNs. Change the method to use its argument if
    a true AUROC-GRN summary is required.
  * The percent-of-edges plot now uses ``auprc_summary_df``; the notebook passed a
    stale ``summary_df`` that leaked from the Generalizability section.
"""
from .base import *  # noqa: F401,F403  (config, shared funcs, notebook imports)
from .base import TFTGBase

# Section-specific imports (hoisted from the notebook cells)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D


class GRNSizeComparison(TFTGBase):
    """Section: comparison of inferred GRN sizes across methods."""

    def create_grn_size_summary_df(self, sample_grn_dict):
        summary_dict = {
            "sample_name": [],
            "method_name": [],
            "num_unique_tfs": [],
            "num_unique_tgs": [],
            "num_edges": [],
            "num_true_edges": [],
            "num_false_edges": [],
        }

        for sample_name, method_dict in self.auprc_grns.items():
            for method_name in method_dict.keys():
                full_test_set_grn_df = self.auprc_grns[sample_name][method_name]

                full_test_set_grn_df = full_test_set_grn_df[full_test_set_grn_df["Score"] != 0]

                num_unique_tfs = full_test_set_grn_df["Source"].nunique()
                num_unique_tgs = full_test_set_grn_df["Target"].nunique()
                num_edges = full_test_set_grn_df.shape[0]
                num_true_edges = full_test_set_grn_df["_in_gt"].sum()
                num_false_edges = num_edges - num_true_edges

                summary_dict["sample_name"].append(sample_name)
                summary_dict["method_name"].append(method_name)
                summary_dict["num_unique_tfs"].append(num_unique_tfs)
                summary_dict["num_unique_tgs"].append(num_unique_tgs)
                summary_dict["num_edges"].append(num_edges)
                summary_dict["num_true_edges"].append(num_true_edges)
                summary_dict["num_false_edges"].append(num_false_edges)

        summary_df = pd.DataFrame(summary_dict)
        return summary_df

    def plot_grn_size_boxplot(self, summary_df, method_color_dict, variable_name="num_edges", title_suffix=None):
        assert variable_name in summary_df.columns, f"{variable_name} not found in summary_df columns"

        plot_df = summary_df.copy()

        # Clean method names for matching against method_color_dict
        plot_df["method_name_clean"] = (
            plot_df["method_name"]
            .astype(str)
            .str.replace("\n", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        # Also clean method_color_dict keys, in case any of them contain newlines
        clean_color_dict = {
            method.replace("\n", " ").strip(): color
            for method, color in method_color_dict.items()
        }

        method_order = [
            method for method in clean_color_dict.keys()
            if method in plot_df["method_name_clean"].unique()
        ]

        def format_method_label(method):
            return (
                method
                .replace(" (own test set)", "\n(own test set)")
                .replace(" (cross test set)", "\n(cross test set)")
                .replace(" (cross-trained)", "\n(cross-trained)")
            )

        fig, ax = plt.subplots(figsize=(5, 4))

        sns.boxplot(
            data=plot_df,
            x="method_name_clean",
            y=variable_name,
            hue="method_name_clean",
            order=method_order,
            hue_order=method_order,
            palette=clean_color_dict,
            showfliers=False,
            width=0.5,
            ax=ax,
            legend=False,
        )

        ax.set_xticks(range(len(method_order)))
        ax.set_xticklabels(
            [format_method_label(method) for method in method_order],
            rotation=45,
            ha="right",
            fontsize=12,
            rotation_mode="anchor"
        )

        ax.tick_params(axis="y", labelsize=12)
        ax.set_ylabel(variable_name.replace("_", " ").title(), fontsize=14)
        ax.set_xlabel("")
        ax.set_title(
            f"{variable_name.replace('_', ' ').title()} by Method Across Samples" + (f"{title_suffix}" if title_suffix else ""),
            fontsize=16,
        )

        fig.tight_layout()

        return fig

    def plot_grn_size_jitter(self, 
        summary_df,
        method_color_dict,
        variable_name="num_edges",
        sample_col="sample_name",
        point_size_col="num_unique_tfs",
        sample_order=None,
        sample_rename_map=None,
        title_suffix=None,
        figsize=(10, 5),
        jitter=0.12,
        min_point_size=50,
        max_point_size=250,
        random_seed=42,
    ):
        """
        Plot GRN sizes across samples.

        Point color represents the GRN inference method.
        Point size represents the value in point_size_col, which defaults
        to the number of unique TFs.
        """
        required_columns = {
            variable_name,
            sample_col,
            "method_name",
        }
        missing_columns = required_columns.difference(summary_df.columns)

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        if sample_rename_map is None:
            sample_rename_map = {}

        plot_df = summary_df.copy()

        # Clean method names
        plot_df["method_name_clean"] = (
            plot_df["method_name"]
            .astype(str)
            .str.replace("\n", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        # Clean method-color dictionary keys in the same way
        clean_color_dict = {
            " ".join(str(method).replace("\n", " ").split()): color
            for method, color in method_color_dict.items()
        }

        # Retain dictionary ordering
        available_methods = set(
            plot_df["method_name_clean"].dropna().unique()
        )
        method_order = [
            method
            for method in clean_color_dict
            if method in available_methods
        ]

        if sample_order is None:
            sample_order = (
                plot_df[sample_col]
                .dropna()
                .drop_duplicates()
                .tolist()
            )

        # Keep only requested samples and methods with assigned colors
        plot_df = plot_df[
            plot_df[sample_col].isin(sample_order)
            & plot_df["method_name_clean"].isin(method_order)
        ].copy()

        if plot_df.empty:
            raise ValueError(
                "No observations remain after filtering by sample and method."
            )

        sample_to_x = {
            sample: position
            for position, sample in enumerate(sample_order)
        }

        plot_df["x_position"] = plot_df[sample_col].map(sample_to_x)

        # Add deterministic horizontal jitter
        rng = np.random.default_rng(random_seed)
        plot_df["x_jittered"] = (
            plot_df["x_position"]
            + rng.uniform(
                -jitter,
                jitter,
                size=len(plot_df),
            )
        )

        # Scale point area by number of unique TFs
        use_size_mapping = (
            point_size_col is not None
            and point_size_col in plot_df.columns
        )

        if use_size_mapping:
            plot_df[point_size_col] = (
                plot_df[point_size_col]
                .astype(float)
            )

            size_min = plot_df[point_size_col].min()
            size_max = plot_df[point_size_col].max()

            def scale_point_size(value):
                if size_max == size_min:
                    return (min_point_size + max_point_size) / 2

                return (
                    min_point_size
                    + (value - size_min)
                    / (size_max - size_min)
                    * (max_point_size - min_point_size)
                )

            plot_df["point_size"] = (
                plot_df[point_size_col]
                .apply(scale_point_size)
            )

        else:
            size_min = None
            size_max = None
            plot_df["point_size"] = 90

        fig, (ax, legend_ax) = plt.subplots(
            ncols=2,
            figsize=figsize,
            gridspec_kw={
                "width_ratios": [3.2, 1.8],
                "wspace": 0.05,
            },
        )

        legend_ax.axis("off")

        # Plot methods separately to construct the color legend
        for method in method_order:
            method_df = plot_df[
                plot_df["method_name_clean"] == method
            ]

            if method_df.empty:
                continue

            ax.scatter(
                method_df["x_jittered"],
                method_df[variable_name],
                s=method_df["point_size"],
                color=clean_color_dict[method],
                label=method,
                alpha=0.85,
                edgecolor="white",
                linewidth=0.7,
                zorder=3,
            )

        # Configure sample labels
        formatted_sample_labels = [
            sample_rename_map.get(sample, sample)
            for sample in sample_order
        ]

        ax.set_xticks(range(len(sample_order)))
        ax.set_xticklabels(
            formatted_sample_labels,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=12,
        )

        y_label_map = {
            "num_edges": "Number of Edges",
            "num_unique_tfs": "Number of Unique TFs",
            "num_unique_tgs": "Number of Unique TGs",
        }

        title_map = {
            "num_edges": "GRN Edge Counts",
            "num_unique_tfs": "Unique TF Counts",
            "num_unique_tgs": "Unique TG Counts",
        }

        y_label = y_label_map.get(
            variable_name,
            variable_name.replace("_", " ").title(),
        )
        plot_title = title_map.get(
            variable_name,
            variable_name.replace("_", " ").title(),
        )

        ax.set_xlabel("")
        ax.set_ylabel(y_label, fontsize=14)
        ax.set_title(
            plot_title + (title_suffix or ""),
            fontsize=16,
        )

        ax.tick_params(axis="y", labelsize=12)

        # Format large y-axis values as 50K, 100K, 1M, etc.
        def format_large_number(value, position):
            if abs(value) >= 1_000_000:
                return f"{value / 1_000_000:g}M"

            if abs(value) >= 1_000:
                return f"{value / 1_000:g}K"

            return f"{value:g}"

        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(format_large_number)
        )

        ax.grid(
            axis="y",
            linestyle="--",
            linewidth=0.8,
            alpha=0.25,
            zorder=0,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Create method legend handles explicitly
        method_handles = [
            Line2D(
                [],
                [],
                linestyle="none",
                marker="o",
                markersize=8,
                markerfacecolor=clean_color_dict[method],
                markeredgecolor="white",
                markeredgewidth=0.7,
                label=method,
            )
            for method in method_order
        ]

        method_legend = legend_ax.legend(
            handles=method_handles,
            title="Method",
            loc="upper left",
            bbox_to_anchor=(0.0, 1.05),
            borderaxespad=0,
            frameon=False,
            facecolor="white",
            fontsize=11,
            title_fontsize=12,
        )

        legend_ax.add_artist(method_legend)

        # TF-size legend
        if use_size_mapping:
            desired_legend_values = [50, 100, 150, 200]

            tf_legend_values = [
                value
                for value in desired_legend_values
                if size_min <= value <= size_max
            ]

            # Use data-derived values if fixed values are outside the data range
            if not tf_legend_values:
                tf_legend_values = (
                    np.linspace(size_min, size_max, 4)
                    .round()
                    .astype(int)
                )
                tf_legend_values = np.unique(
                    tf_legend_values
                ).tolist()

            size_handles = [
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="o",
                    markersize=np.sqrt(
                        scale_point_size(value)
                    ),
                    markerfacecolor="dimgray",
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                    label=f"{value:,}",
                )
                for value in tf_legend_values
            ]

            legend_ax.legend(
                handles=size_handles,
                title="TFs",
                loc="upper left",
                bbox_to_anchor=(0.0, 0.40),
                borderaxespad=0,
                frameon=False,
                facecolor="white",
                fontsize=11,
                title_fontsize=12,
                handletextpad=0.8,
                labelspacing=0.8,
            )

        ax.set_xlim(-0.4, len(sample_order) - 0.5)

        fig.subplots_adjust(
            left=0.10,
            right=0.96,
            top=0.86,
            bottom=0.27,
        )

        return fig

    def add_percent_edges_vs_own(self, 
        summary_df,
        own_method=OWN_MODEL_METHOD,
        edge_col="num_edges",
        percent_col="percent_of_total_edge_combinations",
    ):
        plot_df = summary_df.copy()

        own_edge_df = (
            plot_df[plot_df["method_name"] == own_method]
            [["sample_name", edge_col]]
            .drop_duplicates(subset=["sample_name"])
            .rename(columns={edge_col: "own_test_set_edges"})
        )

        plot_df = plot_df.merge(
            own_edge_df,
            on="sample_name",
            how="left",
            validate="many_to_one",
        )

        missing_samples = plot_df.loc[
            plot_df["own_test_set_edges"].isna(),
            "sample_name"
        ].unique()

        if len(missing_samples) > 0:
            raise ValueError(
                f"Missing own-test-set edge counts for samples: {missing_samples}"
            )

        plot_df[percent_col] = (
            100 * plot_df[edge_col] / plot_df["own_test_set_edges"]
        )

        # plot_df = plot_df[
        #     (plot_df["method_name"] != own_method) & 
        #     (plot_df["method_name"] != CROSS_MODEL_METHOD)
        #     ].copy()

        return plot_df

    def run(self):
        """Execute this section end-to-end (data generation, plotting, saving)."""
        auprc_grns = load_auprc_grns_all_methods()
        # NOTE(refactor): create_grn_size_summary_df reads self.auprc_grns rather
        # than its `sample_grn_dict` argument, matching the notebook. See module
        # docstring for the caveat about auroc_summary_df.
        self.auprc_grns = auprc_grns
        sample_list = list(auprc_grns.keys())

        auroc_grns = {sample: load_generalizability_df(sample, sample) for sample in sample_list}
        print("Loaded AUPRC and AUROC generalizability data for all samples.")
        print(auroc_grns.keys())
        print(auprc_grns.keys())


        auprc_summary_df = self.create_grn_size_summary_df(auprc_grns)
        auroc_summary_df = self.create_grn_size_summary_df(auroc_grns)


        # Plot GRN size boxplots for AUPRC GRNs
        auprc_tf_by_method_boxplot_fig = self.plot_grn_size_boxplot(auprc_summary_df, method_color_dict, variable_name="num_unique_tfs", title_suffix="\n(AUPRC GRNs)")
        auprc_tg_by_method_boxplot_fig = self.plot_grn_size_boxplot(auprc_summary_df, method_color_dict, variable_name="num_unique_tgs", title_suffix="\n(AUPRC GRNs)")
        auprc_edge_by_method_boxplot_fig = self.plot_grn_size_boxplot(auprc_summary_df, method_color_dict, variable_name="num_edges", title_suffix="\n(AUPRC GRNs)")

        auprc_tf_by_method_boxplot_fig.savefig(grn_sizes_by_method_dir / "num_tfs_by_method_boxplot_auprc.png", dpi=300, bbox_inches="tight")
        auprc_tg_by_method_boxplot_fig.savefig(grn_sizes_by_method_dir / "num_tgs_by_method_boxplot_auprc.png", dpi=300, bbox_inches="tight")
        auprc_edge_by_method_boxplot_fig.savefig(grn_sizes_by_method_dir / "num_edges_by_method_boxplot_auprc.png", dpi=300, bbox_inches="tight")

        # Plot GRN size boxplots for AUROC GRNs
        auroc_tf_by_method_boxplot_fig = self.plot_grn_size_boxplot(auroc_summary_df, method_color_dict, variable_name="num_unique_tfs", title_suffix="\n(AUROC GRNs)")
        auroc_tg_by_method_boxplot_fig = self.plot_grn_size_boxplot(auroc_summary_df, method_color_dict, variable_name="num_unique_tgs", title_suffix="\n(AUROC GRNs)")
        auroc_edge_by_method_boxplot_fig = self.plot_grn_size_boxplot(auroc_summary_df, method_color_dict, variable_name="num_edges", title_suffix="\n(AUROC GRNs)")

        auroc_tf_by_method_boxplot_fig.savefig(grn_sizes_by_method_dir / "num_tfs_by_method_boxplot_auroc.png", dpi=300, bbox_inches="tight")
        auroc_tg_by_method_boxplot_fig.savefig(grn_sizes_by_method_dir / "num_tgs_by_method_boxplot_auroc.png", dpi=300, bbox_inches="tight")
        auroc_edge_by_method_boxplot_fig.savefig(grn_sizes_by_method_dir / "num_edges_by_method_boxplot_auroc.png", dpi=300, bbox_inches="tight")





        sample_order = list(sample_rename_map.keys())

        auprc_edge_jitter_fig = self.plot_grn_size_jitter(
            summary_df=auprc_summary_df,
            method_color_dict=method_color_dict,
            variable_name="num_edges",
            sample_col="sample_name",
            point_size_col="num_unique_tfs",
            sample_order=sample_order,
            sample_rename_map=sample_rename_map,
            title_suffix="\n(AUPRC GRNs)",
            figsize=(10, 5),
        )

        auprc_edge_jitter_fig.savefig(
            grn_sizes_by_method_dir / "num_edges_jitter_auprc.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        plt.show()

        auroc_edge_jitter_fig = self.plot_grn_size_jitter(
            summary_df=auroc_summary_df,
            method_color_dict=method_color_dict,
            variable_name="num_edges",
            sample_col="sample_name",
            point_size_col="num_unique_tfs",
            sample_order=sample_order,
            sample_rename_map=sample_rename_map,
            title_suffix="\n(AUROC GRNs)",
            figsize=(10, 5),
        )

        auroc_edge_jitter_fig.savefig(
            grn_sizes_by_method_dir / "num_edges_jitter_auroc.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        plt.show()



        summary_percent_df = self.add_percent_edges_vs_own(
            # NOTE(refactor): notebook referenced a stale `summary_df` global from the
            # Generalizability section; the intended input here is auprc_summary_df.
            auprc_summary_df,
            edge_col="num_edges",
            percent_col="percent_edge_combinations\n",
        )

        percent_edges_boxplot_fig = self.plot_grn_size_boxplot(
            summary_percent_df,
            method_color_dict,
            variable_name="percent_edge_combinations\n",
        )

        percent_edges_boxplot_fig.savefig(
            grn_sizes_by_method_dir / "percent_edge_combinations_test_set_boxplot.png",
            dpi=300,
            bbox_inches="tight",
        )
