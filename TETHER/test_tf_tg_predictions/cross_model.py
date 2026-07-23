"""Section: cross-trained model vs. other GRN inference methods (box/whisker, ROC, PRC, lift, rank plots).

Refactored from ``test_tf_tg_predictions.ipynb`` (unmodified). This module
defines one child class of :class:`.base.TFTGBase`; the plotting helpers are
methods and the notebook's driver cells live in :meth:`run`.
"""
from .base import *  # noqa: F401,F403  (config, shared funcs, notebook imports)
from .base import TFTGBase

# Section-specific imports (hoisted from the notebook cells)
import matplotlib.patheffects as pe


class CrossModelComparison(TFTGBase):
    """Section: cross-trained model vs. other GRN inference methods (box/whisker, ROC, PRC, lift, rank plots)."""

    def _is_binary_like(self, series):
        vals = pd.Series(series).dropna().unique()
        return len(vals) > 0 and set(vals).issubset({0, 1, 0.0, 1.0, True, False})

    def standardize_prediction_score_label_df(self, df):
        """
        Makes cached/generated model prediction files consistent.

        Expected output:
            - Score column
            - _in_gt column when Source/Target are absent
        """
        df = df.copy()

        if "Score" not in df.columns and "score" in df.columns:
            df = df.rename(columns={"score": "Score"})

        if "_in_gt" not in df.columns:
            if "label" in df.columns:
                df["_in_gt"] = df["label"]
            elif "Label" in df.columns:
                df["_in_gt"] = df["Label"]

        # Handle accidentally reversed label/score columns.
        if "Score" in df.columns and "_in_gt" in df.columns:
            score_is_binary = self._is_binary_like(df["Score"])
            label_is_binary = self._is_binary_like(df["_in_gt"])

            if score_is_binary and not label_is_binary:
                old_score = df["Score"].copy()
                df["Score"] = df["_in_gt"]
                df["_in_gt"] = old_score

        if "_in_gt" in df.columns:
            df["_in_gt"] = df["_in_gt"].astype(int)

        return df

    def load_or_generate_tftg_predictions(self, 
        label_file,
        tf_dna_model_chkpt,
        tf_tg_model_chkpt,
        tf_embeddings_tensor,
        tf_mask_tensor,
        data_loader,
        device,
        tf_idx_to_name,
        tg_idx_to_name,
        compile_model=True,
    ):
        if label_file.exists():
            logging.info(f"    - Loading cached predictions from: {label_file.name}")
            prediction_df = pd.read_csv(label_file)
            return self.standardize_prediction_score_label_df(prediction_df)

        logging.info(f"    - Generating predictions and saving to: {label_file.name}")

        tf_tg_model = utils.load_tf_tg_regulation_model(
            tf_dna_model_chkpt,
            tf_tg_model_chkpt,
            tf_embeddings_tensor,
            tf_mask_tensor,
            compile_model=compile_model,
        )

        prediction_df = generate_model_predictions(
            tf_tg_model.model,
            data_loader,
            device,
            tf_idx_to_name,
            tg_idx_to_name,
        )

        prediction_df = self.standardize_prediction_score_label_df(prediction_df)

        label_file.parent.mkdir(parents=True, exist_ok=True)
        prediction_df.to_csv(label_file, index=False)

        return prediction_df

    def _style_xticklabels(self, ax, originals, method_color_dict,
                           sample_rename_map=None, color_xticks=True,
                           rotation=45, fontsize=15):
        sample_rename_map = sample_rename_map or {}
        labels = [sample_rename_map.get(o, o) for o in originals]

        ax.set_xticks(range(len(originals)))
        ax.set_xticklabels(labels, rotation=rotation, ha="right",
                           fontsize=fontsize, rotation_mode="anchor")

        for tick, original in zip(ax.get_xticklabels(), originals):
            if original in TFTG_MODEL_METHODS:

                tick.set_fontsize(fontsize)
                if color_xticks:
                    tick.set_color(method_color_dict.get(original, "black"))
                tick.set_path_effects([pe.withStroke(linewidth=0.6, foreground=tick.get_color())])
            else:
                tick.set_color("black")
                tick.set_fontweight("normal")

    def plot_method_box_and_whisker(self, 
        full_metric_df, 
        selected_column, 
        method_color_dict, 
        sample_rename_map, 
        show_values_above_boxes=True
        ):

        metric_ordered_by_auroc = (
            full_metric_df
            .groupby("method_name")[selected_column]
            .median()
            .sort_values(ascending=False)
            .index
            .tolist()
        )

        sanitized_name = selected_column.replace("_", " ").lower()
        if sanitized_name in ["auroc", "auprc"]:
            sanitized_name = sanitized_name.upper()
            sanitized_name = sanitized_name.replace("AU", "")
        else:
            sanitized_name = sanitized_name.capitalize()

        loosely_dashed = (5, (10, 3))

        fig = plt.figure(figsize=(7, 6))

        ax = sns.boxplot(
            data=full_metric_df, 
            x="method_name", 
            y=selected_column, 
            hue="method_name", 
            width=0.6,
            order=metric_ordered_by_auroc, 
            palette=method_color_dict,
            whiskerprops={"linestyle": loosely_dashed, "linewidth": 1},
            boxprops={"linewidth": 0},
            capprops={"linewidth": 1},
            medianprops={"linewidth": 1},
            showfliers=False
        )

        medians = (
            full_metric_df
            .groupby("method_name")[selected_column]
            .median()
            .reindex(metric_ordered_by_auroc)
        )

        for i, method in enumerate(metric_ordered_by_auroc):
            vals = full_metric_df.loc[full_metric_df["method_name"] == method, selected_column].dropna()
            q1, q3 = vals.quantile([0.25, 0.75])
            top = vals[vals <= q3 + 1.5 * (q3 - q1)].max()

            if show_values_above_boxes:
                annotation = f"{vals.median():.3f}"
            else:
                ranks = medians.rank(method="min", ascending=False).astype(int)
                annotation = str(ranks[method])

            ax.text(i, top + 0.02, annotation, ha="center", va="bottom", fontsize=12)

        self._style_xticklabels(
            ax,
            metric_ordered_by_auroc,
            method_color_dict=method_color_dict,
            sample_rename_map=sample_rename_map,
            color_xticks=True,
            rotation=45,
            fontsize=12
        )

        plt.title(f"{sanitized_name} by Method", fontsize=17)
        plt.xlabel("")
        plt.ylabel(sanitized_name, fontsize=14)
        plt.yticks(fontsize=14)

        plt.ylim((0, 1))

        plt.tight_layout()

        return fig, ax

    def get_labels_and_scores_for_roc(self, df, gt_pairs):
        """
        Returns labels and scores for either:
          1. TF-TG model prediction dfs with _in_gt and Score columns
          2. External method GRN dfs with Source, Target, and Score columns
        """
        df = df.copy()

        if "Score" not in df.columns and "score" in df.columns:
            df = df.rename(columns={"score": "Score"})

        if "Score" not in df.columns:
            raise ValueError("DataFrame is missing a Score column.")

        if "_in_gt" in df.columns:
            labels = df["_in_gt"].astype(int).to_numpy()

        elif "Source" in df.columns and "Target" in df.columns:
            pairs = df["Source"].astype(str) + "\t" + df["Target"].astype(str)
            labels = pairs.isin(gt_pairs).astype(int).to_numpy()

        else:
            raise ValueError(
                "DataFrame must have either _in_gt or Source/Target columns."
            )

        scores = df["Score"].astype(float).to_numpy()

        valid_mask = np.isfinite(labels) & np.isfinite(scores)
        labels = labels[valid_mask]
        scores = scores[valid_mask]

        return labels, scores

    def plot_sample_roc_curves(self, 
        sample_name,
        standardized_method_dfs,
        gt_by_sample_dict,
        method_color_dict,
        sample_rename_map,
        roc_plot_dir,
        method_display_name_map=None,
        figsize=(6, 6),
    ):
        if method_display_name_map is None:
            method_display_name_map = {}

        if sample_name not in standardized_method_dfs:
            raise KeyError(f"Sample {sample_name} not found in standardized_method_dfs")

        if sample_name not in gt_by_sample_dict:
            raise KeyError(f"Sample {sample_name} not found in gt_by_sample_dict")

        sample_title = sample_rename_map.get(sample_name, sample_name)
        gt_pairs = gt_by_sample_dict[sample_name]["gt_pairs"]

        combined_fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=figsize,
            sharex=True,
            sharey=True,
        )

        ax.set_box_aspect(1)

        cell_type_method_auroc = {
            "sample_name": [],
            "method": [],
            "auroc": [],
        }

        auroc_text_lines = []

        for method, df in standardized_method_dfs[sample_name].items():
            if method not in method_color_dict:
                continue

            y_roc, s_roc = self.get_labels_and_scores_for_roc(df, gt_pairs)

            if len(np.unique(y_roc)) < 2:
                logging.warning(
                    f"Skipping ROC for {sample_name} / {method}: only one class present."
                )
                continue

            fpr, tpr, _ = roc_curve(y_roc, s_roc)
            auroc = roc_auc_score(y_roc, s_roc)

            cell_type_method_auroc["sample_name"].append(sample_name)
            cell_type_method_auroc["method"].append(method)
            cell_type_method_auroc["auroc"].append(auroc)

            method_color = method_color_dict.get(method, "#747474")
            auroc_text_lines.append((method, auroc, method_color))

            line_weight = 3.5 if method in TFTG_MODEL_METHODS else 2

            ax.plot(
                fpr,
                tpr,
                lw=line_weight,
                color=method_color,
                label="",
                zorder=3 if method in TFTG_MODEL_METHODS else 2,
            )

        # ROC random baseline
        ax.plot(
            [0, 1],
            [0, 1],
            lw=2,
            linestyle="--",
            color="black",
            alpha=0.6,
            zorder=1,
        )

        auroc_text_lines_sorted = sorted(
            auroc_text_lines,
            key=lambda x: x[1],
            reverse=True,
        )

        ax.set_title(sample_title, fontsize=26)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=18)

        legend_rows = []

        for method, auroc, method_color in auroc_text_lines_sorted:
            display_method = method_display_name_map.get(method, method)

            color_box = DrawingArea(16, 16, 0, 0)
            color_box.add_artist(
                Rectangle(
                    (0, 1),
                    18,
                    18,
                    facecolor=method_color,
                    edgecolor="black",
                    linewidth=0.75,
                )
            )

            label_text = TextArea(
                f"{display_method} = {auroc:.3f}",
                textprops=dict(
                    color="black",
                    fontsize=18,
                    fontweight="bold" if method in TFTG_MODEL_METHODS else "normal",
                ),
            )

            row = HPacker(
                children=[color_box, label_text],
                align="center",
                pad=0.1,
                sep=6,
            )

            legend_rows.append(row)

        packed_legend = VPacker(
            children=legend_rows,
            align="left",
            pad=0,
            sep=8,
        )

        anchored_text = AnchoredOffsetbox(
            loc="upper center",
            child=packed_legend,
            pad=0.5,
            frameon=False,
            bbox_to_anchor=(0.5, -0.22),
            bbox_transform=ax.transAxes,
            borderpad=0.4,
        )

        ax.add_artist(anchored_text)

        combined_fig.text(
            0.5,
            -0.05,
            "False Positive Rate",
            ha="center",
            fontsize=24,
        )

        combined_fig.text(
            -0.05,
            0.45,
            "True Positive Rate",
            va="center",
            rotation="vertical",
            fontsize=24,
        )

        combined_fig.subplots_adjust(
            left=0.04,
            right=0.98,
            bottom=0.10,
            top=0.88,
            wspace=0.08,
        )

        roc_plot_dir = Path(roc_plot_dir)

        combined_fig.savefig(
            roc_plot_dir / f"{sample_name}_auroc.png",
            dpi=300,
            bbox_inches="tight",
        )

        auroc_df = pd.DataFrame(cell_type_method_auroc)

        return combined_fig, auroc_df

    def plot_sample_prc_curves(self, 
        sample_name,
        auprc_all_method_dfs,
        method_color_dict,
        sample_rename_map,
        plot_lift=False,
    ):
        sample_title = sample_rename_map.get(sample_name, sample_name)

        combined_fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(6, 6),
            sharex=True,
            sharey=True,
        )

        ax.set_box_aspect(1)

        cell_type_method_auprc = {
            "sample_name": [],
            "method": [],
            "auprc": [],
            "rand_auprc": [],
        }

        if sample_name not in auprc_all_method_dfs:
            raise KeyError(f"Sample {sample_name} not found in auprc_all_method_dfs")

        auprc_text_lines = []
        auprc_metric_rows = []

        for method in auprc_all_method_dfs[sample_name].keys():
            if method not in method_color_dict:
                continue

            auprc_df = auprc_all_method_dfs[sample_name][method]

            y_auprc = auprc_df["_in_gt"].astype(int).to_numpy()
            s_auprc = auprc_df["Score"].astype(float).to_numpy()

            if len(np.unique(y_auprc)) < 2:
                auprc = np.nan
                rand_auprc = np.nan
                continue

            auprc = average_precision_score(y_auprc, s_auprc)
            prec, rec, _ = precision_recall_curve(y_auprc, s_auprc)

            rand_scores = plotting_utils._create_random_distribution(s_auprc)
            rand_prec, rand_rec, _ = precision_recall_curve(y_auprc, rand_scores)
            rand_auprc = average_precision_score(y_auprc, rand_scores)

            auprc_metric_rows.append({
                "sample_name": sample_name,
                "method": method,
                "auprc": auprc,
                "rand_auprc": rand_auprc,
            })

            method_color = method_color_dict.get(method, "#747474")

            if plot_lift:
                auprc_lift = auprc / rand_auprc if rand_auprc > 0 else np.nan
                auprc_text_lines.append((method, auprc_lift, method_color))
            else:
                auprc_text_lines.append((method, auprc, method_color))

            line_weight = 3 if method in [OWN_MODEL_METHOD, CROSS_MODEL_METHOD] else 2

            if plot_lift:
                rand_prec_on_real_rec = interpolate_precision_at_recall(
                    source_rec=rand_rec,
                    source_prec=rand_prec,
                    target_rec=rec,
                )

                precision_lift = prec / np.clip(rand_prec_on_real_rec, 1e-12, None)

                y_vals = rec
                x_vals = precision_lift

                rand_y_vals = rec
                rand_x_vals = np.ones_like(rec)

            else:
                y_vals = rec
                x_vals = prec

                rand_y_vals = rand_rec
                rand_x_vals = rand_prec


            ax.step(
                y_vals,
                x_vals,
                where="post",
                lw=line_weight,
                color=method_color,
                label="",
                zorder=3,
            )

            ax.step(
                rand_y_vals,
                rand_x_vals,
                where="post",
                lw=1,
                linestyle="--",
                color=method_color,
                label="",
                zorder=3,
                alpha=0.75,
            )

        auprc_text_lines_sorted = sorted(
            auprc_text_lines,
            key=lambda x: x[1],
            reverse=True,
        )



        ax.tick_params(labelsize=18)

        legend_rows = []

        for method, auprc, method_color in auprc_text_lines_sorted:
            color_box = DrawingArea(16, 16, 0, 0)
            color_box.add_artist(
                Rectangle(
                    (0, 1),
                    18,
                    18,
                    facecolor=method_color,
                    edgecolor="black",
                    linewidth=0.75,
                )
            )

            label_text = TextArea(
                f"{method} = {auprc:.3f}",
                textprops=dict(
                    color="black",
                    fontsize=18,
                    fontweight="bold" if method in TFTG_MODEL_METHODS else "normal",
                ),
            )

            row = HPacker(
                children=[color_box, label_text],
                align="center",
                pad=0.1,
                sep=6,
            )

            legend_rows.append(row)

        packed_legend = VPacker(
            children=legend_rows,
            align="left",
            pad=0,
            sep=8,
        )

        anchored_text = AnchoredOffsetbox(
            loc="upper center",
            child=packed_legend,
            pad=0.5,
            frameon=False,
            bbox_to_anchor=(0.5, -0.22),
            bbox_transform=ax.transAxes,
            borderpad=0.4,
        )

        ax.add_artist(anchored_text)

        combined_fig.text(
            0.5,
            -0.05,
            "Recall",
            ha="center",
            fontsize=24,
        )

        if plot_lift:
            ylabel = "Precision / Random Baseline"
        else:
            ylabel = "Precision"

        combined_fig.text(
            -0.05,
            0.45,
            ylabel,
            va="center",
            rotation="vertical",
            fontsize=24,
        )

        ax.set_title(sample_title, fontsize=26)
        ax.set_xlim(0, 1)
        if not plot_lift:
            ax.set_ylim(0, 1)

        combined_fig.subplots_adjust(
            left=0.04,
            right=0.98,
            bottom=0.10,
            top=0.88,
            wspace=0.08,
        )

        return combined_fig, ax

    def lift_by_method_boxplot(self, 
        full_metric_df,
        metric_col,
        rand_col,
        method_color_dict,
        title=None,
        method_order=None,
        sample_rename_map=None,
        figsize=(7, 5),
        color_xticks=True,
        show_values_above_boxes=True,
        showfliers=False,
    ):
        if sample_rename_map is None:
            sample_rename_map = {}

        plot_df = full_metric_df[["method_name", metric_col, rand_col]].dropna().copy()
        plot_df = plot_df[plot_df[rand_col] > 0]
        plot_df["lift"] = plot_df[metric_col] / plot_df[rand_col]

        if method_order is None:
            method_order = (
                plot_df.groupby("method_name")["lift"]
                .median()
                .sort_values(ascending=False)
                .index
                .tolist()
            )

        loosely_dashed = (5, (10, 3))

        fig, ax = plt.subplots(figsize=figsize)

        sns.boxplot(
            data=plot_df,
            x="method_name",
            y="lift",
            hue="method_name",
            order=method_order,
            width=0.6,
            palette=method_color_dict,
            whiskerprops={"linestyle": loosely_dashed, "linewidth": 1},
            boxprops={"linewidth": 0},
            capprops={"linewidth": 1},
            medianprops={"linewidth": 1},
            showfliers=showfliers,
            legend=False,
            ax=ax,
        )

        ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6, zorder=0)

        y_span = plot_df["lift"].max() - plot_df["lift"].min()
        offset = max(y_span * 0.02, 0.02)

        for i, method in enumerate(method_order):
            vals = plot_df.loc[plot_df["method_name"] == method, "lift"]
            if vals.empty:
                continue

            q1, q3 = vals.quantile([0.25, 0.75])
            top = vals[vals <= q3 + 1.5 * (q3 - q1)].max()

            annotation = f"{vals.median():.2f}" if show_values_above_boxes else str(i + 1)
            ax.text(i, top + offset, annotation, ha="center", va="bottom", fontsize=14)

        self._style_xticklabels(
            ax, method_order, method_color_dict, sample_rename_map,
            color_xticks, rotation=45, fontsize=14,
        )

        ax.tick_params(axis="y", labelsize=14)
        ax.set_ylabel("Lift", fontsize=15)
        ax.set_xlabel("")
        ax.set_title(title or "Lift by Method", fontsize=17)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

        plt.tight_layout()
        return fig

    def metric_df_to_rank_df(self, 
        full_metric_df,
        metric_col,
        experiment_col="sample_name",
        method_col="method_name",
        higher_is_better=True,
    ):
        metric_df = (
            full_metric_df
            [[experiment_col, method_col, metric_col]]
            .dropna()
            .copy()
        )

        metric_df = (
            metric_df
            .groupby([experiment_col, method_col], as_index=False)
            .agg(metric_value=(metric_col, "median"))
        )

        metric_df = metric_df.rename(
            columns={
                experiment_col: "experiment",
                method_col: "method",
            }
        )

        metric_df["rank"] = (
            metric_df
            .groupby("experiment")["metric_value"]
            .rank(
                method="min",
                ascending=not higher_is_better,
            )
            .astype(int)
        )

        all_ranks_df = metric_df.sort_values(["experiment", "rank"]).copy()

        rank_df = (
            all_ranks_df
            .groupby("method", as_index=False)
            .agg(
                avg_rank=("rank", "mean"),
                median_rank=("rank", "median"),
                mean_metric=("metric_value", "mean"),
            )
            .sort_values(["avg_rank", "median_rank"], ascending=True)
            .reset_index(drop=True)
        )

        return all_ranks_df, rank_df

    def avg_rank_by_method_plot(self, 
        avg_rank_df,
        method_color_dict,
        title,
        sample_rename_map=None,
        figsize=(7, 4),
        color_xticks=True,
    ):
        if sample_rename_map is None:
            sample_rename_map = {}

        plot_df = avg_rank_df.copy()
        order = plot_df["method"].tolist()

        fig, ax = plt.subplots(figsize=figsize)

        sns.barplot(
            data=plot_df,
            x="method",
            y="avg_rank",
            order=order,
            hue="method",
            palette=method_color_dict,
            dodge=False,
            legend=False,
            ax=ax,
        )

        self._style_xticklabels(ax, order, method_color_dict, sample_rename_map, color_xticks)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
        ax.set_ylabel("Average Rank", fontsize=15)
        ax.set_xlabel("")
        ax.set_title(title, fontsize=17)

        ax.set_ylim(0, max(plot_df["avg_rank"].max() + 0.5, 1.5))

        plt.tight_layout()
        return fig

    def avg_rank_by_method_lollipop_plot(self, 
        avg_rank_df,
        method_color_dict,
        title,
        sample_rename_map=None,
        color_xticks=True,
        figsize=(7, 4),
    ):
        if sample_rename_map is None:
            sample_rename_map = {}

        plot_df = avg_rank_df.copy()
        order = plot_df["method"].tolist()

        y_positions = np.arange(len(plot_df))
        avg_ranks = plot_df["avg_rank"].to_numpy()
        colors = [method_color_dict.get(m, "gray") for m in plot_df["method"]]

        fig, ax = plt.subplots(figsize=figsize)

        max_rank = int(np.ceil(plot_df["avg_rank"].max()))
        left_edge = max(max_rank, 1.5)

        ax.hlines(
            y=y_positions,
            xmin=left_edge,
            xmax=avg_ranks,
            color=colors,
            linewidth=2,
            alpha=0.8,
            zorder=1,
        )

        ax.scatter(avg_ranks, y_positions, color=colors, s=200, zorder=2)

        ax.set_xlim(left_edge, 0.5)
        ax.set_xticks(np.arange(1, max_rank + 1))
        ax.set_yticks(y_positions)
        labels = [sample_rename_map.get(m, m) for m in order]
        ax.set_yticklabels(labels, fontsize=15)

        for tick, original in zip(ax.get_yticklabels(), order):
            if original in TFTG_MODEL_METHODS:
                tick.set_fontweight("bold")
                if color_xticks:
                    tick.set_color(method_color_dict.get(original, "black"))
            else:
                tick.set_color("black")
                tick.set_fontweight("normal")

        ax.invert_yaxis()

        ax.tick_params(axis="x", labelsize=15)
        ax.set_xlabel("Average Rank", fontsize=15)
        ax.set_ylabel("")
        ax.set_title(title, fontsize=17)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle="--", alpha=0.3)

        plt.tight_layout()
        return fig

    def experiment_by_method_rank_heatmap(self, 
        all_ranks_df,
        rank_df,
        method_color_dict,
        title=None,
        sample_order=None,
        sample_rename_map=None,
        figsize=(10, 4),
        square_cells=True,
        color_xticks=True,
        show_values_in_boxes=True,
        random_baseline_series=None,
        random_baseline_label="Random",
    ):
        if sample_rename_map is None:
            sample_rename_map = {}

        if show_values_in_boxes:
            plot_value = "metric_value"
            cbar_label = "Metric Value"
            rounding_format = ".2f"
            cmap = "viridis"
        else:
            plot_value = "rank"
            cbar_label = "Rank"
            rounding_format = ".0f"
            cmap = "viridis_r"

        rank_heatmap_df = all_ranks_df.pivot(
            index="experiment",
            columns="method",
            values=plot_value,
        )

        method_order = rank_df["method"].tolist()

        if sample_order is None:
            sample_order = rank_heatmap_df.index.tolist()

        sample_order = [
            exp for exp in sample_order
            if exp in rank_heatmap_df.index
        ]

        rank_heatmap_df = rank_heatmap_df.reindex(
            index=sample_order,
            columns=method_order,
        )

        if random_baseline_series is not None:
            rank_heatmap_df[random_baseline_label] = (
                random_baseline_series.reindex(rank_heatmap_df.index)
            )
            method_order = method_order + [random_baseline_label]

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            rank_heatmap_df,
            annot=True,
            fmt=rounding_format,
            cmap=cmap,
            linewidths=0.5,
            linecolor="white",
            annot_kws={"size": 14, "fontweight": "bold"},
            ax=ax,
        )

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label("", fontsize=15)

        if square_cells:
            ax.set_aspect("equal")

        self._style_xticklabels(ax, method_order, method_color_dict, sample_rename_map,
                       color_xticks, rotation=55)

        new_y_labels = []
        for tick in ax.get_yticklabels():
            original = tick.get_text()
            new = sample_rename_map.get(original, original)
            new_y_labels.append(new)

        ax.set_yticklabels(new_y_labels, rotation=0, fontsize=15)

        ax.set_title(title or "Method Rank by Test Set", fontsize=17)
        ax.set_xlabel("")
        ax.set_ylabel("")

        plt.tight_layout()
        return fig

    def rank_by_method_boxplot(self, 
        all_ranks_df,
        rank_df,
        method_color_dict,
        title=None,
        sample_rename_map=None,
        figsize=(7, 6),
        color_xticks=True,
        show_values_above_boxes=True,
        showfliers=False,
    ):
        if sample_rename_map is None:
            sample_rename_map = {}

        plot_df = all_ranks_df.copy()
        order = rank_df["method"].tolist()

        loosely_dashed = (5, (10, 3))

        fig, ax = plt.subplots(figsize=figsize)

        sns.boxplot(
            data=plot_df,
            x="method",
            y="rank",
            hue="method",
            order=order,
            width=0.6,
            palette=method_color_dict,
            whiskerprops={"linestyle": loosely_dashed, "linewidth": 1},
            boxprops={"linewidth": 0},
            capprops={"linewidth": 1},
            medianprops={"linewidth": 1},
            showfliers=showfliers,
            legend=False,
            ax=ax,
        )

        for i, method in enumerate(order):
            vals = plot_df.loc[plot_df["method"] == method, "rank"].dropna()
            if vals.empty:
                continue

            q1, q3 = vals.quantile([0.25, 0.75])
            top = vals[vals <= q3 + 1.5 * (q3 - q1)].max()

            if show_values_above_boxes:
                annotation = f"{vals.median():.1f}"
            else:
                annotation = str(i + 1)

            ax.text(i, top + 0.15, annotation, ha="center", va="bottom", fontsize=12)

        self._style_xticklabels(
            ax, order, method_color_dict, sample_rename_map,
            color_xticks, rotation=45, fontsize=12,
        )

        ax.tick_params(axis="y", labelsize=14)
        ax.set_ylabel("Rank", fontsize=15)
        ax.set_xlabel("")
        ax.set_title(title or "Method Ranks by Test Set", fontsize=17)

        ax.set_ylim(0.5, plot_df["rank"].max() + 0.8)
        # ax.invert_yaxis()

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        plt.tight_layout()
        return fig

    def run(self):
        """Execute this section end-to-end (data generation, plotting, saving)."""
        samples_to_run = [
            ("mESC", "E7.5_rep1", "mouse_hepatocytes", "hepatocytes_1"),
            ("mESC", "E8.5_rep1", "mouse_hepatocytes", "hepatocytes_1"),
            ("Macrophage", "buffer_1", "K562", "sample_1"),
            ("Macrophage", "buffer_2", "K562", "sample_1"),
            ("K562", "sample_1", "Macrophage", "buffer_1"),
            ("mouse_hepatocytes", "hepatocytes_1", "mESC", "E7.5_rep1"),
            ("mouse_hepatocytes", "hepatocytes_3", "mESC", "E7.5_rep1"),
        ]







        standardized_method_dfs = {}
        label_by_method_dict = {}
        score_by_method_dict = {}
        metric_by_method_list = []
        gt_by_sample_dict = {}

        sample_list = [sample_name for _, sample_name, _, _ in samples_to_run]
        auprc_all_method_dfs = load_auprc_grns_all_methods(sample_list=sample_list)

        subset_size = 10_000
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for model_cell_type, sample_name, cross_model_cell_type, cross_model_sample_name in samples_to_run:

            logging.info(
                f"Processing test sample: {sample_name} | "
                f"Test cell type: {model_cell_type} | "
                f"Cross-trained model: {cross_model_cell_type}/{cross_model_sample_name}"
            )

            cell_type_cache_dir = DATA_DIR / f"{model_cell_type}_cache"

            model_label_file = (
                score_label_save_dir
                / f"{sample_name}_model_vs_{sample_name}_grn_{subset_size}.csv"
            )

            cross_model_label_file = (
                score_label_save_dir
                / f"{cross_model_sample_name}_model_vs_{sample_name}_grn_{subset_size}.csv"
            )

            # Load cached test dataset for the evaluation/test sample.
            data_loader, metadata, manifest, tf_embeddings_tensor, tf_mask_tensor = utils.load_training_cache_dataset(
                sample_name=sample_name,
                cell_type_cache_dir=cell_type_cache_dir,
                split_type="test",
                subset_size=subset_size,
            )

            # Load full test labels / edge metadata.
            tftg_inputs_test = torch.load(
                cell_type_cache_dir / "tf_tg_training_cache" / sample_name / "tftg_inputs_test.pt",
                weights_only=False,
            )

            tf_idx_to_name, tg_idx_to_name = create_tf_tg_index_to_name_mappings(metadata)

            test_set_tf_indices = list(tftg_inputs_test["tf_idx"].numpy())
            test_set_tg_indices = list(tftg_inputs_test["tg_idx"].numpy())

            tf_names = [tf_idx_to_name[int(idx)].upper() for idx in test_set_tf_indices]
            tg_names = [tg_idx_to_name[int(idx)].upper() for idx in test_set_tg_indices]

            # -----------------------------
            # Own-model predictions
            # -----------------------------
            own_tf_tg_model_chkpt = tf_tg_model_checkpoints[model_cell_type][sample_name]
            own_tf_dna_model_chkpt = config.tf_dna_model_checkpoints[model_cell_type]

            prediction_df = self.load_or_generate_tftg_predictions(
                label_file=model_label_file,
                tf_dna_model_chkpt=own_tf_dna_model_chkpt,
                tf_tg_model_chkpt=own_tf_tg_model_chkpt,
                tf_embeddings_tensor=tf_embeddings_tensor,
                tf_mask_tensor=tf_mask_tensor,
                data_loader=data_loader,
                device=device,
                tf_idx_to_name=tf_idx_to_name,
                tg_idx_to_name=tg_idx_to_name,
                compile_model=False,
            )

            # -----------------------------
            # Cross-trained model predictions
            # -----------------------------
            cross_tf_tg_model_chkpt = tf_tg_model_checkpoints[cross_model_cell_type][cross_model_sample_name]
            cross_tf_dna_model_chkpt = config.tf_dna_model_checkpoints[cross_model_cell_type]

            cross_model_prediction_df = self.load_or_generate_tftg_predictions(
                label_file=cross_model_label_file,
                tf_dna_model_chkpt=cross_tf_dna_model_chkpt,
                tf_tg_model_chkpt=cross_tf_tg_model_chkpt,
                tf_embeddings_tensor=tf_embeddings_tensor,
                tf_mask_tensor=tf_mask_tensor,
                data_loader=data_loader,
                device=device,
                tf_idx_to_name=tf_idx_to_name,
                tg_idx_to_name=tg_idx_to_name,
                compile_model=False,
            )

            # -----------------------------
            # Ground truth
            # -----------------------------
            tf_tg_label_df, gt_pairs, gt_tfs, gt_targets = create_tf_tg_label_df(tftg_inputs_test)

            gt_tfs = gt_tfs.intersection(set(tf_names))
            gt_targets = gt_targets.intersection(set(tg_names))

            gt_pairs = {
                pair
                for pair in gt_pairs
                if pair.split("\t")[0] in gt_tfs and pair.split("\t")[1] in gt_targets
            }

            gt_by_sample_dict[sample_name] = {
                "gt_pairs": gt_pairs,
                "gt_tfs": gt_tfs,
                "gt_targets": gt_targets,
                "test_cell_type": model_cell_type,
                "test_sample": sample_name,
            }

            # -----------------------------
            # Other method GRNs
            # -----------------------------
            OTHER_METHOD_MUON_DIR = Path(
                "/gpfs/Labs/Uzun/DATA/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER/other_method_grns"
            )

            linger_path = (
                OTHER_METHOD_MUON_DIR
                / "LINGER_muon"
                / f"linger_{model_cell_type}_{sample_name}.tsv"
            )

            scenic_plus_path = (
                OTHER_METHOD_MUON_DIR
                / "SCENIC_muon"
                / f"scenicplus_{model_cell_type}_{sample_name}.tsv"
            )

            cell_oracle_path = (
                OTHER_METHOD_MUON_DIR
                / "CellOracle_muon"
                / f"celloracle_{model_cell_type}_{sample_name}.tsv"
            )

            pando_path = (
                OTHER_METHOD_MUON_DIR
                / "Pando_muon"
                / f"pando_{model_cell_type}_{sample_name}.tsv"
            )

            figr_path = (
                OTHER_METHOD_MUON_DIR
                / "FigR_muon"
                / f"figr_{model_cell_type}_{sample_name}.tsv"
            )

            method_info = {
                "SCENIC+": {
                    "path": scenic_plus_path,
                    "tf_col": "Source",
                    "target_col": "Target",
                    "score_col": "Score",
                },
                "LINGER": {
                    "path": linger_path,
                    "tf_col": "Source",
                    "target_col": "Target",
                    "score_col": "Score",
                },
                "CellOracle": {
                    "path": cell_oracle_path,
                    "tf_col": "Source",
                    "target_col": "Target",
                    "score_col": "Score",
                },
                "Pando": {
                    "path": pando_path,
                    "tf_col": "Source",
                    "target_col": "Target",
                    "score_col": "Score",
                },
                "FigR": {
                    "path": figr_path,
                    "tf_col": "Source",
                    "target_col": "Target",
                    "score_col": "Score",
                },
            }

            standardized_method_dfs[sample_name] = {}

            for method_name, info in method_info.items():
                df_std = load_and_standardize_method(method_name, info)

                mask = df_std["Source"].isin(gt_tfs) & df_std["Target"].isin(gt_targets)
                df_filtered = df_std.loc[mask].copy()

                standardized_method_dfs[sample_name][method_name] = df_filtered

            # Important:
            # Use stable method labels, not sample names.
            standardized_method_dfs[sample_name][OWN_MODEL_METHOD] = prediction_df
            standardized_method_dfs[sample_name][CROSS_MODEL_METHOD] = cross_model_prediction_df

            label_by_method_dict[sample_name] = {}
            score_by_method_dict[sample_name] = {}

            # -----------------------------
            # Metric computation
            # -----------------------------
            for method_name, df in standardized_method_dfs[sample_name].items():
                df = self.standardize_prediction_score_label_df(df)

                if "Source" in df.columns and "Target" in df.columns:
                    labels = [
                        1 if pair in gt_pairs else 0
                        for pair in df["Source"] + "\t" + df["Target"]
                    ]
                elif "_in_gt" in df.columns:
                    labels = df["_in_gt"].astype(int).tolist()
                else:
                    raise ValueError(
                        f"{method_name} for {sample_name} has neither "
                        "Source/Target columns nor an _in_gt label column."
                    )

                scores = df["Score"].tolist()

                label_by_method_dict[sample_name][method_name] = labels
                score_by_method_dict[sample_name][method_name] = scores

                metrics_df = compute_metrics(
                    method_name,
                    sample_name,
                    df,
                    gt_pairs,
                    score_threshold=0.5,
                )

                metrics_df["test_cell_type"] = model_cell_type
                metrics_df["test_sample"] = sample_name

                labeled_auprc_df = auprc_all_method_dfs[sample_name][method_name]
                metrics_df["auprc"] = average_precision_score(labeled_auprc_df["_in_gt"], labeled_auprc_df["Score"])
                metrics_df["rand_auprc"] = average_precision_score(labeled_auprc_df["_in_gt"], np.random.rand(len(labeled_auprc_df["_in_gt"])))

                if sample_name == "hepatocytes_1" and method_name == "Pando":
                    metrics_df["auroc"] = roc_auc_score(labeled_auprc_df["_in_gt"], labeled_auprc_df["Score"])
                    metrics_df["rand_auroc"] = roc_auc_score(labeled_auprc_df["_in_gt"], np.random.rand(len(labeled_auprc_df["_in_gt"])))

                if method_name == OWN_MODEL_METHOD:
                    metrics_df["model_eval_type"] = "own_test_set"
                    metrics_df["train_cell_type"] = model_cell_type
                    metrics_df["train_sample"] = sample_name

                elif method_name == CROSS_MODEL_METHOD:
                    metrics_df["model_eval_type"] = "cross_trained"
                    metrics_df["train_cell_type"] = cross_model_cell_type
                    metrics_df["train_sample"] = cross_model_sample_name

                else:
                    metrics_df["model_eval_type"] = "external_method"
                    metrics_df["train_cell_type"] = np.nan
                    metrics_df["train_sample"] = np.nan

                metric_by_method_list.append(metrics_df)

                safe_method_name = (
                    method_name
                    .replace(" ", "_")
                    .replace("/", "_")
                    .replace("(", "")
                    .replace(")", "")
                )

        full_metric_df = pd.concat(metric_by_method_list, ignore_index=True)

        # full_metric_df.to_csv(
        #     full_metric_df_path,
        #     index=False,
        # )

        full_metric_df = pd.read_csv(
            full_metric_df_path
        )





        font_path = fm.findfont("Arial", fallback_to_default=False)

        # Selected columns by method box and whisker plot
        methods = ["auroc", "auprc", "accuracy", "early_precision", "precision", "recall", "f1"]

        method_box_and_whisker_plots = {}
        for selected_column in methods:

            sanitized_name = selected_column.replace("_", " ").lower()
            if sanitized_name in ["auroc", "auprc"]:
                sanitized_name = sanitized_name.upper()
                sanitized_name = sanitized_name.replace("AU", "")
            else:
                sanitized_name = sanitized_name.capitalize()

            method_comparison_boxplot_fig, method_comparison_boxplot_ax = self.plot_method_box_and_whisker(
                full_metric_df, 
                selected_column, 
                method_color_dict, 
                sample_rename_map, 
                show_values_above_boxes=True
            )

            method_box_and_whisker_plots[selected_column] = (method_comparison_boxplot_fig, method_comparison_boxplot_ax)

            plt.savefig(
                method_comparison_boxplot_dir / f"{sanitized_name.lower()}_by_method_boxplot.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.show()




        sample_roc_curves = {}
        for sample_name in sample_order:
            if sample_name not in standardized_method_dfs:
                continue

            fig, ax = self.plot_sample_roc_curves(
                sample_name=sample_name,
                standardized_method_dfs=standardized_method_dfs,
                gt_by_sample_dict=gt_by_sample_dict,
                method_color_dict=method_color_dict,
                sample_rename_map=sample_rename_map,
                roc_plot_dir=roc_plot_dir,
                method_display_name_map=method_display_name_map,
                figsize=(6, 6),
            )

            sample_roc_curves = (fig, ax)

            plt.show()

            fig.savefig(model_vs_other_method_roc_curve_fig_dir / f"{sample_name}_auroc.png", dpi=300, bbox_inches="tight")

        for sample_name in sample_order:
            if sample_name not in auprc_all_method_dfs:
                continue

            fig, sample_auprc_df = self.plot_sample_prc_curves(
                sample_name=sample_name,
                auprc_all_method_dfs=auprc_all_method_dfs,
                method_color_dict=method_color_dict,
                sample_rename_map=sample_rename_map,
            )

            plt.show()


            fig.savefig(model_vs_other_method_prc_curve_fig_dir / f"{sample_name}_auprc.png", dpi=300, bbox_inches="tight")

        early_auprc_all_method_dfs = {}
        for sample_name in sample_order:
            if sample_name not in auprc_all_method_dfs:
                continue

            early_auprc_all_method_dfs[sample_name] = {}

            for method in auprc_all_method_dfs[sample_name].keys():
                if method not in method_color_dict:
                    continue

                early_auprc_all_method_dfs[sample_name][method] = {}

                auprc_df = auprc_all_method_dfs[sample_name][method]

                auprc_df_sorted = auprc_df.sort_values(by="Score", ascending=False)
                top_ten_percent_count = int(0.1 * len(auprc_df_sorted))
                auprc_df_top_10_pct = auprc_df_sorted.head(top_ten_percent_count).copy()  # Top 10% of edges

                early_auprc_all_method_dfs[sample_name][method] = auprc_df_top_10_pct

        for sample_name in sample_order:
            if sample_name not in auprc_all_method_dfs:
                continue
            fig, sample_auprc_df = self.plot_sample_prc_curves(
                sample_name=sample_name,
                auprc_all_method_dfs=auprc_all_method_dfs,
                method_color_dict=method_color_dict,
                sample_rename_map=sample_rename_map,
                plot_lift=True
            )

            plt.show()


            fig.savefig(model_vs_other_method_prc_curve_fig_dir / f"{sample_name}_auprc_lift.png", dpi=300, bbox_inches="tight")


        for metric_col, rand_col, plot_title in [
            ("auroc", "rand_auroc", "AUROC Lifts"),
            ("auprc", "rand_auprc", "AUPRC Lifts"),
        ]:
            lift_by_method_fig = self.lift_by_method_boxplot(
                full_metric_df,
                metric_col=metric_col,
                rand_col=rand_col,
                method_color_dict=method_color_dict,
                title=plot_title,
                sample_rename_map=sample_rename_map,
            )
            lift_by_method_fig.savefig(
                lift_boxplot_dir / f"{metric_col}_lift_by_method_boxplot.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.show()








        metric_name_map = {
            "auroc": "AUROC",
            "auprc": "AUPRC",
            "accuracy": "Accuracy",
            "early_precision": "Early Precision",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
        }

        boxplot_metric_name_map = {
            "auroc": "ROC",
            "auprc": "PRC",
            "accuracy": "Accuracy",
            "early_precision": "Early Precision",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
        }

        metrics = [
            "auroc",
            "auprc",
            "accuracy",
            "early_precision",
            "precision",
            "recall",
            "f1",
        ]

        RAND_COL_BY_METRIC = {
            "auroc": "rand_auroc",
            "auprc": "rand_auprc",
        }

        for metric_col in metrics:
            metric_label = metric_name_map.get(
                metric_col,
                metric_col.replace("_", " ").title(),
            )

            boxplot_metric_label = boxplot_metric_name_map.get(
                metric_col,
                metric_col.replace("_", " ").title(),
            )

            all_ranks_df, rank_df = self.metric_df_to_rank_df(
                full_metric_df=full_metric_df,
                metric_col=metric_col,
                experiment_col="sample_name",
                method_col="method_name",
                higher_is_better=True,
            )

            safe_metric_name = metric_col.lower().replace(" ", "_")

            # Full method average-rank barplot
            full_avg_rank_fig = self.avg_rank_by_method_plot(
                rank_df,
                method_color_dict=method_color_dict,
                title=f"Average {metric_label} Rank",
                sample_rename_map=sample_rename_map,
                figsize=(7, 4),
            )

            full_avg_rank_fig.savefig(
                rank_bar_plot_dir / f"average_{safe_metric_name}_rank_all_methods.png",
                dpi=300,
                bbox_inches="tight",
            )

            # Full method average-rank lollipop plot
            full_avg_rank_fig = self.avg_rank_by_method_lollipop_plot(
                rank_df,
                method_color_dict=method_color_dict,
                title=f"Mean Method Rankings for {metric_label}",
                sample_rename_map=sample_rename_map,
                figsize=(7, 4),
            )

            full_avg_rank_fig.savefig(
                rank_lollipop_plot_dir / f"average_{safe_metric_name}_rank_all_methods_lollipop.png",
                dpi=300,
                bbox_inches="tight",
            )

            rand_col = RAND_COL_BY_METRIC.get(metric_col)

            if rand_col is not None and rand_col in full_metric_df.columns:
                random_baseline = (
                    full_metric_df
                    .groupby("sample_name")[rand_col]
                    .median()
                )
            else:
                random_baseline = None

            # Full method rank heatmap
            full_rank_heatmap_fig = self.experiment_by_method_rank_heatmap(
                all_ranks_df,
                rank_df,
                method_color_dict=method_color_dict,
                sample_order=sample_order,
                sample_rename_map=sample_rename_map,
                title=f"{metric_label}",
                figsize=(7, 4.5),
                square_cells=False,
                color_xticks=True,
                show_values_in_boxes=True,
                random_baseline_series=random_baseline,
            )

            full_rank_heatmap_fig.savefig(
                rank_heatmap_plot_dir / f"{safe_metric_name}_rank_heatmap_all_methods.png",
                dpi=300,
                bbox_inches="tight",
            )

            plt.show()

            full_rank_boxplot_fig = self.rank_by_method_boxplot(
                all_ranks_df,
                rank_df,
                method_color_dict=method_color_dict,
                title=f"{boxplot_metric_label} rank",
                sample_rename_map=sample_rename_map,
                figsize=(7, 6),
                color_xticks=True,
                show_values_above_boxes=True,
                showfliers=False,
            )

            full_rank_boxplot_path = rank_boxplot_plot_dir / f"{safe_metric_name}_rank_boxplot_all_methods.png"
            full_rank_boxplot_fig.savefig(
                full_rank_boxplot_path,
                dpi=300,
                bbox_inches="tight",
            )

            plt.show()
