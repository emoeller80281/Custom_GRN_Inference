"""Section: model and inference-method stability (Jaccard) analysis.

Refactored from ``test_tf_tg_predictions.ipynb`` (unmodified). This module
defines one child class of :class:`.base.TFTGBase`; the plotting helpers are
methods and the notebook's driver cells live in :meth:`run`.
"""
from .base import *  # noqa: F401,F403  (config, shared funcs, notebook imports)
from .base import TFTGBase


class StabilityAnalysis(TFTGBase):
    """Section: model and inference-method stability (Jaccard) analysis."""

    def load_tether_stability_results(self, stability_result_dir, models_to_plot):
        stability_labeled_score_dfs = {}
        stability_metric_dfs = []

        for sample_name in models_to_plot:
            stability_labeled_score_dfs[sample_name] = {}
            for subsample_num in range(0, 10):
                prediction_save_file = stability_result_dir / "labeled_grns" / f"{sample_name}_stability_{subsample_num}_grn.csv"
                metric_save_file = stability_result_dir / "comparison_metric_files" / f"{sample_name}_stability_{subsample_num}_metrics.csv"

                if prediction_save_file.exists():
                    labeled_score_df = pd.read_csv(prediction_save_file)
                    stability_labeled_score_dfs[sample_name][subsample_num] = labeled_score_df
                else:
                    logging.debug(f"Missing prediction file: {prediction_save_file}")

                if metric_save_file.exists():
                    metric_df = pd.read_csv(metric_save_file)
                    stability_metric_dfs.append(metric_df)
                else:
                    logging.debug(f"Missing metric file: {metric_save_file}")

        if stability_metric_dfs:
            combined_stability_metric_df = pd.concat(stability_metric_dfs, ignore_index=True)
        else:
            combined_stability_metric_df = pd.DataFrame()
            logging.debug("No stability metric files found.")

        return stability_labeled_score_dfs, combined_stability_metric_df

    def calculate_jaccard_index(self, df_x, df_y):
        df_x_sorted = df_x.sort_values(by="Score", ascending=False)
        df_y_sorted = df_y.sort_values(by="Score", ascending=False)

        df_x_top_10_pct = df_x_sorted.head(int(0.1 * len(df_x_sorted)))
        df_y_top_10_pct = df_y_sorted.head(int(0.1 * len(df_y_sorted)))

        edges_x = set(zip(df_x_top_10_pct["Source"], df_x_top_10_pct["Target"]))
        edges_y = set(zip(df_y_top_10_pct["Source"], df_y_top_10_pct["Target"]))

        union = edges_x | edges_y
        intersection = edges_x & edges_y
        jaccard_index = len(intersection) / len(union) if union else 0.0

        return jaccard_index

    def plot_method_stability_by_sample_boxplot(self, jaccard_plot_df, sample_order, sample_rename_map):
        fig = plt.figure(figsize=(10, 6))
        plt.title("Stability", fontsize=16)
        sns.boxplot(
            data=jaccard_plot_df, 
            x="sample_name", 
            y="jaccard_index", 
            showfliers=False,
            )

        # Rename using sample_rename_map
        plt.xticks(ticks=range(len(sample_order)), 
                labels=[sample_rename_map.get(sample, sample) for sample in sample_order], 
                rotation=45, ha="right", fontsize=12)

        plt.yticks(fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.ylabel("Pairwise Jaccard Index\n(Top 10% Edges, 10 subsamples)", fontsize=14)
        plt.xlabel("", fontsize=14)
        plt.ylim(0, 1)

        return fig

    def label_df(self, df, gt_tfs, gt_tgs, gt_pairs):
        df = df.copy()

        df = df[
            (df["Source"].str.upper().isin(gt_tfs)) &
            (df["Target"].str.upper().isin(gt_tgs))
            ]

        df["_in_gt"] = (
            (df["Source"].str.upper() + "\t" + df["Target"].str.upper()).isin(gt_pairs)
        )
        return df

    def run(self):
        """Execute this section end-to-end (data generation, plotting, saving)."""
        models_to_plot = [
            "E7.5_rep1",
            "E8.5_rep1",
            "buffer_1",
            "buffer_2",
            "sample_1",
            "hepatocytes_1",
            "hepatocytes_3",
        ]

        stability_result_dir = RESULT_DIR / "stability_evaluation"


        stability_labeled_score_dfs, combined_stability_metric_df = self.load_tether_stability_results(stability_result_dir, models_to_plot)


        # Calculate the pairwise Jaccard indices for the top 10% of edges across subsamples for each sample
        sample_jaccard_indices = {}
        for sample_name, score_by_subsample in stability_labeled_score_dfs.items():
            sample_jaccard_indices[sample_name] = []
            available_subsamples = sorted(score_by_subsample.keys())

            for i, subsample_num_x in enumerate(available_subsamples):
                for subsample_num_y in available_subsamples[i + 1:]:
                    df_x = score_by_subsample[subsample_num_x]
                    df_y = score_by_subsample[subsample_num_y]

                    random_df_x = df_x.copy()
                    random_df_y = df_y.copy()

                    random_df_x["Score"] = random_df_x["Score"].sample(frac=1, random_state=42).reset_index(drop=True)
                    random_df_y["Score"] = random_df_y["Score"].sample(frac=1, random_state=42).reset_index(drop=True)

                    jaccard_index = self.calculate_jaccard_index(df_x, df_y)
                    random_jaccard_index = self.calculate_jaccard_index(random_df_x, random_df_y)

                    sample_jaccard_indices[sample_name].append((subsample_num_x, subsample_num_y, jaccard_index, random_jaccard_index))

        # Plot boxplots of Jaccard indices for each sample
        jaccard_plot_data = []
        for sample_name, jaccard_list in sample_jaccard_indices.items():
            for subsample_num_x, subsample_num_y, jaccard_index, random_jaccard_index in jaccard_list:
                jaccard_plot_data.append({
                    "method_name": OWN_MODEL_METHOD,
                    "sample_name": sample_name,
                    "subsample_pair": f"{subsample_num_x}-{subsample_num_y}",
                    "jaccard_index": jaccard_index,
                    "random_jaccard_index": random_jaccard_index
                })

        tether_jaccard_plot_df = pd.DataFrame(jaccard_plot_data)


        fig = self.plot_method_stability_by_sample_boxplot(tether_jaccard_plot_df, sample_order, sample_rename_map)
        fig.show()

        tether_jaccard_plot_df.groupby("sample_name")["jaccard_index"].median()

        other_method_stability_grn_dir = Path("/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.GRN_BENCHMARKING.MOELLER/multiGRNtools/stability_formatted_GRNs")


        other_method_stability_score_dfs = {}
        for method_name_dir in other_method_stability_grn_dir.iterdir():
            if method_name_dir.is_dir():
                method_name = method_name_dir.name
                print(f"Processing stability GRNs for method: {method_name}")

                other_method_stability_score_dfs[method_name] = {}
                for sample_name in models_to_plot:

                    cell_type = org_dict[sample_name][1]

                    sample_dir = method_name_dir / f"{cell_type}_{sample_name}"

                    # Load the merged ground truth
                    cell_type_cache_dir = DATA_DIR / f"{cell_type}_cache"
                    merged_ground_truth_df = pd.read_parquet(cell_type_cache_dir / f"{cell_type}_merged_ground_truth.parquet")

                    gt_tfs = set(merged_ground_truth_df["Source"].str.upper().unique())
                    gt_tgs = set(merged_ground_truth_df["Target"].str.upper().unique())
                    gt_pairs = (merged_ground_truth_df["Source"].str.upper() + "\t" + merged_ground_truth_df["Target"].str.upper()).drop_duplicates()

                    if sample_dir.exists() and sample_dir.is_dir():

                        other_method_stability_score_dfs[method_name][sample_name] = {}
                        for subsample_num in range(0, 10):
                            prediction_save_file = sample_dir / f"subsample_{subsample_num}.tsv"

                            if prediction_save_file.exists():
                                labeled_score_df = self.label_df(pd.read_csv(prediction_save_file, sep="\t"), gt_tfs, gt_tgs, gt_pairs)
                                other_method_stability_score_dfs[method_name][sample_name][subsample_num] = labeled_score_df
                            else:
                                logging.debug(f"Missing prediction file: {prediction_save_file}")

        print()
        other_method_jaccard_indices = {}
        for method_name, subsample_dict in other_method_stability_score_dfs.items():
            print(f"{method_name}")
            for sample_name, subsample_dict in subsample_dict.items():
                print(f"  {sample_name}")

                cell_type = org_dict[sample_name][1]

                for subsample_num, df in subsample_dict.items():
                    print(f"    Subsample {subsample_num}, Edges: {len(df):,} (True: {df['_in_gt'].sum():,}, False: {len(df) - df['_in_gt'].sum():,})")

        # Calculate the pairwise Jaccard indices for the top 10% of edges across subsamples for each sample
        for method_name, subsample_dict in other_method_stability_score_dfs.items():
            for sample_name, score_by_subsample in subsample_dict.items():
                other_method_jaccard_indices[method_name] = {}
                other_method_jaccard_indices[method_name][sample_name] = []
                available_subsamples = sorted(score_by_subsample.keys())

                for i, subsample_num_x in enumerate(available_subsamples):
                    for subsample_num_y in available_subsamples[i + 1:]:
                        if subsample_num_x != subsample_num_y:
                            df_x = score_by_subsample[subsample_num_x]
                            df_y = score_by_subsample[subsample_num_y]

                            random_df_x = df_x.copy()
                            random_df_y = df_y.copy()

                            random_df_x["Score"] = random_df_x["Score"].sample(frac=1, random_state=42).reset_index(drop=True)
                            random_df_y["Score"] = random_df_y["Score"].sample(frac=1, random_state=42).reset_index(drop=True)

                            jaccard_index = self.calculate_jaccard_index(df_x, df_y)
                            random_jaccard_index = self.calculate_jaccard_index(random_df_x, random_df_y)

                            other_method_jaccard_indices[method_name][sample_name].append((subsample_num_x, subsample_num_y, jaccard_index, random_jaccard_index))

        # Plot boxplots of Jaccard indices for each sample
        other_method_jaccard_plot_data = []
        for method_name, sample_dict in other_method_jaccard_indices.items():
            for sample_name, jaccard_list in sample_dict.items():
                for subsample_num_x, subsample_num_y, jaccard_index, random_jaccard_index in jaccard_list:
                    other_method_jaccard_plot_data.append({
                        "method_name": method_name,
                        "sample_name": sample_name,
                        "subsample_pair": f"{subsample_num_x}-{subsample_num_y}",
                        "jaccard_index": jaccard_index,
                        "random_jaccard_index": random_jaccard_index
                    })

        other_method_jaccard_plot_df = pd.DataFrame(other_method_jaccard_plot_data)

        jaccard_plot_df = pd.concat([tether_jaccard_plot_df, other_method_jaccard_plot_df], ignore_index=False)

        jaccard_plot_df.groupby("method_name").median("jaccard_index")

        jaccard_plot_df.to_csv(stability_result_dir / "stability_jaccard_indices_by_method.csv", index=False)

        plt.title("Stability", fontsize=16)
        sns.boxplot(
            data=jaccard_plot_df, 
            x="method_name", 
            y="jaccard_index", 
            showfliers=False,
            hue="method_name",
            )

        # Rename using sample_rename_map
        # plt.xticks(ticks=range(len(sample_order)), 
        #            labels=[sample_rename_map.get(sample, sample) for sample in sample_order], 
        #            rotation=45, ha="right", fontsize=12)

        plt.yticks(fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.ylabel("Pairwise Jaccard Index\n(Top 10% Edges, 10 subsamples)", fontsize=14)
        plt.xlabel("", fontsize=14)
        plt.ylim(0, 1)
        plt.show()
