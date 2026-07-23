"""Section: GIF of the true/false score histogram across training epochs.

Refactored from ``test_tf_tg_predictions.ipynb`` (unmodified). This module
defines one child class of :class:`.base.TFTGBase`; the plotting helpers are
methods and the notebook's driver cells live in :meth:`run`.
"""
from .base import *  # noqa: F401,F403  (config, shared funcs, notebook imports)
from .base import TFTGBase

# Section-specific imports (hoisted from the notebook cells)
import imageio.v3 as iio


class TrainingHistogramGIF(TFTGBase):
    """Section: GIF of the true/false score histogram across training epochs."""

    def run(self):
        """Execute this section end-to-end (data generation, plotting, saving)."""
        model_cell_type = "Macrophage"
        model_training_sample = "buffer_3"

        chkpt_dir = utils.find_latest_checkpoint(model_cell_type, model_training_sample).parent
        print(f"Using checkpoint directory: {chkpt_dir.name}")

        chkpt_files = list(chkpt_dir.glob("epoch=*-val_auroc=*-val_loss=*.ckpt"))
        if not chkpt_files:
            logging.warning(f"No checkpoint files found for {model_training_sample} in {chkpt_dir.name}")

        chkpt_files.sort(key=lambda f: int(f.stem.split("-")[0].split("=")[1]), reverse=False)
        chkpt_nums = [int(f.stem.split("-")[0].split("=")[1]) for f in chkpt_files]

        # only plot every 10 epochs
        chkpt_files = [f for f in chkpt_files if int(f.stem.split("-")[0].split("=")[1]) % 10 == 0]
        print(f"Found {len(chkpt_files)} checkpoint files with epochs: {chkpt_nums[:2]} ... {chkpt_nums[-2:]}")

        per_epoch_plot_data = {}
        all_comparison_df_list = []
        subset_size = 3000
        for chkpt_file in tqdm(chkpt_files, desc="Evaluating Checkpoints", ncols=100):
            dataset_split_type = "val"

            epoch_num = int(chkpt_file.stem.split("-")[0].split("=")[1])

            tf_tg_model_checkpoints[model_cell_type][model_training_sample] = chkpt_file

            comparison_result = run_prediction_vs_test_set(
                tf_tg_model_checkpoints=tf_tg_model_checkpoints,
                model_cell_type=model_cell_type,
                model_training_sample=model_training_sample,
                test_set_cell_type=model_cell_type,
                evaluation_sample=model_training_sample,
                dataset_split_type=dataset_split_type,
                subset_size=subset_size,
                show_progress_bar=False,
            )

            metric_df = comparison_result["metric_df"]
            metric_df["epoch"] = epoch_num

            plot_data = comparison_result["plot_data"]

            all_labels_flat = plot_data[0]
            all_scores_flat = plot_data[1]

            title = f"{model_cell_type} {model_training_sample}\nEpoch {epoch_num}"

            per_epoch_plot_data[title] = (all_labels_flat, all_scores_flat)

            all_comparison_df_list.append(metric_df)

        all_epoch_df = pd.concat(all_comparison_df_list, ignore_index=True)


        plot_dir = PROJECT_DIR / "plots" / "histogram_per_epoch" / f"{model_cell_type}_{model_training_sample}"
        plot_dir.mkdir(parents=True, exist_ok=True)
        (plot_dir / "epoch_figs").mkdir(parents=True, exist_ok=True)

        for title in tqdm(per_epoch_plot_data.keys(), desc="Plotting Histograms", ncols=100):        
            labels = per_epoch_plot_data[title][0]
            scores = per_epoch_plot_data[title][1]

            epoch = int(title.split("\nEpoch ")[1])

            histogram_fig = plotting_utils.plot_score_histograms(
                labels=labels,
                scores=scores,
                n_bins=50,
                y_log=False,
                panel_kind="hist",
                density=False,
                title = title,
                y_lim=(0, 100),
                x_lim=(0, 1)
            )

            histogram_fig.savefig(plot_dir / "epoch_figs" / f"epoch_{epoch}.png")
            plt.close(histogram_fig)

        # Combine all histogram plots into a GIF in epoch order
        gif_path = plot_dir / f"{model_cell_type}_{model_training_sample}_histograms.gif"

        filenames = sorted(plot_dir.glob("epoch_figs/epoch_*.png"), key=lambda f: int(f.stem.split("_")[1]))
        images = [iio.imread(str(f)) for f in filenames]

        default_duration = 200
        # The lagging
        lag = 2000
        # Pause the GIF by extending the duration of the last frame
        duration = [default_duration] * (len(images)-1) + [lag]
        iio.imwrite(gif_path, images, duration=duration, loop=0)
