"""Section: evaluate TF-DNA binding model performance.

Refactored from ``test_tf_tg_predictions.ipynb`` (unmodified). This module
defines one child class of :class:`.base.TFTGBase`; the plotting helpers are
methods and the notebook's driver cells live in :meth:`run`.
"""
from .base import *  # noqa: F401,F403  (config, shared funcs, notebook imports)
from .base import TFTGBase

# Section-specific imports (hoisted from the notebook cells)
import scripts.train_tf_to_dna_model as tf_dna_train
from torch.utils.data import Dataset, DataLoader, Subset


class TFDNAModelEvaluation(TFTGBase):
    """Section: evaluate TF-DNA binding model performance."""

    def load_tf_dna_training_data(self, 
        cell_type: str,
        batch_size: int = 64,
    ):
        training_cache_dir = DATA_DIR / f"{cell_type}_cache"
        tf_dna_input_cache_dir = training_cache_dir / "tf_dna_training_cache"

        # Shared cache files for both TF-to-TG and TF-to-DNA training
        tf_embedding_cache_path = training_cache_dir / "tf_embeddings.pt"
        tf_mask_cache_path = training_cache_dir / "tf_masks.pt"

        # TF-DNA training specific cache files
        tf_dna_edge_tf_idx_cache_path = tf_dna_input_cache_dir / "edge_tf_idx.pt"
        tf_dna_edge_peak_idx_cache_path = tf_dna_input_cache_dir / "edge_peak_idx.pt"
        tf_dna_edge_labels_cache_path = tf_dna_input_cache_dir / "edge_labels.pt"
        tf_dna_tf_lengths_cache_path = tf_dna_input_cache_dir / "tf_lengths.pt"
        tf_dna_peak_onehot_cache_path = tf_dna_input_cache_dir / "peak_onehot_array.pt"

        tf_dna_train_idx_cache_path = tf_dna_input_cache_dir / "train_idx.pt"
        tf_dna_val_idx_cache_path = tf_dna_input_cache_dir / "val_idx.pt"
        tf_dna_test_idx_cache_path = tf_dna_input_cache_dir / "test_idx.pt"

        # Name to ID dictionaries
        tf_name_to_idx_cache_path = training_cache_dir / "tf_name_to_idx.csv"
        tf_dna_peak_id_to_idx_cache_path = tf_dna_input_cache_dir / "peak_id_to_idx.csv"

        # Load cached data
        edge_tf_idx_tensor: torch.Tensor = torch.load(tf_dna_edge_tf_idx_cache_path, weights_only=True)
        edge_peak_idx_tensor: torch.Tensor = torch.load(tf_dna_edge_peak_idx_cache_path, weights_only=True)
        edge_labels_tensor: torch.Tensor = torch.load(tf_dna_edge_labels_cache_path, weights_only=True)
        tf_embeddings_tensor: torch.Tensor = torch.load(tf_embedding_cache_path, weights_only=True)
        tf_mask_tensor: torch.Tensor = torch.load(tf_mask_cache_path, weights_only=True)
        peak_tensor: torch.Tensor = torch.load(tf_dna_peak_onehot_cache_path, weights_only=True)

        tf_name_to_idx_df = pd.read_csv(tf_name_to_idx_cache_path)
        peak_id_to_idx_df = pd.read_csv(tf_dna_peak_id_to_idx_cache_path)

        # Load train/val/test splits
        train_idx: torch.Tensor = torch.load(tf_dna_train_idx_cache_path, weights_only=True)
        val_idx: torch.Tensor = torch.load(tf_dna_val_idx_cache_path, weights_only=True)
        test_idx: torch.Tensor = torch.load(tf_dna_test_idx_cache_path, weights_only=True)

        if peak_tensor.dtype == torch.uint8:
            peak_tensor = peak_tensor.float()

        edge_dataset = tf_dna_train.TFPeakEdgeDataset(
            edge_tf_idx=edge_tf_idx_tensor,
            edge_peak_idx=edge_peak_idx_tensor,
            edge_labels=edge_labels_tensor,
            peak_tensor=peak_tensor,
        )

        train_dataset = Subset(edge_dataset, train_idx.tolist())
        val_dataset = Subset(edge_dataset, val_idx.tolist())
        test_dataset = Subset(edge_dataset, test_idx.tolist())

        # Create dataloaders for each split
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

        return {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "tf_embeddings_tensor": tf_embeddings_tensor,
            "tf_mask_tensor": tf_mask_tensor,
            "peak_tensor": peak_tensor,
            "edge_dataset": edge_dataset,
            "tf_name_to_idx_df": tf_name_to_idx_df,
            "peak_name_to_idx_df": peak_id_to_idx_df,
        }

    def create_tf_peak_index_to_name_mappings(self, training_data):
        #series to dict
        tf_name_to_idx_dict = training_data["tf_name_to_idx_df"].set_index("tf_name")["tf_idx"].to_dict()
        peak_name_to_idx_dict = training_data["peak_name_to_idx_df"].set_index("peak_id")["peak_idx"].to_dict()

        tf_idx_to_name = {idx: name for name, idx in tf_name_to_idx_dict.items()}
        peak_idx_to_name = {idx: name for name, idx in peak_name_to_idx_dict.items()}
        return tf_idx_to_name, peak_idx_to_name

    def tf_dna_binding_roc_plot(self, all_scores_flat, all_labels_flat, method_color_dict, organism_code):
        tf_dna_roc_prc_fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_aspect("equal")

        rng = np.random.default_rng(42)
        random_curve_plotted = False

        fpr, tpr, _ = roc_curve(all_labels_flat, all_scores_flat)
        auroc = roc_auc_score(all_labels_flat, all_scores_flat)

        ax.plot(
            fpr,
            tpr,
            lw=3,
            color=method_color_dict[OWN_MODEL_METHOD],
            label=f"AUROC = {auroc:.3f}",
        )

        # Plot one shuffled/random baseline only
        if not random_curve_plotted:
            rand_scores = rng.permutation(all_scores_flat)
            rand_fpr, rand_tpr, _ = roc_curve(all_labels_flat, rand_scores)
            rand_auroc = roc_auc_score(all_labels_flat, rand_scores)

            ax.plot(
                rand_fpr,
                rand_tpr,
                color="black",
                linestyle="--",
                lw=2,
                alpha=0.6,
                zorder=1,
                label=f"Random = {rand_auroc:.3f}",
            )

            random_curve_plotted = True


        ax.set_title("AUROC", fontsize=30)
        ax.set_xlabel("False Positive Rate", fontsize=20)
        ax.set_ylabel("True Positive Rate", fontsize=20)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.tick_params(axis="both", labelsize=16)

        ax.legend(
            loc="center",
            bbox_to_anchor=(0.5, -0.30),
            frameon=False,
            fontsize=20,
        )

        tf_dna_roc_prc_fig.subplots_adjust(
            left=0.10,
            right=0.72,
            bottom=0.10,
            top=0.90,
        )

        return tf_dna_roc_prc_fig

    def tf_dna_binding_prc_plot(self, all_scores_flat, all_labels_flat, method_color_dict, organism_code):
        tf_dna_prc_fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_aspect("equal")

        rng = np.random.default_rng(42)

        precision, recall, _ = precision_recall_curve(all_labels_flat, all_scores_flat)
        auprc = average_precision_score(all_labels_flat, all_scores_flat)

        ax.plot(
            recall,
            precision,
            lw=3,
            color=method_color_dict[OWN_MODEL_METHOD],
            label=f"AUPRC = {auprc:.3f}",
        )

        # Plot one shuffled/random baseline
        rand_scores = rng.permutation(all_scores_flat)
        rand_precision, rand_recall, _ = precision_recall_curve(all_labels_flat, rand_scores)
        rand_auprc = average_precision_score(all_labels_flat, rand_scores)

        ax.plot(
            rand_recall,
            rand_precision,
            color="black",
            linestyle="--",
            lw=2,
            alpha=0.6,
            zorder=1,
            label=f"Random = {rand_auprc:.3f}",
        )

        ax.set_title("AUPRC", fontsize=30)
        ax.set_xlabel("Recall", fontsize=20)
        ax.set_ylabel("Precision", fontsize=20)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.tick_params(axis="both", labelsize=16)

        ax.legend(
            loc="center",
            bbox_to_anchor=(0.5, -0.35),
            frameon=False,
            fontsize=20,
        )

        tf_dna_prc_fig.subplots_adjust(
            left=0.10,
            right=0.72,
            bottom=0.25,
            top=0.90,
        )

        return tf_dna_prc_fig

    def run(self):
        """Execute this section end-to-end (data generation, plotting, saving)."""
        organism_code = "mm10"
        cell_type = "mESC"
        sample_name = "E7.5_rep1"
        cell_type_cache_dir = DATA_DIR / f"{cell_type}_cache"

        training_data = self.load_tf_dna_training_data(
            cell_type=cell_type,
            batch_size=512,
        )

        tf_embeddings_tensor = training_data["tf_embeddings_tensor"]
        tf_mask_tensor = training_data["tf_mask_tensor"]

        tf_dna_model = utils.load_tf_dna_model(
            tf_dna_model_path=config.tf_dna_model_checkpoints[cell_type],
            tf_embeddings_tensor=tf_embeddings_tensor,
            tf_mask_tensor=tf_mask_tensor,
            compile_model=False
        )


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = tf_dna_model.model
        model = model.to(device)

        criterion = torch.nn.BCEWithLogitsLoss()
        score_threshold = 0.5
        pooling_mode = "lse"
        pooling_temperature = 1.0

        model.eval()

        total_loss = 0.0
        n_edges = 0

        tf_indices_list = []
        peak_indices_list = []
        all_scores = []
        all_labels = []
        plot_data = {}

        tf_idx_to_name, peak_idx_to_name = self.create_tf_peak_index_to_name_mappings(training_data)

        test_loader = training_data["test_loader"]

        # print(f"Evaluating on {dataset_split_type} set")
        with torch.inference_mode():
            for batch in tqdm(test_loader, desc="Evaluating", ncols=100):

                tf_idx = batch["tf_idx"].long()
                peak_idx = batch["peak_idx"].long()
                labels = batch["label"]

                tf_embedding = tf_embeddings_tensor[tf_idx].to(device, non_blocking=True)
                tf_mask = tf_mask_tensor[tf_idx].to(device, non_blocking=True)

                peak_embedding = batch["peak_embedding"].float().to(device, non_blocking=True)

                binding_logits = model.forward(
                    tf_embedding=tf_embedding,
                    tf_mask=tf_mask,
                    peak_embedding=peak_embedding,
                )

                scores = torch.sigmoid(binding_logits)

                all_scores.append(scores.detach().cpu().numpy().ravel())
                all_labels.append(labels.numpy().ravel())

                tf_indices_list.append(tf_idx.numpy().ravel())
                peak_indices_list.append(peak_idx.numpy().ravel())

        all_tf_indices = np.concatenate(tf_indices_list)
        all_peak_indices = np.concatenate(peak_indices_list)
        all_scores_flat = np.concatenate(all_scores)
        all_labels_flat = np.concatenate(all_labels)

        tf_names = [tf_idx_to_name[idx] for idx in all_tf_indices]
        peak_names = [peak_idx_to_name[idx] for idx in all_peak_indices]

        prediction_df = pd.DataFrame({
            "TF": tf_names,
            "DNA": peak_names,
            "Score": all_scores_flat,
            "Label": all_labels_flat
        })

        # Plot roc curve with scores and labels
        accuracy = accuracy_score(all_labels_flat, all_scores_flat > score_threshold)
        precision = precision_score(all_labels_flat, all_scores_flat > score_threshold)
        recall = recall_score(all_labels_flat, all_scores_flat > score_threshold)
        f1 = f1_score(all_labels_flat, all_scores_flat > score_threshold)
        auprc = average_precision_score(all_labels_flat, all_scores_flat)
        auroc = roc_auc_score(all_labels_flat, all_scores_flat)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Average Precision: {auprc:.4f}")
        print(f"ROC AUC: {auroc:.4f}")




        tf_dna_roc_fig = self.tf_dna_binding_roc_plot(
            all_scores_flat=all_scores_flat,
            all_labels_flat=all_labels_flat,
            method_color_dict=method_color_dict,
            organism_code=organism_code,
        )

        tf_dna_prc_fig = self.tf_dna_binding_prc_plot(
            all_scores_flat=all_scores_flat,
            all_labels_flat=all_labels_flat,
            method_color_dict=method_color_dict,
            organism_code=organism_code,
        )

        tf_dna_roc_fig_path = tf_dna_plots / f"roc_{organism_code}_tf_dna_vs_test_set.png"
        tf_dna_prc_fig_path = tf_dna_plots / f"prc_{organism_code}_tf_dna_vs_test_set.png"

        tf_dna_roc_fig.savefig(
            tf_dna_roc_fig_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.15,
        )

        tf_dna_prc_fig.savefig(
            tf_dna_prc_fig_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.15,
        )
