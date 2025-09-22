import os
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
import sc_multi_transformer
from transformer_dataset import TransformerDataset

#=================================== USER SETTINGS ===================================
# ----- User Settings -----
load_model = False
window_size = 800
num_cells = 1000
chrom_id = "chr19"
force_recalculate = True

atac_data_filename = "mESC_filtered_L2_E7.5_rep1_ATAC_processed.parquet"
rna_data_filename = "mESC_filtered_L2_E7.5_rep1_RNA_processed.parquet"

# ----- Model Configurations -----
# Logging
log_every_n_steps = 5

# Model Size
d_model = 256
tf_channels = 64
kernel_and_stride_size = 1

# Encoder Settings
encoder_nhead = 8
encoder_dim_feedforward = 1024
encoder_num_layers = 6
dropout = 0.1

# Train-test split
validation_fraction = 0.15

# Training configurations
epochs = 75
batch_size = 5
effective_batch_size = 16
warmup_steps = 100
learning_rate = 3e-4
patience = 5
epochs_before_patience = 3

#=====================================================================================

PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
MM10_GENOME_DIR = os.path.join(PROJECT_DIR, "data/reference_genome/mm10")
MM10_GENE_TSS_FILE = os.path.join(PROJECT_DIR, "data/genome_annotation/mm10/mm10_TSS.bed")
GROUND_TRUTH_DIR = os.path.join(PROJECT_DIR, "ground_truth_files")
SAMPLE_INPUT_DIR = os.path.join(PROJECT_DIR, "input/mESC/filtered_L2_E7.5_rep1")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output/transformer_testing_output")
DEBUG_FILE = os.path.join(PROJECT_DIR, "LOGS/transformer_training.debug")

MM10_FASTA_FILE = os.path.join(MM10_GENOME_DIR, f"{chrom_id}.fa")
MM10_CHROM_SIZES_FILE = os.path.join(MM10_GENOME_DIR, "chrom.sizes")

time_now = datetime.now().strftime("%d%m%y%H%M%S")

TRAINING_DIR=os.path.join(PROJECT_DIR, f"training_stats_{time_now}/")

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, f"checkpoints_{time_now}")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "training_stats"), exist_ok=True)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        
    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = torch.nn.HuberLoss()(output, targets, delta=1.0, reduction="mean")
        loss.backward()
        self.optimizer.step()
        
    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)
            
    def _save_checkpoint(self, epoch):
        ckp = self.model.state_dict()
        torch.save(ckp, "checkpoint.pt")
        print(f"Epoch {epoch} | Training checkpoint saved at checkpoint.pt")
        
    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
                
def load_train_objs():
    train_set = TransformerDataset()
    
if __name__ == "__main__":
    pass