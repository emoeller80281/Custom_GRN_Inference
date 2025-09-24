import os
import torch
import pandas as pd
import logging
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformer_2 import MultiomicTransformer
from transformer_2_dataset import MultiomicTransformerDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, get_rank, is_initialized, is_available

PROJECT_DIR = "/gpfs/Labs/Uzun/SCRIPTS/PROJECTS/2024.SINGLE_CELL_GRN_INFERENCE.MOELLER"
DATA_DIR = os.path.join(PROJECT_DIR, "dev/testing_scripts/transformer_data")

D_MODEL = 128
NUM_HEADS = 8
D_FF = 256
DROPOUT = 0.1

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
def setup_logging():
    if is_available() and is_initialized():
        rank = get_rank()
    else:
        rank = 0

    # Clear existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    level = logging.INFO if rank == 0 else logging.ERROR
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(message)s",
        handlers=[logging.StreamHandler()],
        force=True
    )
    
class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int,
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model, device_ids=[gpu_id])

    def _run_batch(self, batch, targets):
        atac_wins, tf_tensor = batch
        self.optimizer.zero_grad()
        output = self.model(atac_wins, tf_tensor)   # << your model signature
        loss = F.mse_loss(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = self.train_data.batch_size
        if self.gpu_id == 0:
            logging.info(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for atac_wins, tf_tensor, targets in self.train_data:
            atac_wins, tf_tensor, targets = atac_wins.to(self.gpu_id), tf_tensor.to(self.gpu_id), targets.to(self.gpu_id)
            self._run_batch((atac_wins, tf_tensor), targets)

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        if self.gpu_id == 0:
            logging.info(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

def load_train_objs():
    dataset = MultiomicTransformerDataset(
        data_dir=DATA_DIR,
        chrom_id="chr19"
    ) 
    model = MultiomicTransformer(
        D_MODEL, NUM_HEADS, D_FF, DROPOUT, 
        dataset.num_tf, dataset.num_windows, dataset.num_tg
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return dataset, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int, world_size: int, rank: int):
    train_frac, val_frac, test_frac = 0.7, 0.15, 0.15
    n_total = len(dataset)
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)
    n_test = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=DistributedSampler(train_set, num_replicas=world_size, rank=rank))
    val_loader   = DataLoader(val_set, batch_size=batch_size, sampler=DistributedSampler(val_set, num_replicas=world_size, rank=rank))
    test_loader  = DataLoader(test_set, batch_size=batch_size, sampler=DistributedSampler(test_set, num_replicas=world_size, rank=rank))

    return train_loader, val_loader, test_loader
    
def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    print(f"Hello from rank {rank}", flush=True)
    setup_logging()
    logging.info(f"DDP initialized for rank {rank}")
    
    if rank == 0:
        logging.info("Loading Training Objectives")
    dataset, model, optimizer = load_train_objs()
    if rank == 0:
        logging.info("Preparing DataLoader")
    train_loader, val_loader, test_loader = prepare_dataloader(dataset, batch_size, world_size, rank)
    if rank == 0:
        logging.info("Initializing Trainer")
    trainer = Trainer(model, train_loader, optimizer, rank, save_every)
    if rank == 0:
        logging.info("\n ----- Training -----")
    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == "__main__":
    total_epochs = 20
    save_every = 5
    batch_size = 2
    world_size = torch.cuda.device_count()
    print(world_size, flush=True)
    mp.spawn(main, args=(world_size, save_every, total_epochs, batch_size), nprocs=world_size)