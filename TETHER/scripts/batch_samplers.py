"""Batch samplers shared by the TF-DNA and TF-TG trainers.

Kept in its own module on purpose. utils.py imports from
scripts/train_tf_to_tg_model.py at module level, and scripts/train_tf_to_dna_model.py
imports utils, so a TF-TG trainer that reached into the TF-DNA trainer for this class
would close the loop:

    train_tf_to_tg_model -> train_tf_to_dna_model -> utils -> train_tf_to_tg_model

and fail with "cannot import name ... from partially initialized module". This module
depends on nothing in the project, so both trainers can import it freely.
"""

import gc
import math

import numpy as np
from torch.utils.data import Sampler


class LengthGroupedBatchSampler(Sampler):
    """Batch edges whose TFs have similar protein length, so each batch can be cropped short.

    The TF encoder and cross-attention run over the padded table width (5,588 for mm10),
    while the average training edge sits behind a 558-residue protein -- about 10x of the
    work is padding. LitTFPeakBindingModel._shared_step crops each batch to its own longest
    real protein, but that only pays off if a batch's proteins are similar lengths: with
    globally shuffled edges a batch of 512 almost always contains one long protein, and the
    crop measured just 1.2x. Grouping by length first is what turns it into ~7.5x.

    Randomness is preserved at two levels so this is not simply "train in length order":
    edges are shuffled, cut into megabatches of `megabatch_multiplier` batches, sorted by
    length only *within* a megabatch, and the resulting batches are then shuffled again.
    A batch is therefore length-homogeneous while the epoch order stays random.

    What this does change: batch composition now correlates with protein length, so the
    three BatchNorm1d layers in the peak encoder see a different distribution per batch than
    under uniform shuffling, and gradient noise is no longer i.i.d. across batches. This is
    on unconditionally; to fall back to uniform shuffling, pass `shuffle=True` DataLoaders
    with `batch_size=` instead of `batch_sampler=` below. Checkpoints trained with grouping
    are not directly comparable to older ones -- check val AUROC before assuming they are.

    DDP: every rank builds the identical batch list (same seed and epoch) and then takes a
    strided slice of it, so ranks never share an edge and always get equal batch counts.
    Lightning must be told not to wrap this -- see use_distributed_sampler=False.
    """

    def __init__(
        self,
        lengths,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        num_replicas: int = 1,
        rank: int = 0,
        megabatch_multiplier: int = 64,
    ):
        self.lengths = np.asarray(lengths)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.megabatch = self.batch_size * int(megabatch_multiplier)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_batches(self):
        n = len(self.lengths)
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(n) if self.shuffle else np.arange(n)

        batches = []
        for start in range(0, n, self.megabatch):
            chunk = order[start : start + self.megabatch]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            for b in range(0, len(chunk), self.batch_size):
                batches.append(chunk[b : b + self.batch_size].tolist())

        if self.shuffle:
            rng.shuffle(batches)

        if self.num_replicas > 1:
            # Truncate to a multiple of world size: an uneven batch count deadlocks DDP at
            # the end of an epoch, because ranks synchronise per step.
            usable = (len(batches) // self.num_replicas) * self.num_replicas
            batches = batches[self.rank : usable : self.num_replicas]

        return batches

    def __iter__(self):
        batches = self._build_batches()
        # Lightning calls set_epoch on the batch sampler, but advance anyway so the order
        # still varies if it ever stops doing so. Every rank advances identically.
        self.epoch += 1
        yield from batches

    def __len__(self):
        n = len(self.lengths)
        full, remainder = divmod(n, self.megabatch)
        n_batches = full * (self.megabatch // self.batch_size)
        if remainder:
            n_batches += math.ceil(remainder / self.batch_size)
        if self.num_replicas > 1:
            n_batches = n_batches // self.num_replicas
        return n_batches


def dataloader_worker_init(worker_id: int) -> None:
    """Make a forked DataLoader worker ignore everything it inherited from the parent.

    torch.compile builds Triton/MLIR objects in the parent process, and each
    mlir::MLIRContext owns an llvm::StdThreadPool. DataLoader workers are forked, so a
    worker inherits those objects but none of the pool's threads. The worker never uses
    them -- but the first cyclic-GC pass that happens to collect one runs ~MLIRContext,
    which calls pthread_cond_destroy on a condition variable whose threads do not exist
    here, and the worker wedges in futex forever. Batches come back in strict round-robin
    order, so one wedged worker stops the whole loader: GPU utilisation falls to 0% and
    the caller sits in _try_get_data until the job is killed.

    The hang is timing-dependent -- it lands wherever the GC threshold happens to trip --
    which is why it hit different samples at different steps and looked in turn like an
    allocator, shared-memory, or GPU-contention problem.

    gc.freeze() moves every object alive at fork time into a permanent generation the
    collector never scans, so those destructors never run in the child. Objects the worker
    allocates afterwards are collected normally, so this does not leak.
    """
    gc.freeze()
