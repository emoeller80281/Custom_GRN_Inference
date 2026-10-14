import logging
import math
import torch

import numpy as np
from torch.utils.data import Sampler


class LengthGroupedBatchSampler(Sampler):
    """
    Groups samples by length, then shuffles the order of the batches.
    
    Each batch is padded to its longest sample, so grouping by length reduces the amount of
    padding for a given batch on average.
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
        self.lengths = np.asarray(lengths) # TF sequence lengths for each TF-DNA edge
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
        # Number of TF-DNA edges in the dataset 
        # (lengths is a 1D array of TF sequence lengths for each edge)
        n = len(self.lengths) 
        
        # Order for shuffled or sequential indices of the dataset
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(n) if self.shuffle else np.arange(n)

        # Sort the indices in each megabatch by length, then split into batches of batch_size
        batches = []
        for start in range(0, n, self.megabatch):
            # Sort the chunk indices by TF protein length
            chunk = order[start : start + self.megabatch]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            
            # Split the sorted chunk into batches of batch_size, and append to the list of batches
            for b in range(0, len(chunk), self.batch_size):
                batches.append(chunk[b : b + self.batch_size].tolist())

        # Shuffle the order of the sorted batches, so that the model sees different batches each epoch
        if self.shuffle:
            rng.shuffle(batches)

        # Split the batches across the ranks for distributed training, if applicable
        # Each rank needs to see the same number of batches for synchronization
        if self.num_replicas > 1:
            usable = (len(batches) // self.num_replicas) * self.num_replicas
            batches = batches[self.rank : usable : self.num_replicas]

        return batches

    def __iter__(self):
        batches = self._build_batches()
        
        self.epoch += 1
        yield from batches

    def __len__(self):
        """Number of batches in an epoch"""
        # Number of TF-DNA edges in the dataset
        n = len(self.lengths)
        
        # Number of batches that fit in the megabatch, plus any remainder that forms a smaller batch
        full, remainder = divmod(n, self.megabatch)
        n_batches = full * (self.megabatch // self.batch_size)
        if remainder:
            n_batches += math.ceil(remainder / self.batch_size)
        if self.num_replicas > 1:
            n_batches = n_batches // self.num_replicas
        return n_batches


class SampleBalancedSampler(Sampler):
    """Draw a source sample uniformly, then an edge uniformly inside it.

    Same distribution as a WeightedRandomSampler with 1/len(dataset) weights,
    but avoids torch.multinomial, which caps at 2^24 categories.
    """

    def __init__(self, datasets, seed, edges_per_sample=None):
        # Get the size of each sample's dataset
        self.sizes = torch.tensor([len(d) for d in datasets], dtype=torch.long)
        
        # Stores where each dataset starts in the concatenated dataset
        self.offsets = torch.cumsum(self.sizes, 0) - self.sizes
        
        # Specify number of edges to draw per sample (default is the size of the smallest sample)
        if edges_per_sample is None:
            edges_per_sample = int(self.sizes.min())
        if edges_per_sample <= 0:
            raise ValueError(f"edges_per_sample must be positive, got {edges_per_sample}")
        
        # Total number of samples to draw (each sample contributes edges_per_sample edges)
        self.balanced_edges_per_sample = edges_per_sample * len(self.sizes)
        
        # Generator advances to draw different batches each epoch, but is seeded so
        # that multiple runs will draw the same samples if they use the same seed.
        self.generator = torch.Generator().manual_seed(seed)
        logging.info(
            "Sample-balanced epoch: %s edges from each of %d samples = %s per epoch "
            "(%s training edges in total)",
            f"{edges_per_sample:,}", len(self.sizes),
            f"{self.balanced_edges_per_sample:,}", f"{int(self.sizes.sum()):,}",
        )

    def __len__(self):
        """Number of samples"""
        return self.balanced_edges_per_sample

    def __iter__(self):
        """
        Yield edge indices of the concatenated dataset, sampled uniformly from each sample.
        """
        # Randomly select a sample for each edge
        random_sample_num = torch.randint(
            low=0, high=len(self.sizes),            # Available sample indices to draw from
            size=(self.balanced_edges_per_sample,), # Number of edges to draw
            generator=self.generator
        )
        
        # For each sample, draw edges from within that sample
        within = (
            # Generate a random float in [0, 1) for each sample (random percent of the dataset)
            # e.g. torch.tensor([0.1, 0.5, 0.9]) for 3 samples
            torch.rand(
                self.balanced_edges_per_sample,
                generator=self.generator, 
                dtype=torch.double
            ) * self.sizes[random_sample_num] # multiplies by the size of the sample e.g. (0.1 * 985 = 98.5, 0.5 * 985 = 492.5, 0.9 * 985 = 886.5)
        ).long() # discards decimal, giving a row index within the sample (e.g. [98, 492, 886])
        
        # Yield one edge index at a time, which is the offset of the sample plus the row index within that sample
        # (dataloader will group these into batches of the specified batch size)
        yield from (self.offsets[random_sample_num] + within).tolist()