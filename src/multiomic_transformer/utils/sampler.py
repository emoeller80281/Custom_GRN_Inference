from torch.utils.data import Sampler
import math
import torch
import torch.distributed as dist

class DistributedWeightedSampler(Sampler):
    """
    Weighted sampler for DDP. Ensures each rank samples a distinct subset
    according to weights, with equal number of samples per rank.
    """

    def __init__(self, weights, num_samples, num_replicas=None, rank=None, replacement=True, seed=0):
        if num_replicas is None:
            if not dist.is_initialized():
                raise RuntimeError("Requires distributed package to be initialized")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_initialized():
                raise RuntimeError("Requires distributed package to be initialized")
            rank = dist.get_rank()

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = int(math.ceil(num_samples / num_replicas))
        self.total_size = self.num_samples * num_replicas
        self.num_replicas = num_replicas
        self.rank = rank
        self.replacement = replacement
        self.seed = seed

    def __iter__(self):
        # Ensure reproducibility across epochs
        g = torch.Generator()
        g.manual_seed(self.seed + dist.get_rank())

        # Sample with weights
        indices = torch.multinomial(self.weights, self.total_size, self.replacement, generator=g).tolist()

        # Subsample for this rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        return iter(indices)

    def __len__(self):
        return self.num_samples
