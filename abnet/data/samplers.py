"""Identity-balanced batch sampler.

The paper trains with "a batch size of 32 with each batch containing 8 person
and 4 clips for each person". That structure is what makes the batch-hard
triplet loss (Eq. 5) and the in-batch pair supervision for the Eq. 8 weighting
MLP work: every anchor is guaranteed both a positive and a negative.
"""

import copy
import math
import random
from collections import defaultdict
from typing import Dict, Iterator, List, Sequence

import numpy as np
import torch
from torch.utils.data import Sampler

from abnet.utils.dist import get_rank, get_world_size


class RandomIdentitySampler(Sampler):
    """Yield indices in blocks of ``num_instances`` clips per identity.

    Args:
        pids: identity label of every dataset element, in dataset order.
        batch_size: total clips per batch; must be divisible by
            ``num_instances``.
        num_instances: clips per identity (``K``), so ``P = batch_size // K``.
        seed: base seed; the epoch is mixed in via :meth:`set_epoch`.
    """

    def __init__(
        self,
        pids: Sequence[int],
        batch_size: int = 32,
        num_instances: int = 4,
        seed: int = 0,
    ):
        if batch_size % num_instances != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by "
                f"num_instances ({num_instances})"
            )
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances
        self.seed = seed
        self.epoch = 0

        self.index_by_pid: Dict[int, List[int]] = defaultdict(list)
        for index, pid in enumerate(pids):
            self.index_by_pid[int(pid)].append(index)
        self.pids = sorted(self.index_by_pid)

        if len(self.pids) < self.num_pids_per_batch:
            raise ValueError(
                f"need at least {self.num_pids_per_batch} identities for a batch of "
                f"{batch_size} with K={num_instances}, but the split has {len(self.pids)}"
            )

        # Each identity contributes floor(n/K)*K indices per epoch (min K).
        self._length = 0
        for pid in self.pids:
            count = len(self.index_by_pid[pid])
            count = max(count - count % self.num_instances, self.num_instances)
            self._length += count
        # Truncate to whole batches.
        self._length -= self._length % self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self._length

    def _build_batches(self) -> List[List[int]]:
        rng = random.Random(self.seed + self.epoch)

        # Per identity: shuffle, pad up to K if needed, chop into K-sized blocks.
        blocks_by_pid: Dict[int, List[List[int]]] = {}
        for pid in self.pids:
            indices = copy.deepcopy(self.index_by_pid[pid])
            if len(indices) < self.num_instances:
                indices = [
                    rng.choice(self.index_by_pid[pid]) for _ in range(self.num_instances)
                ]
            rng.shuffle(indices)
            blocks = [
                indices[i : i + self.num_instances]
                for i in range(0, len(indices) - self.num_instances + 1, self.num_instances)
            ]
            blocks_by_pid[pid] = blocks

        available = [pid for pid, blocks in blocks_by_pid.items() if blocks]
        batches: List[List[int]] = []
        while len(available) >= self.num_pids_per_batch:
            chosen = rng.sample(available, self.num_pids_per_batch)
            batch: List[int] = []
            for pid in chosen:
                batch.extend(blocks_by_pid[pid].pop(0))
                if not blocks_by_pid[pid]:
                    available.remove(pid)
            batches.append(batch)
        return batches

    def __iter__(self) -> Iterator[int]:
        batches = self._build_batches()
        flat = [index for batch in batches for index in batch]
        return iter(flat[: self._length])


class DistributedRandomIdentitySampler(RandomIdentitySampler):
    """P x K sampler that shards *whole batches* across ranks.

    Sharding at batch granularity (rather than at sample granularity) is what
    keeps the P x K structure intact on every rank, which the batch-hard triplet
    loss needs. ``batch_size`` here is the per-rank batch size.
    """

    def __init__(
        self,
        pids: Sequence[int],
        batch_size: int = 8,
        num_instances: int = 4,
        seed: int = 0,
        num_replicas: int = None,
        rank: int = None,
    ):
        super().__init__(pids, batch_size=batch_size, num_instances=num_instances, seed=seed)
        self.num_replicas = num_replicas if num_replicas is not None else get_world_size()
        self.rank = rank if rank is not None else get_rank()

        total_batches = self._length // self.batch_size
        self.num_batches_per_rank = total_batches // self.num_replicas
        self._length = self.num_batches_per_rank * self.batch_size
        if self.num_batches_per_rank == 0:
            raise ValueError(
                f"not enough identity-balanced batches ({total_batches}) for "
                f"{self.num_replicas} ranks; lower run.batch_size or sampler.num_instances"
            )

    def __iter__(self) -> Iterator[int]:
        batches = self._build_batches()
        usable = self.num_batches_per_rank * self.num_replicas
        batches = batches[:usable]
        mine = batches[self.rank :: self.num_replicas][: self.num_batches_per_rank]
        return iter([index for batch in mine for index in batch])


class InferenceSampler(Sampler):
    """Deterministic, contiguous shard of ``[0, size)`` for the current rank.

    Pads the tail by repeating indices so every rank sees the same count, which
    keeps ``all_gather`` shapes aligned; the evaluator drops the duplicates by
    sample index.
    """

    def __init__(self, size: int, num_replicas: int = None, rank: int = None):
        self.size = size
        self.num_replicas = num_replicas if num_replicas is not None else get_world_size()
        self.rank = rank if rank is not None else get_rank()
        self.num_samples = math.ceil(size / self.num_replicas)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        indices = np.arange(self.size)
        padded = self.num_samples * self.num_replicas
        if padded > self.size:
            indices = np.concatenate([indices, indices[: padded - self.size]])
        shard = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(shard.tolist())


def seed_worker(worker_id: int) -> None:
    """Give each dataloader worker its own numpy/random stream."""
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
