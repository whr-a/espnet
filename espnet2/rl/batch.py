"""Native multimodal rollout tensors without a training-framework dependency."""

import pickle
from pathlib import Path

import numpy as np


class RolloutBatch:
    def __init__(self, tensors, non_tensors):
        self.batch = tensors
        self.non_tensor_batch = non_tensors
        sizes = {len(value) for value in [*tensors.values(), *non_tensors.values()]}
        if len(sizes) != 1:
            raise ValueError("All rollout columns must have the same batch dimension")

    @classmethod
    def from_dict(cls, tensors, non_tensors):
        return cls(tensors, non_tensors)

    def __len__(self):
        return len(next(iter(self.batch.values())))

    def select_idxs(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        return type(self)(
            {key: value[indices.tolist()] for key, value in self.batch.items()},
            {key: value[indices] for key, value in self.non_tensor_batch.items()},
        )

    def to(self, device):
        self.batch = {key: value.to(device) for key, value in self.batch.items()}
        return self

    def save_to_disk(self, path):
        with Path(path).open("wb") as stream:
            pickle.dump(self, stream)
