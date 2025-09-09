from sklearn.metrics import pairwise_distances_argmin
import numpy as np
import os

def _predict_one(npy_path: str, _CENTERS: np.ndarray):
    """Return (utt_id, 'id id id ...') for one utterance."""
    utt_id = os.path.splitext(os.path.basename(npy_path))[0]
    X = np.load(npy_path, mmap_mode="r")  # shape (T, D), float32
    X = X.T
    if X.size == 0:
        return utt_id, ""
    # nearest-centroid indices for all frames of this utterance
    ids = pairwise_distances_argmin(X, _CENTERS, metric="euclidean")  # (T,)
    return utt_id, " ".join(map(str, ids.tolist()))


centers_path = "/work/nvme/bbjs/bsu5/lrac_espnet/espnet/egs_lrac/lrac/codec1/centroids.npy"
_CENTERS = np.load(centers_path, mmap_mode="r").astype(np.float32)
wav_scp = "/work/nvme/bbjs/hwang41/lrac/espnet/egs_lrac/lrac/codec1/dump/raw/train_all/wav.scp"

with open(wav_scp, "r") as f:
    for line in f:
        utt_id, path = line.strip().split()
        npy_path = os.path.basename(path).replace(".wav", ".npy")
        npy_path = os.path.join("/work/nvme/bbjs/bsu5/lrac_espnet/espnet/egs_lrac/lrac/codec1/canary_embedding", npy_path)
        utt_id, ids = _predict_one(npy_path, _CENTERS)
        print(utt_id, ids)