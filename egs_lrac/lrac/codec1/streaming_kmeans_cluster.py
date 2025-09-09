#!/usr/bin/env python3
"""
Streaming MiniBatchKMeans over encoder features (no full-matrix in RAM).

Stages:
  --stage extract   : read wav.scp, run SemanticEncoder, save each utt as (T, D) .npy
  --stage kmeans    : stream all per-utt .npy, partial_fit MiniBatchKMeans
  --stage predict   : stream per-utt .npy to write cluster labels

Outputs:
  - <feature_save_dir>/<utt_id>.npy  (float32, shape (T, D))
  - <out_dir>/centroids.npy          (float32, shape (K, D))
  - <out_dir>/labels.txt             (utt_id + space-separated cluster ids)
"""

import os
import io
import json
import argparse
import subprocess
from typing import List, Tuple, Iterable

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf
from sklearn.cluster import MiniBatchKMeans
import pdb
from sklearn.metrics import pairwise_distances_argmin
from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------------
# Audio I/O helpers
# -----------------------------

def parse_wav_scp(wav_scp_path: str) -> List[Tuple[str, str]]:
    items = []
    with open(wav_scp_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            utt_id, source = line.split(maxsplit=1)
            items.append((utt_id, source))
    return items


def load_audio_from_source(source: str) -> Tuple[np.ndarray, int]:
    if source.endswith("|"):
        cmd = source[:-1].strip()
        data = subprocess.check_output(cmd, shell=True)
        with sf.SoundFile(io.BytesIO(data)) as f:
            audio = f.read(dtype="float32", always_2d=True)
            sr = f.samplerate
    else:
        audio, sr = sf.read(source, dtype="float32", always_2d=True)

    if audio.ndim == 2 and audio.shape[1] > 1:
        audio = audio.mean(axis=1, dtype="float32")
    else:
        audio = audio.squeeze(-1).astype("float32")
    return audio, sr


def resample_to_24k(wav: torch.Tensor, sr: int, target_sr: int = 24000) -> torch.Tensor:
    if sr == target_sr:
        return wav
    wav2d = wav.unsqueeze(0)
    wav_rs = torchaudio.functional.resample(wav2d, sr, target_sr)
    return wav_rs.squeeze(0)


# -----------------------------
# Dataset / Collate
# -----------------------------

class WavScpDataset(Dataset):
    def __init__(self, wav_scp_path: str, target_sr: int = 24000):
        self.items = parse_wav_scp(wav_scp_path)
        self.target_sr = target_sr

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        utt_id, source = self.items[idx]
        audio_np, sr = load_audio_from_source(source)
        wav = torch.from_numpy(audio_np)
        wav_24k = resample_to_24k(wav, sr, self.target_sr)
        return utt_id, wav_24k


def collate_pad(batch):
    utt_ids, waves = zip(*batch)
    lengths = torch.tensor([w.shape[0] for w in waves], dtype=torch.long)
    padded = nn.utils.rnn.pad_sequence(waves, batch_first=True)
    return list(utt_ids), padded, lengths

# -----------------------------
# Streaming utilities
# -----------------------------

def iter_feature_files(feature_dir: str, shuffle: bool = False) -> Iterable[str]:
    files = [os.path.join(feature_dir, f) for f in os.listdir(feature_dir) if f.endswith(".npy")]
    if shuffle:
        rng = np.random.default_rng(0)
        rng.shuffle(files)
    return files


def stream_frames_in_chunks(np_path: str, max_frames: int) -> Iterable[np.ndarray]:
    """
    Lazily stream a .npy feature file in slices along time axis.
    Assumes file saved as (D, T). Returns chunks of shape (t, D).
    """
    X = np.load(np_path, mmap_mode="r")
    X = X.T
    if X.ndim != 2:
        return
    T = X.shape[0]
    for s in range(0, T, max_frames):
        yield X[s:s + max_frames]

def buffered_batches(files, mb_batch_frames: int, init_min_frames: int):
    """
    Yield batches for MiniBatchKMeans.partial_fit with a guarantee that the
    FIRST yielded batch has at least init_min_frames rows (>= n_clusters).
    Subsequent batches are ~mb_batch_frames.
    """
    buf = []
    total = 0
    initialized = False

    def flush(target_size):
        nonlocal buf, total
        if total == 0:
            return None
        X = np.concatenate(buf, axis=0)  # (total, D)
        buf, total = [], 0
        # Split into chunks of ~target_size
        for s in range(0, X.shape[0], target_size):
            yield X[s:s+target_size]

    for npy_path in files:
        for chunk in stream_frames_in_chunks(npy_path, mb_batch_frames):
            if chunk.size == 0:
                continue
            buf.append(chunk)          # chunk is (t, D)
            total += chunk.shape[0]

            # before init: require a big enough batch
            if not initialized and total >= init_min_frames:
                for B in flush(mb_batch_frames):
                    initialized = True
                    yield B
            # after init: flush when we have at least one batch
            elif initialized and total >= mb_batch_frames:
                for B in flush(mb_batch_frames):
                    yield B

    # leftover
    if total > 0:
        if not initialized:
            # last resort: if dataset is too small, still try once
            yield np.concatenate(buf, axis=0)
        else:
            yield np.concatenate(buf, axis=0)

# ---- Globals for worker processes ----
_CENTERS = None  # numpy array (K, D)

def _init_predict_worker(centers_path: str):
    """Initializer for worker processes: load centroids once (read-only)."""
    global _CENTERS
    _CENTERS = np.load(centers_path, mmap_mode="r").astype(np.float32)

def _predict_one(npy_path: str):
    """Return (utt_id, 'id id id ...') for one utterance."""
    global _CENTERS
    utt_id = os.path.splitext(os.path.basename(npy_path))[0]
    X = np.load(npy_path, mmap_mode="r")  # shape (T, D), float32
    X = X.T
    if X.size == 0:
        return utt_id, ""
    # nearest-centroid indices for all frames of this utterance
    ids = pairwise_distances_argmin(X, _CENTERS, metric="euclidean")  # (T,)
    return utt_id, " ".join(map(str, ids.tolist()))

# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    # IO
    parser.add_argument("--wav_scp", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--feature_save_dir", type=str, required=True,
                        help="Directory to save per-utterance features (T, D) float32")

    # Stages
    parser.add_argument("--stage", type=str, default="kmeans",
                        choices=["extract", "kmeans", "predict", "all"])

    # Compute
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--target_sr", type=int, default=24000)

    # KMeans params
    parser.add_argument("--n_clusters", type=int, default=8192)
    parser.add_argument("--mb_batch_frames", type=int, default=819200,
                        help="Frames per partial_fit (controls RAM).")
    parser.add_argument("--passes", type=int, default=8,
                        help="Number of streaming passes over the dataset.")
    parser.add_argument("--sk_batch_size", type=int, default=4096,
                        help="MiniBatchKMeans internal mini-batch size.")

    # Prediction
    parser.add_argument("--labels_filename", type=str, default="labels.txt")

    # Misc
    parser.add_argument("--shuffle_files", action="store_true",
                        help="Shuffle feature file order per pass (recommended).")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.feature_save_dir, exist_ok=True)

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # ---------------------------------
    # Stage: kmeans (streaming)
    # ---------------------------------
    if args.stage in ("kmeans", "all"):
        print("[STAGE] kmeans -> training MiniBatchKMeans by streaming frames")

        files = list(iter_feature_files(args.feature_save_dir, shuffle=args.shuffle_files))
        if not files:
            raise RuntimeError(f"No .npy features found in {args.feature_save_dir}")

        # Infer D from the first file
        probe = np.load(files[0], mmap_mode="r")
        if probe.ndim != 2:
            raise RuntimeError(f"Bad feature shape in {files[0]}: {probe.shape}")
        D = probe.shape[0]
        print(f"[INFO] Feature dimension D = {D}")

        kmeans = MiniBatchKMeans(
            n_clusters=args.n_clusters,
            random_state=0,
            batch_size=args.sk_batch_size,
            n_init="auto",
            compute_labels=False,
            reassignment_ratio=0.0,   # stable for large K
        )

        init_min_frames = max(args.n_clusters, args.mb_batch_frames)  # critical!
        for ep in range(args.passes):
            if args.shuffle_files:
                rng = np.random.default_rng(seed=ep)
                rng.shuffle(files)
            pbar = tqdm(files, desc=f"Pass {ep+1}/{args.passes}")
            # We’ll iterate pbar just to show progress, but feed KMeans from the buffered generator
            # Create a *separate* iterator over the same file list:
            file_iter = iter(files)
            batch_iter = buffered_batches(file_iter, args.mb_batch_frames, init_min_frames)

            for B in batch_iter:
                # print(B.shape)
                # First partial_fit will now see at least n_clusters samples
                kmeans.partial_fit(B)
            # advance pbar to full to complete the visual for this pass
            pbar.update(len(files) - pbar.n)
            pbar.close()

        centroids_path = os.path.join(args.out_dir, "centroids.npy")
        np.save(centroids_path, kmeans.cluster_centers_.astype(np.float32))
        print(f"[STAGE] kmeans -> saved {centroids_path}")

        # # stash the trained model for the next stage
        # # (you can also pickle if you prefer)
        # np.save(os.path.join(args.out_dir, "_kmeans_inertia.npy"),
        #         np.array([kmeans.inertia_], dtype=np.float64))

        # also save a tiny JSON metadata
        meta = {
            "n_clusters": int(args.n_clusters),
            "D": int(D),
            "passes": int(args.passes),
            "mb_batch_frames": int(args.mb_batch_frames),
            "sk_batch_size": int(args.sk_batch_size),
        }
        with open(os.path.join(args.out_dir, "kmeans_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print("[STAGE] kmeans -> done")

    # ---------------------------------
    # Stage: predict (streaming)
    # ---------------------------------
    if args.stage in ("predict", "all"):
        print("[STAGE] predict -> assigning labels per utterance")

        centroids_path = os.path.join(args.out_dir, "centroids.npy")
        if not os.path.exists(centroids_path):
            raise FileNotFoundError(f"Missing centroids at {centroids_path}. Run --stage kmeans or all first.")
        centers = np.load(centroids_path, mmap_mode="r")

        labels_path = os.path.join(args.out_dir, args.labels_filename)
        files = list(iter_feature_files(args.feature_save_dir, shuffle=False))

        # You can expose this as a CLI arg; here we pick sensible default
        predict_workers = min(8, os.cpu_count() or 1)

        # Warm probe to print D (optional)
        probe = np.load(files[0], mmap_mode="r")
        if probe.ndim != 2:
            raise RuntimeError(f"Bad feature shape in {files[0]}: {probe.shape}")
        print(f"[INFO] Predict on {len(files)} utterances, feature dim D={probe.shape[1]}, workers={predict_workers}")

        # Run workers; main process does the single-file write
        with open(labels_path, "w", encoding="utf-8") as fout:
            with ProcessPoolExecutor(max_workers=predict_workers,
                                    initializer=_init_predict_worker,
                                    initargs=(centroids_path,)) as ex:
                futures = {ex.submit(_predict_one, p): p for p in files}
                for fut in tqdm(as_completed(futures), total=len(futures), desc="Predicting"):
                    utt_id, ids_str = fut.result()
                    if ids_str:
                        fout.write(f"{utt_id} {ids_str}\n")
                    else:
                        fout.write(f"{utt_id}\n")
        print(f"[STAGE] predict -> wrote {labels_path}")
        print("Done.")


if __name__ == "__main__":
    main()
