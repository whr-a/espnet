#!/usr/bin/env python3
"""
K-Means clustering on encoder features, saving per-utterance embeddings.

- Reads wav.scp
- Extracts features with SemanticEncoder
- Saves each utterance embedding as {utt_id}.npy
- Runs MiniBatchKMeans on saved features
- Outputs centroids.npy and labels.txt
"""

import os
import io
import json
import argparse
import subprocess
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf
from sklearn.cluster import MiniBatchKMeans
from espnet2.gan_codec.shared.encoder.semantic_encoder.encoder_nemo import SemanticEncoder


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
# Semantic Encoder
# -----------------------------

def load_semantic_encoder(device: str):
    cfg_path = "/work/nvme/bbjs/hwang41/lrac/espnet/espnet2/gan_codec/shared/encoder/semantic_encoder/conf/1b_encoder.yaml"
    preproc_path = "/work/nvme/bbjs/hwang41/lrac/canary-1b-flash/split/preprocessor_standalone.pt"
    encoder_path = "/work/nvme/bbjs/hwang41/lrac/canary-1b-flash/split/encoder_standalone.pt"

    model = SemanticEncoder(cfg_path=cfg_path).to(device)
    model.load_preprocessor(preproc_path)
    model.load_encoder(encoder_path)
    model.eval()
    return model


# -----------------------------
# Feature extraction helpers
# -----------------------------

@torch.no_grad()
def encode_batch(encoder, signals_24k, lengths, device: str):
    sig = signals_24k.to(device)
    lens = lengths.to(device)
    encoded, out_lens = encoder(input_signal=sig, length=lens)
    return encoded, out_lens


def slice_to_time_feat_per_item(feat_b, out_len_i: int):
    if feat_b.ndim == 2:
        if feat_b.shape[0] >= feat_b.shape[1]:
            x = feat_b[:out_len_i, :]
        else:
            x = feat_b[:, :out_len_i].transpose(0, 1)
    else:
        x = feat_b.reshape(-1, feat_b.shape[-1])
    return x.detach().cpu().numpy()


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_scp", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_clusters", type=int, default=8192)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--feature_save_dir", type=str,
                        default="/work/hdd/bbjs/bsu5/lrac/canary_embedding",
                        help="Directory to save per-utterance features")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.feature_save_dir, exist_ok=True)

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    ds = WavScpDataset(args.wav_scp, target_sr=24000)
    dl = DataLoader(ds, batch_size=args.batch_size,
                    shuffle=False, num_workers=args.num_workers,
                    collate_fn=collate_pad)

    encoder = load_semantic_encoder(device)

    # Extract and save features + collect for kmeans
    # all_feats = []
    pbar = tqdm(dl, desc="Extracting + saving features")

    for utt_ids, signals_24k, lengths in pbar:
        # First check which utterances already have saved features
        feats_to_compute = []
        idx_to_compute = []
        for i, utt in enumerate(utt_ids):
            out_path = os.path.join(args.feature_save_dir, f"{utt}.npy")
            if os.path.exists(out_path):
                feat_i = np.load(out_path)
                # all_feats.append(feat_i)
            else:
                feats_to_compute.append((i, utt, out_path))
                idx_to_compute.append(i)

        # Skip encoder if everything was already cached
        if not feats_to_compute:
            continue

        # Encode only the batch items that need computation
        feats_b, out_lens = encode_batch(encoder, signals_24k, lengths, device)

        for i, utt, out_path in feats_to_compute:
            out_len_i = int(out_lens[i].item())
            feat_i = slice_to_time_feat_per_item(feats_b[i], out_len_i)
            if feat_i.size == 0:
                continue
            np.save(out_path, feat_i.astype(np.float32))
    #         all_feats.append(feat_i)

    # # Train KMeans
    # print("Stacking all features for KMeans...")
    # X = np.concatenate(all_feats, axis=0)
    # kmeans = MiniBatchKMeans(n_clusters=args.n_clusters,
    #                          random_state=0,
    #                          batch_size=4096,
    #                          n_init="auto")
    # kmeans.fit(X)
    # np.save(os.path.join(args.out_dir, "centroids.npy"),
    #         kmeans.cluster_centers_.astype(np.float32))

    # # Assign labels from saved features
    # labels_path = os.path.join(args.out_dir, "labels.txt")
    # with open(labels_path, "w", encoding="utf-8") as f:
    #     for utt_id, _ in ds.items:
    #         feat = np.load(os.path.join(args.feature_save_dir, f"{utt_id}.npy"))
    #         ids = kmeans.predict(feat)
    #         ids_str = " ".join(str(int(v)) for v in ids.tolist())
    #         f.write(f"{utt_id} {ids_str}\n")

    # print("Done. Features saved in", args.feature_save_dir)


if __name__ == "__main__":
    main()
