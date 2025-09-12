#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Distill a student ConformerEncoder (configured via OmegaConf YAML) to Canary encoder embeddings.

Example:
python distill_canary_encoder_nemo_conformer_omegaconf.py \
  --wav_scp /work/nvme/bbjs/hwang41/lrac/espnet/egs_lrac/lrac/codec1/dump/raw/train_all/wav.scp \
  --emb_dir /work/nvme/bbjs/bsu5/lrac_espnet/espnet/egs_lrac/lrac/codec1/canary_embedding \
  --student_cfg ./student_conformer.yaml \
  --preproc_pt /work/nvme/bbjs/hwang41/lrac/canary-1b-flash/split/preprocessor_standalone.pt \
  --out_dir ./distill_runs/s_canary_conformer_stream \
  --epochs 10 --batch_size 16 --lr 2e-4 --amp

Where student_conformer.yaml looks like:

preprocessor:
  _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
  sample_rate: 16000
  normalize: per_feature
  window_size: 0.025
  window_stride: 0.01
  window: hann
  features: 128
  n_fft: 512
  log: true
  frame_splicing: 1
  dither: 1.0e-05
  pad_to: 0
  pad_value: 0.0

encoder:
  _target_: nemo.collections.asr.modules.ConformerEncoder
  feat_in: 128
  feat_out: -1
  n_layers: 20
  d_model: 448
  subsampling: dw_striding
  subsampling_factor: 8
  subsampling_conv_channels: 128
  causal_downsampling: true
  ff_expansion_factor: 3
  self_attention_model: rel_pos
  n_heads: 8
  att_context_size: [-1, 4]   # streaming look-ahead after subsampling
  xscaling: false
  untie_biases: true
  pos_emb_max_len: 5000
  conv_kernel_size: 15
  conv_norm_type: batch_norm
  conv_context_size: [14, 0]  # causal conv (left, right) when K=15 -> [14,0]
  dropout: 0.1
  dropout_pre_encoder: 0.1
  dropout_emb: 0.0
  dropout_att: 0.1
"""

import os
import math
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from omegaconf import OmegaConf

# NeMo
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor, ConformerEncoder


# -------------------------
# I/O utilities
# -------------------------

def read_wav_scp(path: str) -> Dict[str, str]:
    table = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            utt, wav = parts
            table[utt] = wav
    return table


def intersect_with_embeddings(wav_map: Dict[str, str], emb_dir: str) -> List[Tuple[str, str, str]]:
    triples = []
    for utt, wavp in wav_map.items():
        embp = os.path.join(emb_dir, f"{utt}.npy")
        if os.path.isfile(embp):
            triples.append((utt, wavp, embp))
    if not triples:
        raise RuntimeError(f"No overlapping utterances between wav.scp and {emb_dir}")
    return triples


# -------------------------
# Canary preprocessor loader (OmegaConf + .pt)
# -------------------------

def load_nemo_preprocessor_from_cfg_and_pt(
    preproc_cfg: dict,
    preproc_pt: Optional[str] = None
) -> AudioToMelSpectrogramPreprocessor:
    """
    Build the AudioToMelSpectrogramPreprocessor from the student config.
    If preproc_pt is provided, try to load state dict (and overwrite cfg if the .pt bundles it).
    """
    if preproc_pt is None:
        pre = AudioToMelSpectrogramPreprocessor.from_config_dict(preproc_cfg)
        pre.eval()
        return pre

    blob = torch.load(preproc_pt, map_location="cpu")

    # If the .pt bundles its own cfg, prefer it for absolute reproducibility
    if isinstance(blob, dict) and "state_dict" in blob and "cfg" in blob:
        pre = AudioToMelSpectrogramPreprocessor.from_config_dict(blob["cfg"])
        pre.load_state_dict(blob["state_dict"], strict=True)
        pre.eval()
        return pre

    # Else: instantiate from the student config, load the provided state dict
    pre = AudioToMelSpectrogramPreprocessor.from_config_dict(preproc_cfg)
    sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    if sd is not None:
        pre.load_state_dict(sd, strict=False)  # allow False in case the .pt has only buffers
    pre.eval()
    return pre


# -------------------------
# Dataset & collate (raw wave + teacher embedding)
# -------------------------

class DistillDataset(Dataset):
    def __init__(self, triples: List[Tuple[str, str, str]], sample_rate: int = 16000, max_audio_sec: float = None):
        self.triples = triples
        self.sr = sample_rate
        self.max_audio_sec = max_audio_sec

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx: int):
        utt, wavp, embp = self.triples[idx]
        wav, sr = torchaudio.load(wavp)  # [C, T]
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        wav = wav.mean(dim=0, keepdim=True)  # [1, T]

        if self.max_audio_sec is not None:
            max_T = int(self.max_audio_sec * self.sr)
            if wav.size(-1) > max_T:
                wav = wav[:, :max_T]

        targ = torch.from_numpy(np.load(embp)).float()  # [T_teacher, D_teacher]
        return {"utt": utt, "wav": wav, "targ": targ}


def pad_1d_wavs(wavs: List[torch.Tensor]):
    B = len(wavs)
    Tm = max(x.size(-1) for x in wavs)
    out = wavs[0].new_zeros((B, 1, Tm))
    lens = []
    for i, x in enumerate(wavs):
        t = x.size(-1)
        out[i, 0, :t] = x
        lens.append(t)
    return out, torch.tensor(lens, dtype=torch.long)


def collate_fn(batch):
    utts  = [b["utt"] for b in batch]
    wavs  = [b["wav"] for b in batch]      # [1, T]
    targs = [b["targ"].transpose(0, 1) for b in batch]     # [T_t, D]
    wav_pad, wav_len = pad_1d_wavs(wavs)   # [B, 1, T], [B]
    return {"utt": utts, "wav": wav_pad, "wav_len": wav_len, "targs": targs}


# -------------------------
# Front-end wrapper (NeMo preprocessor)
# -------------------------

class NemoFrontend(nn.Module):
    """
    Wrap Canary's AudioToMelSpectrogramPreprocessor.
    Input:  wav [B, 1, T], wav_len [B] (samples)
    Output: feats [B, Tm, 128], feat_len [B] (mel frames)
    """
    def __init__(self, nemo_preproc: AudioToMelSpectrogramPreprocessor):
        super().__init__()
        self.pp = nemo_preproc

    @torch.no_grad()
    def forward(self, wav: torch.Tensor, wav_len: torch.Tensor):
        feats, feat_lens = self.pp(input_signal=wav, length=wav_len)  # [B, 128, Tm], [B]
        # feats = feats.transpose(1, 2).contiguous()                    # [B, Tm, 128]
        return feats, feat_lens


# -------------------------
# Student model: ConformerEncoder + projection head
# -------------------------

class StudentConformerModel(nn.Module):
    def __init__(
        self,
        nemo_preproc: AudioToMelSpectrogramPreprocessor,
        encoder_cfg: dict,
        teacher_dim: int = 1024,
    ):
        super().__init__()
        self.frontend = NemoFrontend(nemo_preproc)

        # Ensure minimal required fields exist or provide sensible defaults
        enc_cfg = dict(encoder_cfg)  # shallow copy
        enc_cfg.setdefault("feat_in", 128)
        enc_cfg.setdefault("feat_out", -1)  # NeMo will use d_model if -1
        enc_cfg.setdefault("n_layers", enc_cfg.get("num_layers", 12))
        enc_cfg.setdefault("d_model", 448)
        enc_cfg.setdefault("n_heads", 8)
        enc_cfg.setdefault("subsampling", "dw_striding")
        enc_cfg.setdefault("subsampling_factor", 8)
        enc_cfg.setdefault("subsampling_conv_channels", 128)
        enc_cfg.setdefault("ff_expansion_factor", 3)
        enc_cfg.setdefault("self_attention_model", "rel_pos")
        enc_cfg.setdefault("att_context_size", [-1, -1])
        enc_cfg.setdefault("dropout", 0.1)
        enc_cfg.setdefault("dropout_att", 0.1)
        enc_cfg.setdefault("dropout_pre_encoder", 0.1)
        enc_cfg.setdefault("dropout_emb", 0.0)
        enc_cfg.setdefault("xscaling", False)
        enc_cfg.setdefault("untie_biases", True)
        enc_cfg.setdefault("pos_emb_max_len", 5000)
        enc_cfg.setdefault("conv_kernel_size", 15)
        enc_cfg.setdefault("conv_norm_type", "batch_norm")
        # Streaming-compatible fields (if given)
        if "causal_downsampling" not in enc_cfg:
            enc_cfg["causal_downsampling"] = False
        # conv_context_size optional: [left, right] for depthwise conv
        # If absent, the module uses default (often symmetric); for streaming supply [K-1, 0].

        # Instantiate the ConformerEncoder from the dict
        try:
            self.encoder = ConformerEncoder.from_config_dict(enc_cfg)
        except Exception as e:
            raise RuntimeError(
                "Failed to instantiate ConformerEncoder from the provided encoder cfg. "
                f"Check for field name mismatches with your NeMo version. Error:\n{e}"
            )

        # Determine output dim for projection head
        feat_out = enc_cfg.get("feat_out", -1)
        d_out = self.encoder.d_model if feat_out in (-1, None) else feat_out
        self.proj = nn.Linear(d_out, teacher_dim)

    def forward(self, wav: torch.Tensor, wav_len: torch.Tensor):
        feats, feats_len = self.frontend(wav, wav_len)      # [B, Tm, 128], [B]
        y, y_len = self.encoder(audio_signal=feats, length=feats_len)    # [B, T', d_out], [B]
        y = self.proj(y.transpose(1, 2)).transpose(1, 2)                                    # [B, T', teacher_dim]
        return y, y_len


# -------------------------
# Distillation loss (MSE + 1 - cosine) with interpolation
# -------------------------

def interpolate_time(x: torch.Tensor, target_T: int) -> torch.Tensor:
    if x.size(0) == target_T:
        return x
    x_ = x.transpose(0, 1).unsqueeze(0)  # [1, D, T]
    x_ = F.interpolate(x_, size=target_T, mode="linear", align_corners=False)
    return x_.squeeze(0).transpose(0, 1).contiguous()


def distill_loss(student: torch.Tensor, targs_list: List[torch.Tensor], stud_len: torch.Tensor,
                 mse_w: float = 1.0, cos_w: float = 1.0):
    B, Tprime, D = student.shape
    device = student.device
    total_mse, total_cos, count = 0.0, 0.0, 0

    for b in range(B):
        T_s = int(stud_len[b].item())
        if T_s <= 0:
            continue
        s = student[b, :T_s]      # [T_s, D]
        t = targs_list[b].to(device)
        t = interpolate_time(t, T_s)

        mse = F.mse_loss(s, t)
        s_norm = F.normalize(s, dim=-1)
        t_norm = F.normalize(t, dim=-1)
        cos = 1.0 - (s_norm * t_norm).sum(dim=-1).mean()

        total_mse += mse
        total_cos += cos
        count += 1

    if count == 0:
        return student.new_tensor(0.0), {"mse": 0.0, "cos": 0.0}

    loss = mse_w * (total_mse / count) + cos_w * (total_cos / count)
    return loss, {"mse": (total_mse / count).item(), "cos": (total_cos / count).item()}


# -------------------------
# Train loop
# -------------------------

def save_ckpt(state: dict, out_dir: str, name: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    torch.save(state, path)
    return path


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[Info] Using device: {device}")

    # Data
    wav_map = read_wav_scp(args.wav_scp)
    triples = intersect_with_embeddings(wav_map, args.emb_dir)

    # Split
    if args.valid_ratio > 0.0:
        n_total = len(triples)
        n_valid = max(1, int(n_total * args.valid_ratio))
        triples_train = triples[:-n_valid]
        triples_valid = triples[-n_valid:]
    else:
        triples_train, triples_valid = triples, []

    ds_train = DistillDataset(triples_train, sample_rate=24000, max_audio_sec=args.max_audio_sec)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    dl_valid = None
    if len(triples_valid) > 0:
        ds_valid = DistillDataset(triples_valid, sample_rate=24000, max_audio_sec=args.max_audio_sec)
        dl_valid = DataLoader(ds_valid, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # Load student config (OmegaConf)
    if args.student_cfg is None:
        raise RuntimeError("--student_cfg is required (YAML with 'preprocessor:' and 'encoder:' blocks).")
    student_cfg = OmegaConf.load(args.student_cfg)
    if "preprocessor" not in student_cfg or "encoder" not in student_cfg:
        raise RuntimeError("student_cfg must contain both 'preprocessor:' and 'encoder:' sections.")

    # Preprocessor (Canary-compatible)
    nemo_preproc = load_nemo_preprocessor_from_cfg_and_pt(
        preproc_cfg=OmegaConf.to_container(student_cfg.preprocessor, resolve=True),
        preproc_pt=args.preproc_pt
    ).to(device)
    nemo_preproc.eval()

    # Model
    model = StudentConformerModel(
        nemo_preproc=nemo_preproc,
        encoder_cfg=OmegaConf.to_container(student_cfg.encoder, resolve=True),
        teacher_dim=args.teacher_dim
    ).to(device)

    accum_steps = 4

    # Optim
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp))

    best_val = float("inf")
    global_step = 0
    accum_steps = max(1, accum_steps)
    # for logging smoothness across accumulation
    running = {"loss": 0.0, "mse": 0.0, "cos": 0.0}
    logged_updates = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"loss": 0.0, "mse": 0.0, "cos": 0.0}
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)

        optimizer.zero_grad(set_to_none=True)
        for ibatch, batch in enumerate(pbar):
            wav = batch["wav"].squeeze(1).to(device)         # [B, 1, T]
            wav_len = batch["wav_len"].to(device) # [B]
            targs = batch["targs"]
            print('Max length of targs:', max(wav_len))

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.amp)):
                y, y_len = model(wav, wav_len)    # [B, D, T'], [B]
                loss, stats = distill_loss(y.transpose(1, 2), targs, y_len, mse_w=args.mse_w, cos_w=args.cos_w)

            # scale by accumulation steps
            loss_to_backprop = loss / accum_steps
            scaler.scale(loss_to_backprop).backward()

            # step only on accumulation boundary
            step_boundary = ((ibatch + 1) % accum_steps == 0)
            if step_boundary:
                if args.grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # logging once per optimizer update
                running["loss"] += loss.item()
                running["mse"]  += stats["mse"]
                running["cos"]  += stats["cos"]
                logged_updates  += 1

                pbar.set_postfix({
                    "loss": f"{running['loss']/max(1,logged_updates):.4f}",
                    "mse":  f"{running['mse']/max(1,logged_updates):.4f}",
                    "cos":  f"{running['cos']/max(1,logged_updates):.4f}",
                    "gs":   global_step
                })

        # Save per-epoch
        ckpt_path = save_ckpt({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "args": vars(args),
            "student_cfg": OmegaConf.to_container(student_cfg, resolve=True),
        }, args.out_dir, f"epoch{epoch:03d}.pt")
        print(f"[Info] Saved: {ckpt_path}")

        # Validation
        if dl_valid is not None:
            model.eval()
            val_loss = val_mse = val_cos = 0.0
            n = 0
            with torch.no_grad():
                for batch in tqdm(dl_valid, desc="Valid", dynamic_ncols=True):
                    wav = batch["wav"].squeeze(1).to(device)
                    wav_len = batch["wav_len"].to(device)
                    targs = batch["targs"]
                    y, y_len = model(wav, wav_len)
                    loss, stats = distill_loss(y.transpose(1, 2), targs, y_len, mse_w=args.mse_w, cos_w=args.cos_w)
                    val_loss += loss.item(); val_mse += stats["mse"]; val_cos += stats["cos"]; n += 1
            val_loss /= max(1, n); val_mse /= max(1, n); val_cos /= max(1, n)
            print(f"[Valid] loss={val_loss:.4f} mse={val_mse:.4f} cos={val_cos:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                best_path = save_ckpt({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "args": vars(args),
                    "val_loss": val_loss,
                    "student_cfg": OmegaConf.to_container(student_cfg, resolve=True),
                }, args.out_dir, "best.pt")
                print(f"[Info] New best: {best_val:.4f} -> {best_path}")


# -------------------------
# CLI
# -------------------------

def get_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--wav_scp", type=str, required=True)
    p.add_argument("--emb_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    # Configs
    p.add_argument("--student_cfg", type=str, required=True,
                   help="OmegaConf YAML with 'preprocessor:' and 'encoder:' blocks")
    p.add_argument("--preproc_pt", type=str, default=None,
                   help="Optional preprocessor_standalone.pt to ensure Canary-exact frontend")

    # Train
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--cpu", action="store_true")

    # Loader
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_audio_sec", type=float, default=None)
    p.add_argument("--valid_ratio", type=float, default=0.05)

    # Loss
    p.add_argument("--mse_w", type=float, default=1.0)
    p.add_argument("--cos_w", type=float, default=1.0)

    # Teacher embedding dim (projection head)
    p.add_argument("--teacher_dim", type=int, default=1024)

    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)
