# LibriSpeech compatibility entry for OWSM RL

The implementation and canonical profiles are in the
[OWSM recipe](../../owsm_v4/s2t1/README.md#optional-miles-rl-stage).
This directory retains only its forwarding wrapper and configuration symlinks.
Use the shared [RL guide](../../../espnet2/rl/README.md) for preparation,
one-/two-H100 training and evaluation.

Prepare real LibriSpeech train/dev splits using the ordinary ASR recipe. The
compatibility wrapper accepts its existing model and data arguments:

```bash
./run.sh --wav-scp ../asr1/data/train_clean_100/wav.scp \
  --text ../asr1/data/train_clean_100/text --train-config /path/to/native.yaml \
  --checkpoint /path/to/native.pth --work-dir exp/owsm_rl
```

The canonical training module is `python -m espnet2.bin.rl_train`. One-/two-GPU
profiles keep32 prompts × 8 candidates and five default updates; WER improvement
must be assessed independently on a disjoint held-out manifest.
