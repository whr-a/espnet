# Low Resouce Audio Codec Challenge (LRAC) Baseline Model Recipe

This repository contains the baseline ESPNet recipe for the [Low-Resource Audio Codec Challenge (LRAC)](https://lrac.short.gy/).

The challenge targets low-complexity low-bitrate speech codecs that can operate under everyday noise and reverberation, while meeting strict constraints on computational cost, latency, and bitrate.

The challenge has two tracks: 
- Track 1 (Transparency Codec): Focuses on minimizing percetual distortion while remaining robust to noise and reverberation. 
- Track 2 (Enhancement Codec): Extends Track 1 by incorporating denoising and dereverberation. 

More details: [Challenge Tracks](https://lrac.short.gy/tracks).

## Getting Started

Each track has its own subfolder (track1/, track2/). Before running the baseline recipes:

1. Download and preprocess the data using the [LRAC data generation toolkit](https://github.com/cisco-open/lrac_data_generation).
2. Copy the generated data/ folder into both track1/ and track2/.

For a general introduction to ESPNet recipes, see:

- [ESPNet2 Tutorial](https://espnet.github.io/espnet/espnet2_tutorial.html)
- [Speech Codec Recipe](https://espnet.github.io/espnet/recipe/codec1.html)
- [TEMPLATE recipe](https://espnet.github.io/espnet/recipe/)

## Recipe Staged

Each recipe consists of 7 stages:
1. **Data preparation:** Since data preparation is handled by the LRAC data generation repo, this stage only check if the required data folders exist.
2. **Formatting:** Formats wav.scp and reference.scp, and moving them to dump/raw folder. 
3. **Data Filtering:** Removes very long or very short audio files based on the thresholds defined in codec.sh.
4. **Statistics Collection:** Skipped for this baseline, since the codec operates in the raw audio domain. A message is printed instead.
5. **Model Training:** Trains the codec and runs validation after every epoch keeping track of best checkpoints in terms of criterion defined in the config file.
6. **Inference:** Runs inference on the test datasets and saves the outputs.
7. **Scoring:** Evaluates the codec outputs using objective metrics.

## Important Files

- `run.sh`: Main entry point for the recipe. Defines dataset folders and config files, then calls `codec.sh`. Options can be passed via `run.sh` without editing `codec.sh`. For example:

```bash
bash run.sh --stage 1 --stop_stage 1
```
This will run only stage 1.

- `codec.sh`: Main script executing all the 7 stages. We recommend running stages one-by-one for ease of debugging.
- `conf/train.yaml`: Training configuration (preprocessing, model, optimizer and data loader hyperparameters).
- `conf/score.yaml`: Scoring configuration listing objective metrics to compute.
- `local/path.sh`: Adds the ESPNet path to `PYTHONPATH` environment variable.

## Baseline Model

See the [baseline description](https://lrac.short.gy/baseline) on the challenge website for full details. 

The baseline model is a convolutional encoder-decoder neural network with a Residual Vector Quantizer (RVQ) module for quantization. Key files in implementing the baseline model include:

- [`soundstream.py`](https://github.com/cisco-open/espnet/blob/master/espnet2/gan_codec/soundstream/soundstream.py) 
Base SoundStream codec. Modified to support projections in RVQ layers, semantic loss, and Track 2 enhancement losses.
- [`encodec.py`](https://github.com/cisco-open/espnet/blob/master/espnet2/gan_codec/encodec/encodec.py) 
Implements Encodec, from which LRACConvBaseline is derived.
- [`lrac.py`](https://github.com/cisco-open/espnet/blob/master/espnet2/gan_codec/lrac/lrac.py) 
Defines the main LRACConvBaseline class and its generator LRACGenerator.
- [`generic_seanet.py` (encoder)](https://github.com/cisco-open/espnet/blob/master/espnet2/gan_codec/shared/encoder/generic_seanet.py) 
Encoder with more flexible hyperparameter control (kernel size, dilation, causality, etc.) than the Soundstream encoder.
- [`generic_seanet.py` (decoder)](https://github.com/cisco-open/espnet/blob/master/espnet2/gan_codec/shared/decoder/generic_seanet.py) 
Decoder counterpart with extended hyperparameter control.
- [`semantic_loss.py`](https://github.com/cisco-open/espnet/blob/master/espnet2/gan_codec/shared/loss/semantic_loss.py) 
Implements semantic loss as in [1], optionally useful for participants.
- [`core_vq.py`](https://github.com/cisco-open/espnet/blob/master/espnet2/gan_codec/shared/quantizer/modules/core_vq.py) 
Implements Vector Quantization modules. Note that we have made some modifications and additions, mostly borrowing from the original [vector-quantize-pytorch repo](https://github.com/lucidrains/vector-quantize-pytorch) to fix bugs in distributed training and gradient handling. 
- [`gan_codec.py`](https://github.com/cisco-open/espnet/blob/master/espnet2/tasks/gan_codec.py) 
Defines the GANCodecTask, modified to allow on-the-fly noise and reverb augmentation for Track 2.
- [`preprocessor.py`](https://github.com/cisco-open/espnet/blob/master/espnet2/train/preprocessor.py)
Defines CommonPreprocessor used in Track 1 recipe and EnhPreprocessor used in Track 2 recipe.
- [`gan_codec_train.py`](https://github.com/cisco-open/espnet/blob/master/espnet2/bin/gan_codec_train.py) 
Main entry point for training.
- [`gan_codec_inference.py`](https://github.com/cisco-open/espnet/blob/master/espnet2/bin/gan_codec_inference.py) 
Main entry point for inference.

## References
1. Parker, J. D., Smirnov, A., Pons, J., Carr, C. J., Zukowski, Z., Evans, Z., & Liu, X. (2024). Scaling transformers for low-bitrate high-quality speech coding. arXiv preprint arXiv:2411.19842.
