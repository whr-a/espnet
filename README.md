# BSCodec repo (built on ESPNet Codec)
## Demo
We've added a demo section! The codecs in the demo include BSCodec, DAC, Encodec, SemantiCodec, WavTokenizer, Xcodec, Mimo tokenizer. Please check `demo/` for comparisons.
## Installation

`pip install .`

## How to run

The model implementation is in `espnet2/gan_codec/bscodec`, and the recipe is in `egs2/bscodec/codec1`. If you want to run the model:

### Download the model

Install the `exp/` directory in https://huggingface.co/anonymous-release/BSCodec/tree/main and put it under the `egs2/bscodec/codec1`. 

### Organise your test set

Install the `dump/` directory in https://huggingface.co/anonymous-release/BSCodec/tree/main and also put it under the `egs2/bscodec/codec1`.

You can check the `wav.scp` and `utt2num_samples`. Add the "wavid wavpath" pairs to `wav.scp` and "wavid wav_samples_in_24kHz" to `utt2num_samples`. If your wav file is not in 24kHz, please resample it to 24kHz.

### Run script

Check `egs2/bscodec/codec1/run.sh`, modify "model" in the arguments to try other models. `Options: {"BSCodec_band_vq_5band", "BSCodec_band_simvq_3band", "BSCodec_band_simvq_2band}`

The `test_sets` argument should match the name of your test set.

