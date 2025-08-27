# ESPnet
### forked from https://github.com/espnet/espnet

# Usage
First follow the install guidance in ESPnet.

Then run:
```
pip install nemo_toolkit['asr']
```

Then adjust the `$PYTHONPATH`:
```
export PYTHONPATH=/work/nvme/bbjs/hwang41/lrac/espnet/espnet2/gan_codec/shared/encoder/semantic_encoder:/work/nvme/bbjs/hwang41/lrac/espnet:$PYTHONPATH
```

Above config is my `$PYTHONPATH` written in `.bashrc`. you need to adjust the path to yours.

This is for the purpose of using the Nemo library in an editable manner.

# Some paths

## models
`/work/nvme/bbjs/hwang41/lrac/canary-1b-flash/split/encoder_standalone.pt`
`/work/nvme/bbjs/hwang41/lrac/canary-1b-flash/split/preprocessor_standalone.pt`
You can check `/work/nvme/bbjs/hwang41/lrac/espnet/espnet2/gan_codec/shared/encoder/semantic_encoder/encoder_nemo.py` to get to know how to use them.

## dataset
`/work/nvme/bbjs/hwang41/lrac/espnet/egs_lrac/lrac/codec1/dump/raw`

## code
`/work/nvme/bbjs/hwang41/lrac/espnet/espnet2/gan_codec/shared/encoder/semantic_encoder/encoder_nemo.py` this is the code of nemo ASR encoder.

