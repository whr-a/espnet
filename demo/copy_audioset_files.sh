#!/bin/bash

# Target audioset files
AUDIOSET_FILES=(
    "audioset_-0Gj8-vB1q4.wav"
    "audioset_-0RWZT-miFs.wav"
    "audioset_-0vPFx-wRRI.wav"
    "audioset_-65CfQUX9Ng.wav"
    "audioset_--U7joUcTCo.wav"
)

# Target musdb18 files
MUSDB18_FILES=(
    "musdb18_Al_James_-_Schoolboy_Facination.stem.wav"
    "musdb18_Arise_-_Run_Run_Run.stem.wav"
    "musdb18_Georgia_Wonder_-_Siren.stem.wav"
)

# Base demo directory
DEMO_DIR="/u/hwang41/hwang41/3ai/espnet/egs_band/bandcodec/codec1/exp_unicodec/demo"

# Define source directories for audioset (test_audio_big)
declare -A AUDIOSET_DIRS=(
    ["DAC_6kbps"]="/u/hwang41/hwang41/3ai/espnet/egs_band/bandcodec/codec1/exp_unicodec/codec_3domain_dac_raw_fs24000/decode_639epoch/test_audio_big/wav"
    ["DAC_4.5kbps"]="/u/hwang41/hwang41/3ai/espnet/egs_band/bandcodec/codec1/exp_unicodec/codec_3domain_dac_raw_fs24000/decode4_5kbps_639epoch/test_audio_big/wav"
    ["DAC_3kbps"]="/u/hwang41/hwang41/3ai/espnet/egs_band/bandcodec/codec1/exp_unicodec/codec_3domain_dac_raw_fs24000/decode3_639epoch/test_audio_big/wav"
    ["encodec_3kbps"]="/u/hwang41/hwang41/3ai/espnet/egs_band/bandcodec/codec1/exp_unicodec/codec_fake_encodec_3_0_raw_fs24000/decode_1epoch/test_audio_big/wav"
    ["encodec_6kbps"]="/u/hwang41/hwang41/3ai/espnet/egs_band/bandcodec/codec1/exp_unicodec/codec_fake_encodec_6_0_raw_fs24000/decode_1epoch/test_audio_big/wav"
    ["MimoTokenizer"]="/u/hwang41/hwang41/3ai/espnet/egs_band/bandcodec/codec1/exp_unicodec/codec_fake_mimo_raw_fs24000/decode_1epoch/test_audio_big/wav"
    ["SemantiCodec"]="/u/hwang41/hwang41/3ai/espnet/egs_band/bandcodec/codec1/exp_unicodec/codec_fake_semanticodec_raw_fs24000/decode_1epoch/test_audio_small/wav"
    ["WavTokenizer"]="/u/hwang41/hwang41/3ai/espnet/egs_band/bandcodec/codec1/exp_unicodec/codec_fake_wavtokenizer_raw_fs24000/decode_1epoch/test_audio_big/wav"
    ["Xcodec"]="/u/hwang41/hwang41/3ai/espnet/egs_band/bandcodec/codec1/exp_unicodec/codec_fake_xcodec_raw_fs24000/decode_1epoch/test_audio_big/wav"
    ["BSCodec_2band"]="/work/nvme/bbjs/shi3/codec_haoran/espnet/egs_band/bandcodec/codec1/exp_unicodec/codec_v6_4_2_raw_fs24000/decode_187epoch/test_audio_big/wav"
)

# Define source directories for musdb18 (test_music)
declare -A MUSDB18_DIRS=(
    ["DAC_6kbps"]="/u/hwang41/hwang41/3ai/espnet/egs_band/bandcodec/codec1/exp_unicodec/codec_3domain_dac_raw_fs24000/decode_639epoch/test_music/wav"
    ["DAC_4.5kbps"]="/u/hwang41/hwang41/3ai/espnet/egs_band/bandcodec/codec1/exp_unicodec/codec_3domain_dac_raw_fs24000/decode4_5kbps_639epoch/test_music/wav"
    ["DAC_3kbps"]="/u/hwang41/hwang41/3ai/espnet/egs_band/bandcodec/codec1/exp_unicodec/codec_3domain_dac_raw_fs24000/decode3_639epoch/test_music/wav"
    ["encodec_3kbps"]="/u/hwang41/hwang41/3ai/espnet/egs_band/bandcodec/codec1/exp_unicodec/codec_fake_encodec_3_0_raw_fs24000/decode_1epoch/test_music/wav"
    ["encodec_6kbps"]="/u/hwang41/hwang41/3ai/espnet/egs_band/bandcodec/codec1/exp_unicodec/codec_fake_encodec_6_0_raw_fs24000/decode_1epoch/test_music/wav"
    ["MimoTokenizer"]="/u/hwang41/hwang41/3ai/espnet/egs_band/bandcodec/codec1/exp_unicodec/codec_fake_mimo_raw_fs24000/decode_1epoch/test_music/wav"
    ["SemantiCodec"]="/u/hwang41/hwang41/3ai/espnet/egs_band/bandcodec/codec1/exp_unicodec/codec_fake_semanticodec_raw_fs24000/decode_1epoch/test_music/wav"
    ["WavTokenizer"]="/u/hwang41/hwang41/3ai/espnet/egs_band/bandcodec/codec1/exp_unicodec/codec_fake_wavtokenizer_raw_fs24000/decode_1epoch/test_music/wav"
    ["Xcodec"]="/u/hwang41/hwang41/3ai/espnet/egs_band/bandcodec/codec1/exp_unicodec/codec_fake_xcodec_raw_fs24000/decode_1epoch/test_music/wav"
    ["BSCodec_2band"]="/work/nvme/bbjs/shi3/codec_haoran/espnet/egs_band/bandcodec/codec1/exp_unicodec/codec_v6_4_2_raw_fs24000/decode_187epoch/test_music/wav"
)

# Copy audioset files for each codec
echo "=== Copying Audioset files ==="
for codec_name in "${!AUDIOSET_DIRS[@]}"; do
    source_dir="${AUDIOSET_DIRS[$codec_name]}"
    target_dir="${DEMO_DIR}/${codec_name}"

    echo "Processing ${codec_name}..."

    if [ ! -d "$source_dir" ]; then
        echo "  WARNING: Source directory does not exist: $source_dir"
        continue
    fi

    if [ ! -d "$target_dir" ]; then
        echo "  WARNING: Target directory does not exist: $target_dir"
        continue
    fi

    # Copy each audioset file
    for file in "${AUDIOSET_FILES[@]}"; do
        if [ -f "${source_dir}/${file}" ]; then
            cp "${source_dir}/${file}" "${target_dir}/"
            echo "  ✓ Copied ${file}"
        else
            echo "  ✗ File not found: ${file}"
        fi
    done

    echo ""
done

# Copy musdb18 files for each codec
echo "=== Copying MUSDB18 files ==="
for codec_name in "${!MUSDB18_DIRS[@]}"; do
    source_dir="${MUSDB18_DIRS[$codec_name]}"
    target_dir="${DEMO_DIR}/${codec_name}"

    echo "Processing ${codec_name}..."

    if [ ! -d "$source_dir" ]; then
        echo "  WARNING: Source directory does not exist: $source_dir"
        continue
    fi

    if [ ! -d "$target_dir" ]; then
        echo "  WARNING: Target directory does not exist: $target_dir"
        continue
    fi

    # Copy each musdb18 file
    for file in "${MUSDB18_FILES[@]}"; do
        if [ -f "${source_dir}/${file}" ]; then
            cp "${source_dir}/${file}" "${target_dir}/"
            echo "  ✓ Copied ${file}"
        else
            echo "  ✗ File not found: ${file}"
        fi
    done

    echo ""
done

echo "All files copied successfully!"
