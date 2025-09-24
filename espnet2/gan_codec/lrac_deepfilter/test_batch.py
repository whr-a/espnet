#!/usr/bin/env python3
"""Test script for Lrac DeepFilterNet batch generator."""

import sys
import os
import yaml
import torch
import torchaudio
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from espnet2.gan_codec.lrac_deepfilter.lrac_deepfilter_batch_simple import LracGeneratorBatchDF


def load_config_from_yaml(yaml_path):
    """Load configuration from YAML file."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    codec_conf = config['codec_conf']

    # Prepare parameters
    encoder_params = codec_conf['encoder_params']
    decoder_params = codec_conf['decoder_params']

    # Prepare quantizer parameters - rename keys
    quantizer_params = {
        'codebook_dim': codec_conf['quantizer_params']['quantizer_codebook_dim'],
        'n_q': codec_conf['quantizer_params']['quantizer_n_q'],
        'bins': codec_conf['quantizer_params']['quantizer_bins'],
        'decay': codec_conf['quantizer_params']['quantizer_decay'],
        'kmeans_init': codec_conf['quantizer_params']['quantizer_kmeans_init'],
        'kmeans_iters': codec_conf['quantizer_params']['quantizer_kmeans_iters'],
        'threshold_ema_dead_code': codec_conf['quantizer_params']['quantizer_threshold_ema_dead_code'],
        'target_bandwidth': codec_conf['quantizer_params']['quantizer_target_bandwidth'],
    }

    return {
        'sample_rate': codec_conf['sampling_rate'],
        'encoder_params': encoder_params,
        'decoder_params': decoder_params,
        'quantizer_params': quantizer_params,
    }


def test_generator():
    """Test the batch processing generator."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load config
    config_path = "/work/nvme/bbjs/hwang41/lrac/espnet/egs2/lrac/track2/conf/train.yaml"
    logging.info(f"Loading config from: {config_path}")
    config = load_config_from_yaml(config_path)

    # Test audio
    test_file = "/work/nvme/bbjs/hwang41/lrac/data/lrac_data_generation/multilingual-speech-testing/LRAC-2025-test-data/open-test-set/track_2/noisy/T2_noise_speech_file000.wav"

    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test audio not found: {test_file}")

    logging.info(f"Loading audio from: {test_file}")

    # Load audio
    audio, sr = torchaudio.load(test_file)
    logging.info(f"Audio shape: {audio.shape}, sample rate: {sr}")

    # Resample if needed
    if sr != 24000:
        logging.info(f"Resampling from {sr} to 24000 Hz")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=24000)
        audio = resampler(audio)

    # Take first channel
    if audio.shape[0] > 1:
        audio = audio[0:1]

    # Trim to 3 seconds for testing
    max_samples = 24000 * 3
    if audio.shape[1] > max_samples:
        audio = audio[:, :max_samples]
        logging.info(f"Trimmed to {max_samples/24000:.1f} seconds")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize generator
    logging.info("Initializing generator with DeepFilterNet...")
    try:
        generator = LracGeneratorBatchDF(
            sample_rate=config['sample_rate'],
            encoder_params=config['encoder_params'],
            decoder_params=config['decoder_params'],
            quantizer_params=config['quantizer_params'],
            use_deepfilter=True
        ).to(device)
        generator.eval()
        logging.info("Generator initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize: {e}")
        raise

    # Test with different batch sizes
    for batch_size in [60]:
        logging.info(f"\n{'='*50}")
        logging.info(f"Testing batch size: {batch_size}")
        logging.info(f"{'='*50}")

        # Create batch
        audio_batch = audio.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        import time
        t1 = time.time()
        logging.info(f"Input shape: {audio_batch.shape}")

        try:
            # Test forward
            logging.info("Running forward pass...")
            with torch.no_grad():
                resyn, commit_loss, quant_loss, _ = generator(audio_batch, use_dual_decoder=False)

            logging.info(f"Success! Output shape: {resyn.shape}")
            logging.info(f"Commit loss: {commit_loss.item():.4f}")
            logging.info(f"Quantization loss: {quant_loss.item():.4f}")

            # Check validity
            assert resyn.shape == audio_batch.shape, f"Shape mismatch"
            assert not torch.isnan(resyn).any(), "NaN in output"

            # Test encode/decode
            logging.info("Testing encode/decode...")
            with torch.no_grad():
                codes = generator.encode(audio_batch)
                decoded = generator.decode(codes)

            logging.info(f"Codes shape: {codes.shape}")
            logging.info(f"Decoded shape: {decoded.shape}")

            # Test enhancement only
            if generator.use_deepfilter:
                logging.info("Testing DeepFilterNet only...")
                with torch.no_grad():
                    enhanced = generator.enhance_24k_batch(audio_batch)

                diff = torch.abs(enhanced - audio_batch).mean()
                logging.info(f"Mean difference: {diff.item():.6f}")

            # Cleanup
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            t2 = time.time()
            print("total_time", t2 - t1)
        except Exception as e:
            logging.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            raise

    logging.info("\n" + "="*50)
    logging.info("All tests passed!")

    # Save sample outputs
    output_dir = "/work/nvme/bbjs/hwang41/lrac/espnet/espnet2/gan_codec/lrac_deepfilter/test_outputs"
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        single = audio.unsqueeze(0).to(device)

        # Enhanced audio
        if generator.use_deepfilter:
            enhanced = generator.enhance_24k_batch(single)
            torchaudio.save(
                os.path.join(output_dir, "enhanced.wav"),  # Convert Path to string
                enhanced.squeeze(0).cpu(),
                24000
            )

        # Reconstructed
        resyn, _, _, _ = generator(single)
        torchaudio.save(
            os.path.join(output_dir, "reconstructed.wav"),  # Convert Path to string
            resyn.squeeze(0).cpu(),
            24000
        )

    logging.info(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    test_generator()