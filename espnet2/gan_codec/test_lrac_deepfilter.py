#!/usr/bin/env python3
"""Test script for Lrac DeepFilterNet enhanced generator."""

import sys
import os
import yaml
import torch
import torchaudio
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from espnet2.gan_codec.lrac_deepfilter.lrac_deepfilter_enhanced import LracEnhancedGenerator


def load_config_from_yaml(yaml_path):
    """Load configuration from YAML file."""
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract codec configuration
    codec_conf = config['codec_conf']

    # Prepare encoder parameters
    encoder_params = codec_conf['encoder_params']

    # Prepare decoder parameters
    decoder_params = codec_conf['decoder_params']

    # Prepare quantizer parameters - need to rename keys
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
        'sampling_rate': codec_conf['sampling_rate'],
        'encoder_params': encoder_params,
        'decoder_params': decoder_params,
        'quantizer_params': quantizer_params,
    }


def test_generator_with_batch():
    """Test the enhanced generator with batch processing."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load configuration from YAML
    config_path = "/work/nvme/bbjs/hwang41/lrac/espnet/egs2/lrac/track2/conf/train.yaml"
    logging.info(f"Loading configuration from: {config_path}")
    config = load_config_from_yaml(config_path)

    # Test audio file
    test_file = "/work/nvme/bbjs/hwang41/lrac/data/lrac_data_generation/multilingual-speech-testing/LRAC-2025-test-data/open-test-set/track_2/noisy/T2_noise_speech_file000.wav"

    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test audio file not found: {test_file}")

    logging.info(f"Loading test audio from: {test_file}")

    # Load audio
    audio, sr = torchaudio.load(test_file)
    logging.info(f"Loaded audio shape: {audio.shape}, sample rate: {sr}")

    # Resample to 24kHz if needed
    if sr != 24000:
        logging.info(f"Resampling from {sr} Hz to 24000 Hz")
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=24000)
        audio = resampler(audio)
        sr = 24000

    # Take first channel if stereo
    if audio.shape[0] > 1:
        audio = audio[0:1]  # Keep shape [1, T]

    # Trim audio to a manageable length for testing
    max_samples = 24000 * 5  # 5 seconds max
    if audio.shape[1] > max_samples:
        audio = audio[:, :max_samples]
        logging.info(f"Trimmed audio to {max_samples/24000:.1f} seconds for testing")

    # Prepare batches of different sizes
    batch_sizes = [2, 4]

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Initialize generator with config from YAML
    logging.info("Initializing LracEnhancedGenerator with DeepFilterNet...")
    try:
        generator = LracEnhancedGenerator(
            sample_rate=config['sampling_rate'],
            encoder_params=config['encoder_params'],
            decoder_params=config['decoder_params'],
            quantizer_params=config['quantizer_params'],
            use_deepfilter=True
        ).to(device)
        generator.eval()  # Set to evaluation mode
        logging.info("Generator initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize generator: {e}")
        raise

    # Test with different batch sizes
    for batch_size in batch_sizes:
        logging.info(f"\n{'='*50}")
        logging.info(f"Testing with batch size: {batch_size}")
        logging.info(f"{'='*50}")

        # Create batch by duplicating the audio
        audio_batch = audio.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, C, T]
        audio_batch = audio_batch.to(device)

        logging.info(f"Input batch shape: {audio_batch.shape}")

        try:
            # Test forward pass
            logging.info("Running forward pass...")
            with torch.no_grad():
                resyn_audio, commit_loss, quantization_loss, resyn_audio_real = generator(
                    audio_batch,
                    use_dual_decoder=False
                )

            logging.info(f"Forward pass successful!")
            logging.info(f"Output shape: {resyn_audio.shape}")
            logging.info(f"Commitment loss: {commit_loss.item():.4f}")
            logging.info(f"Quantization loss: {quantization_loss.item():.4f}")

            # Check output validity
            assert resyn_audio.shape == audio_batch.shape, f"Shape mismatch: {resyn_audio.shape} != {audio_batch.shape}"
            assert not torch.isnan(resyn_audio).any(), "NaN values in output"
            assert not torch.isinf(resyn_audio).any(), "Inf values in output"

            # Test encoding and decoding
            logging.info("\nTesting encode/decode...")
            with torch.no_grad():
                codes = generator.encode(audio_batch)
                decoded = generator.decode(codes)

            logging.info(f"Encode/decode successful!")
            logging.info(f"Codes shape: {codes.shape}")
            logging.info(f"Decoded shape: {decoded.shape}")

            # Check codec output
            assert decoded.shape == audio_batch.shape, f"Decoded shape mismatch: {decoded.shape} != {audio_batch.shape}"
            assert not torch.isnan(decoded).any(), "NaN values in decoded output"

            # Test enhancement only (if using DeepFilterNet)
            if generator.use_deepfilter:
                logging.info("\nTesting DeepFilterNet enhancement only...")
                with torch.no_grad():
                    enhanced = generator.enhance_batch_24k(audio_batch)

                logging.info(f"Enhancement successful!")
                logging.info(f"Enhanced shape: {enhanced.shape}")

                # Check if enhancement actually changed the audio
                diff = torch.abs(enhanced - audio_batch).mean()
                snr_improvement = 20 * torch.log10(
                    torch.norm(audio_batch) / torch.norm(audio_batch - enhanced)
                )
                logging.info(f"Mean absolute difference from input: {diff.item():.6f}")
                logging.info(f"Estimated SNR improvement: {snr_improvement.item():.2f} dB")

            # Memory cleanup
            del resyn_audio, commit_loss, quantization_loss
            if resyn_audio_real is not None:
                del resyn_audio_real
            del codes, decoded
            if generator.use_deepfilter:
                del enhanced
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            logging.error(f"Error during testing with batch size {batch_size}: {e}")
            import traceback
            traceback.print_exc()
            raise

    logging.info(f"\n{'='*50}")
    logging.info("All tests passed successfully!")
    logging.info(f"{'='*50}")

    # Save a sample output for inspection
    output_dir = Path(__file__).parent / "test_outputs"
    output_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        # Process single audio for saving
        single_audio = audio.unsqueeze(0).to(device)  # [1, 1, T]

        # Get enhanced audio if DeepFilterNet is enabled
        if generator.use_deepfilter:
            enhanced_single = generator.enhance_batch_24k(single_audio)
        else:
            enhanced_single = single_audio

        # Get reconstructed audio
        resyn_single, _, _, _ = generator(single_audio)

        # Save outputs
        input_path = output_dir / "input_audio.wav"
        enhanced_path = output_dir / "enhanced_output.wav"
        resyn_path = output_dir / "reconstructed_output.wav"

        # Save input for comparison
        torchaudio.save(
            input_path,
            audio.cpu(),
            24000
        )

        # Save enhanced
        if generator.use_deepfilter:
            torchaudio.save(
                enhanced_path,
                enhanced_single.squeeze(0).cpu(),
                24000
            )
            logging.info(f"Saved enhanced audio to: {enhanced_path}")

        # Save reconstructed
        torchaudio.save(
            resyn_path,
            resyn_single.squeeze(0).cpu(),
            24000
        )

        logging.info(f"Saved input audio to: {input_path}")
        logging.info(f"Saved reconstructed audio to: {resyn_path}")


def test_multi_gpu():
    """Test multi-GPU compatibility."""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if not torch.cuda.is_available():
        logging.warning("CUDA not available, skipping multi-GPU test")
        return

    if torch.cuda.device_count() < 2:
        logging.warning(f"Only {torch.cuda.device_count()} GPU(s) available, skipping multi-GPU test")
        return

    logging.info(f"\n{'='*50}")
    logging.info(f"Testing multi-GPU with {torch.cuda.device_count()} GPUs")
    logging.info(f"{'='*50}")

    # Load configuration
    config_path = "/work/nvme/bbjs/hwang41/lrac/espnet/egs2/lrac/track2/conf/train.yaml"
    config = load_config_from_yaml(config_path)

    # Initialize generator
    generator = LracEnhancedGenerator(
        sample_rate=config['sampling_rate'],
        encoder_params=config['encoder_params'],
        decoder_params=config['decoder_params'],
        quantizer_params=config['quantizer_params'],
        use_deepfilter=True
    )

    # Wrap with DataParallel
    generator = torch.nn.DataParallel(generator)
    generator = generator.cuda()
    generator.eval()

    # Create dummy batch
    batch_size = 8  # Larger batch for multi-GPU
    audio_batch = torch.randn(batch_size, 1, 24000 * 2).cuda()  # 2 seconds of audio

    try:
        with torch.no_grad():
            # Access the module for DataParallel
            resyn_audio, commit_loss, quantization_loss, _ = generator(
                audio_batch,
                use_dual_decoder=False
            )

        logging.info("Multi-GPU forward pass successful!")
        logging.info(f"Output shape: {resyn_audio.shape}")
        logging.info(f"Devices used: {generator.device_ids}")

    except Exception as e:
        logging.error(f"Multi-GPU test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Run main test
    test_generator_with_batch()

    # Run multi-GPU test if available
    test_multi_gpu()