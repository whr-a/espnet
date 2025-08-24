#!/usr/bin/env python3
"""
Test script for AutoGroupResidualVectorQuantizer.

This script provides comprehensive tests for the AutoGroupResidualVectorQuantizer
class including forward/backward passes, encode/decode cycles, bandwidth control,
and edge cases.
"""

import torch
import torch.nn.functional as F
from residual_vq import AutoGroupResidualVectorQuantizer


def test_basic_functionality():
    """Test basic forward pass and shapes."""
    print("=== Testing Basic Functionality ===")
    
    batch_size = 2
    dimension = 128
    codebook_dim = 8
    n_q = 4
    bins = 256
    seq_len = 100
    
    quantizer = AutoGroupResidualVectorQuantizer(
        dimension=dimension,
        codebook_dim=codebook_dim,
        n_q=n_q,
        bins=bins,
        quantizer_dropout=0.0,
        frame_residual_vq=True
    )
    
    x = torch.randn(batch_size, dimension, seq_len)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    result = quantizer.forward(x, sample_rate=16000, bandwidth=6.0)
    quantized, codes, bandwidth, commit_loss, quant_loss = result
    
    print(f"✓ Quantized shape: {quantized.shape}")
    print(f"✓ Codes shape: {codes.shape}")
    print(f"✓ Bandwidth: {bandwidth.item():.2f} kb/s")
    print(f"✓ Commit loss: {commit_loss.item():.4f}")
    print(f"✓ Quantization loss: {quant_loss.item():.4f}")
    
    # Check output shapes
    assert quantized.shape == x.shape, f"Expected {x.shape}, got {quantized.shape}"
    assert len(codes.shape) == 3, f"Expected 3D codes tensor, got {codes.shape}"
    assert isinstance(commit_loss.item(), float), "Commit loss should be a scalar"
    assert isinstance(quant_loss.item(), float), "Quantization loss should be a scalar"
    
    print("✓ Basic functionality test passed!\n")


def test_encode_decode_cycle():
    """Test encode-decode consistency."""
    print("=== Testing Encode-Decode Cycle ===")
    
    quantizer = AutoGroupResidualVectorQuantizer(
        dimension=64,
        codebook_dim=8,
        n_q=3,
        bins=128,
        quantizer_dropout=0.0,
        frame_residual_vq=False
    )
    
    x = torch.randn(1, 64, 50)
    sample_rate = 16000
    bandwidth = 4.0
    
    # Method 1: Direct forward
    quantized_direct, codes_direct, _, _, _ = quantizer.forward(x, sample_rate, bandwidth)
    
    # Method 2: Encode then decode
    codes_enc = quantizer.encode(x, sample_rate, bandwidth)
    quantized_dec = quantizer.decode(codes_enc)
    
    print(f"Direct quantized shape: {quantized_direct.shape}")
    print(f"Encoded codes shape: {codes_enc.shape}")
    print(f"Decoded shape: {quantized_dec.shape}")
    
    # Check consistency
    mse_error = F.mse_loss(quantized_direct, quantized_dec)
    print(f"Encode-decode consistency MSE: {mse_error.item():.8f}")
    
    if mse_error < 1e-6:
        print("✓ Encode-decode consistency test passed!")
    else:
        print("✗ Encode-decode consistency test failed!")
        
    # Test reconstruction quality
    reconstruction_mse = F.mse_loss(x, quantized_direct)
    print(f"Reconstruction MSE: {reconstruction_mse.item():.6f}")
    print("✓ Encode-decode cycle test completed!\n")


def test_bandwidth_control():
    """Test bandwidth control mechanism."""
    print("=== Testing Bandwidth Control ===")
    
    quantizer = AutoGroupResidualVectorQuantizer(
        dimension=128,
        codebook_dim=8,
        n_q=6,
        bins=256,
        quantizer_dropout=0.0
    )
    
    x = torch.randn(1, 128, 100)
    sample_rate = 16000
    
    bandwidths = [1.0, 3.0, 6.0, 12.0, None]
    
    for bw in bandwidths:
        if bw is None:
            n_q_expected = quantizer.n_q
            print(f"Testing no bandwidth limit (using all {n_q_expected} quantizers):")
        else:
            n_q_expected = quantizer.get_num_quantizers_for_bandwidth(sample_rate, bw)
            print(f"Testing bandwidth {bw:.1f} kb/s (expected {n_q_expected} quantizers):")
        
        result = quantizer.forward(x, sample_rate, bandwidth=bw)
        actual_bw = result[2].item()
        
        if bw is not None:
            expected_bw = n_q_expected * quantizer.get_bandwidth_per_quantizer(sample_rate)
            print(f"  Expected bandwidth: {expected_bw:.2f} kb/s")
        print(f"  Actual bandwidth: {actual_bw:.2f} kb/s")
        
        # Test encoding with same bandwidth
        codes = quantizer.encode(x, sample_rate, bandwidth=bw)
        print(f"  Encoded codes shape: {codes.shape}")
        print()
    
    print("✓ Bandwidth control test completed!\n")


def test_quantizer_dropout():
    """Test quantizer dropout behavior."""
    print("=== Testing Quantizer Dropout ===")
    
    quantizer = AutoGroupResidualVectorQuantizer(
        dimension=64,
        codebook_dim=8,
        n_q=4,
        bins=128,
        quantizer_dropout=0.5,
        frame_residual_vq=True
    )
    
    x = torch.randn(2, 64, 50)
    sample_rate = 16000
    
    # Test in training mode
    quantizer.train()
    result_train = quantizer.forward(x, sample_rate)
    
    # Test in evaluation mode
    quantizer.eval()
    result_eval = quantizer.forward(x, sample_rate)
    
    print(f"Training mode - commit loss: {result_train[3].item():.4f}")
    print(f"Evaluation mode - commit loss: {result_eval[3].item():.4f}")
    
    # Test multiple runs in training mode for randomness
    quantizer.train()
    losses = []
    for i in range(5):
        result = quantizer.forward(x, sample_rate)
        losses.append(result[3].item())
    
    print(f"Training losses across 5 runs: {[f'{l:.4f}' for l in losses]}")
    loss_variance = torch.tensor(losses).var().item()
    print(f"Loss variance: {loss_variance:.6f}")
    
    print("✓ Quantizer dropout test completed!\n")


def test_edge_cases():
    """Test edge cases and robustness."""
    print("=== Testing Edge Cases ===")
    
    quantizer = AutoGroupResidualVectorQuantizer(
        dimension=32,
        codebook_dim=4,
        n_q=3,
        bins=64
    )
    
    sample_rate = 16000
    
    # Test 1: Very small input
    print("Testing very small input...")
    x_small = torch.randn(1, 32, 5)
    result_small = quantizer.forward(x_small, sample_rate, bandwidth=2.0)
    print(f"✓ Small input processed: {result_small[0].shape}")
    
    # Test 2: Very large input
    print("Testing very large input...")
    x_large = torch.randn(1, 32, 2000)
    result_large = quantizer.forward(x_large, sample_rate, bandwidth=8.0)
    print(f"✓ Large input processed: {result_large[0].shape}")
    
    # Test 3: Single time step
    print("Testing single time step...")
    x_single = torch.randn(1, 32, 1)
    result_single = quantizer.forward(x_single, sample_rate, bandwidth=1.0)
    print(f"✓ Single time step processed: {result_single[0].shape}")
    
    # Test 4: Large batch
    print("Testing large batch...")
    x_batch = torch.randn(8, 32, 100)
    result_batch = quantizer.forward(x_batch, sample_rate, bandwidth=4.0)
    print(f"✓ Large batch processed: {result_batch[0].shape}")
    
    # Test 5: Zero bandwidth (should use 1 quantizer)
    print("Testing zero bandwidth...")
    result_zero = quantizer.forward(x_small, sample_rate, bandwidth=0.0)
    print(f"✓ Zero bandwidth handled: {result_zero[2].item():.2f} kb/s")
    
    print("✓ Edge cases test completed!\n")


def test_gradient_flow():
    """Test gradient computation and backpropagation."""
    print("=== Testing Gradient Flow ===")
    
    quantizer = AutoGroupResidualVectorQuantizer(
        dimension=64,
        codebook_dim=8,
        n_q=3,
        bins=128,
        quantizer_dropout=0.1
    )
    
    x = torch.randn(1, 64, 50, requires_grad=True)
    sample_rate = 16000
    
    # Forward pass
    result = quantizer.forward(x, sample_rate, bandwidth=3.0)
    
    # Compute total loss
    total_loss = result[3] + result[4]  # commit_loss + quant_loss
    print(f"Total loss: {total_loss.item():.4f}")
    
    # Backward pass
    total_loss.backward()
    
    # Check gradients
    if x.grad is not None:
        grad_norm = x.grad.norm().item()
        grad_mean = x.grad.mean().item()
        print(f"✓ Gradients computed successfully")
        print(f"  Gradient norm: {grad_norm:.6f}")
        print(f"  Gradient mean: {grad_mean:.6f}")
        
        # Check for NaN or inf gradients
        has_nan = torch.isnan(x.grad).any().item()
        has_inf = torch.isinf(x.grad).any().item()
        
        if has_nan:
            print("✗ Warning: NaN gradients detected")
        if has_inf:
            print("✗ Warning: Infinite gradients detected")
        
        if not has_nan and not has_inf:
            print("✓ Gradients are healthy (no NaN/inf)")
    else:
        print("✗ No gradients computed")
    
    print("✓ Gradient flow test completed!\n")


def test_different_configurations():
    """Test different model configurations."""
    print("=== Testing Different Configurations ===")
    
    configs = [
        {"dimension": 128, "codebook_dim": 8, "n_q": 4, "bins": 256, "frame_residual_vq": True},
        {"dimension": 256, "codebook_dim": 16, "n_q": 8, "bins": 512, "frame_residual_vq": False},
        {"dimension": 64, "codebook_dim": 4, "n_q": 2, "bins": 128, "frame_residual_vq": True},
        {"dimension": 512, "codebook_dim": 32, "n_q": 6, "bins": 1024, "frame_residual_vq": False},
    ]
    
    for i, config in enumerate(configs):
        print(f"Testing configuration {i+1}: {config}")
        
        quantizer = AutoGroupResidualVectorQuantizer(**config)
        x = torch.randn(1, config["dimension"], 100)
        
        try:
            result = quantizer.forward(x, sample_rate=16000, bandwidth=6.0)
            print(f"✓ Configuration {i+1} successful: {result[0].shape}")
            
            # Test encode/decode
            codes = quantizer.encode(x, sample_rate=16000, bandwidth=6.0)
            decoded = quantizer.decode(codes)
            print(f"✓ Encode/decode successful: {decoded.shape}")
            
        except Exception as e:
            print(f"✗ Configuration {i+1} failed: {e}")
        
        print()
    
    print("✓ Different configurations test completed!\n")


def test_overfitting_ability():
    """Test the overfitting ability of AutoGroupResidualVectorQuantizer.
    
    This test verifies that the quantizer can memorize and perfectly reconstruct
    a small dataset, demonstrating its capacity to learn representations.
    """
    print("=== Testing Overfitting Ability ===")
    
    # Create a small dataset to overfit
    torch.manual_seed(42)  # For reproducible results
    batch_size = 2
    dimension = 32
    seq_len = 20
    num_epochs = 30000
    
    # Generate synthetic training data (normalized)
    train_data = torch.randn(batch_size, dimension, seq_len) * 0.5
    print(f"Training data shape: {train_data.shape}")
    print(f"Training data range: [{train_data.min().item():.3f}, {train_data.max().item():.3f}]")
    
    # Initialize quantizer with sufficient capacity
    quantizer = AutoGroupResidualVectorQuantizer(
        dimension=dimension,
        codebook_dim=8,   # Moderate codebook dimension
        n_q=4,            # Fewer quantizers to start
        bins=256,         # Moderate codebook size
        quantizer_dropout=0.0,  # No dropout for overfitting
        frame_residual_vq=False
    )
    
    # Set up optimizer with lower learning rate and gradient clipping
    optimizer = torch.optim.Adam(quantizer.parameters(), lr=1e-4, weight_decay=1e-6)
    
    sample_rate = 16000
    target_bandwidth = 8.0  # Moderate bandwidth
    
    print(f"Training for {num_epochs} epochs...")
    print("Target: Reconstruction MSE < 1e-3")
    
    losses = []
    reconstruction_errors = []
    best_mse = float('inf')
    patience = 50
    no_improve_count = 0
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        result = quantizer.forward(train_data, sample_rate, bandwidth=target_bandwidth)
        quantized, codes, bandwidth, commit_loss, quant_loss = result
        
        # Check for NaN values
        if torch.isnan(quantized).any() or torch.isnan(commit_loss) or torch.isnan(quant_loss):
            print(f"NaN detected at epoch {epoch+1}, stopping training")
            break
        
        # Compute reconstruction loss
        reconstruction_loss = F.mse_loss(train_data, quantized)
        
        # Total loss with balanced coefficients
        total_loss = reconstruction_loss + 0.01 * commit_loss + 0.01 * quant_loss
        
        # Check for inf/nan losses
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Invalid loss at epoch {epoch+1}: {total_loss.item()}")
            break
        
        # Backward pass with gradient clipping
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(quantizer.parameters(), max_norm=1.0)
        optimizer.step()
        
        current_mse = reconstruction_loss.item()
        losses.append(total_loss.item())
        reconstruction_errors.append(current_mse)
        
        # Early stopping check
        if current_mse < best_mse:
            best_mse = current_mse
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        # Print progress
        if (epoch + 1) % 50 == 0 or epoch < 10 or (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1:3d}: Total Loss = {total_loss.item():.6f}, "
                  f"Reconstruction MSE = {current_mse:.6f}")
        
        # Early stopping if no improvement
        if no_improve_count >= patience and epoch > 100:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
        
        # Stop if target reached
        if current_mse < 1e-3:
            print(f"Target MSE reached at epoch {epoch+1}")
            break
    
    # Final evaluation
    quantizer.eval()
    with torch.no_grad():
        final_result = quantizer.forward(train_data, sample_rate, bandwidth=target_bandwidth)
        final_quantized = final_result[0]
        final_reconstruction_error = F.mse_loss(train_data, final_quantized)
        
        print(f"\nFinal reconstruction MSE: {final_reconstruction_error.item():.8f}")
        
        # Test encode-decode consistency after training
        codes = quantizer.encode(train_data, sample_rate, bandwidth=target_bandwidth)
        decoded = quantizer.decode(codes)
        encode_decode_error = F.mse_loss(final_quantized, decoded)
        print(f"Encode-decode consistency MSE: {encode_decode_error.item():.8f}")
        
        # Compute per-sample reconstruction quality
        per_sample_mse = F.mse_loss(train_data, final_quantized, reduction='none').mean(dim=[1, 2])
        print(f"Per-sample MSE: {per_sample_mse.tolist()}")
        
        # Compute signal-to-noise ratio (SNR)
        signal_power = (train_data ** 2).mean()
        noise_power = ((train_data - final_quantized) ** 2).mean()
        snr_db = 10 * torch.log10(signal_power / (noise_power + 1e-8))
        print(f"Signal-to-Noise Ratio: {snr_db.item():.2f} dB")
        
        # Check overfitting success criteria
        overfitting_threshold = 1e-3
        if final_reconstruction_error < overfitting_threshold:
            print(f"✓ Overfitting test PASSED! MSE {final_reconstruction_error.item():.6f} < {overfitting_threshold}")
        else:
            print(f"✗ Overfitting test FAILED! MSE {final_reconstruction_error.item():.6f} >= {overfitting_threshold}")
            print("  Consider increasing training epochs, model capacity, or learning rate")
        
        # Test generalization (should be poor for overfitted model)
        print("\nTesting generalization on new data...")
        test_data = torch.randn(2, dimension, seq_len)
        test_result = quantizer.forward(test_data, sample_rate, bandwidth=target_bandwidth)
        test_reconstruction_error = F.mse_loss(test_data, test_result[0])
        print(f"Test data reconstruction MSE: {test_reconstruction_error.item():.6f}")
        
        if test_reconstruction_error > final_reconstruction_error * 10:
            print("✓ Model shows expected overfitting behavior (poor generalization)")
        else:
            print("? Model may not be fully overfitted (generalizes well)")
    
    # Plot training curve if possible
    print(f"\nTraining curve summary:")
    print(f"Initial reconstruction MSE: {reconstruction_errors[0]:.6f}")
    print(f"Final reconstruction MSE: {reconstruction_errors[-1]:.6f}")
    print(f"Improvement factor: {reconstruction_errors[0] / reconstruction_errors[-1]:.1f}x")
    
    # Check for convergence
    recent_losses = reconstruction_errors[-20:]  # Last 20 epochs
    loss_variance = torch.tensor(recent_losses).var().item()
    print(f"Loss variance in last 20 epochs: {loss_variance:.8f}")
    
    if loss_variance < 1e-8:
        print("✓ Training converged (low variance in recent losses)")
    else:
        print("? Training may not have fully converged")
    
    print("✓ Overfitting ability test completed!\n")


def main():
    """Run all tests for AutoGroupResidualVectorQuantizer."""
    print("🚀 Starting AutoGroupResidualVectorQuantizer Tests\n")
    
    try:
        # test_basic_functionality()
        # test_encode_decode_cycle()
        # test_bandwidth_control()
        # test_quantizer_dropout()
        # test_edge_cases()
        # test_gradient_flow()
        # test_different_configurations()
        test_overfitting_ability()
        
        # print("🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()