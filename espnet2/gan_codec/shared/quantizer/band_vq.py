import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Optional, Tuple, Union, Any, Callable
from einops import rearrange
from espnet2.gan_codec.shared.quantizer.modules.simvq import SimVQ
from espnet2.gan_codec.shared.quantizer.modules.core_vq import BandVectorQuantization

class BandVQ(nn.Module):

    def __init__(
        self,
        num_bands: int = 3,
        dimension: int = 512,
        codebook_dim: int = 512,
        bins: int = 1024,
        decay: float = 0.99,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        quantizer_dropout: bool = False,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.dimension = dimension
        self.codebook_dim = codebook_dim
        self.bins = bins
        self.decay = decay
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.quantizer_dropout = quantizer_dropout
        self.vq = BandVectorQuantization(
            num_bands=self.num_bands,
            dim=self.dimension,
            codebook_dim=self.codebook_dim,
            codebook_size=self.bins,
            decay=self.decay,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            threshold_ema_dead_code=self.threshold_ema_dead_code,
            quantizer_dropout=self.quantizer_dropout
        )

    def forward(
        self, x: torch.Tensor, bandwidth: Optional[float] = None
    ):
        """Residual vector quantization on the given input tensor.

        Args:
            x (torch.Tensor): Input tensor, [B,band,d,n]
            bandwidth (float): Target bandwidth.
        Returns:
            BandQuantizedResult:
                The quantized (or approximately quantized) representation with
                the associated bandwidth and any penalty term for the loss.
        """
        quantized, codes, commit_loss = self.vq(x)
        return (
            quantized,
            codes,
            torch.mean(commit_loss)
        )

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a given input tensor with the specified sample rate at

        the given bandwidth. The RVQ encode method sets the appropriate
        number of quantizer to use and returns indices for each quantizer.
        """
        codes = self.vq.encode(x)
        return codes

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode the given codes to the quantized representation."""
        
        quantized = self.vq.decode(codes)
        return quantized


class BandSimVQ(nn.Module):
    """
    Band Simple Vector Quantization.
    This version is adapted to work with the provided SimVQ adapter.
    """
    
    def __init__(
        self,
        num_bands: int,
        dim: int,
        codebook_size: int,
        codebook_dim: Optional[int] = None,
        channel_first: bool = True,
        rotation_trick: bool = False,
    ):
        """
        Args:
            num_bands: Number of frequency bands.
            dim: Dimension of input features.
            codebook_size: Size of the codebook.
            codebook_dim: Dimension of codebook vectors.
        """
        super().__init__()
        
        self.num_bands = num_bands
        self.dim = dim
        
        # Create ResidualSimVQ layers for each band using the correct class name.
        self.layers = nn.ModuleList([
            SimVQ(
                dim=dim,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                rotation_trick=rotation_trick,
                # Pass arguments for interface compatibility.
                # The adapter's SimVQ expects `channel_first` to handle [B, D, T] inputs.
                channel_first=True,
            )
            for _ in range(num_bands)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through Band Residual Similarity VQ.
        
        Args:
            x: Input tensor of shape [B, bands, D, T].
            
        Returns:
            quantized_per_band: Quantized output [B, bands, D, T].
            all_indices: Indices [B, bands, T].
            total_loss: Total VQ loss for training.
        """
        B, bands, D, T = x.shape
        assert bands == self.num_bands

        quantized_per_band = []
        all_indices = []
        all_simvq_losses = []
        
        for i in range(bands):
            # Add residual to current band
            inp = x[:, i, :, :]  # Shape: [B, D, T]
            
            # The adapter's `forward` method always returns 3 values:
            # (quantized, indices, total_commitment_loss)
            # It expects `num_quantizers` as the argument name for dropout control.
            quantized, indices, simvq_loss = self.layers[i](
                inp
            )
            
            # Reshape indices from [B, num_quantizers, T] to match expected output format
            # This is done by stacking later. The shape is correct.
            
            quantized_per_band.append(quantized)
            all_indices.append(indices)
            all_simvq_losses.append(simvq_loss)
        
        # Stack results
        quantized_per_band = torch.stack(quantized_per_band, dim=1) # [B, bands, D, T]
        all_indices = torch.stack(all_indices, dim=1) # [B, bands, T]
        
        # The provided adapter gives a summed loss per band, so we average these.
        total_loss = torch.stack(all_simvq_losses).mean()
        
        return quantized_per_band, all_indices, total_loss
    
    def encode(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode input to indices.
        
        Args:
            x: Input tensor [B, bands, D, T].
            
        Returns:
            indices: Indices tensor [B, bands, T].
        """
        _, indices, _ = self.forward(x)
        return indices
    
    def decode(
        self,
        indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode indices to reconstructed signal.
        
        Args:
            indices: Indices tensor [B, bands, num_quantizers, T].
            
        Returns:
            reconstructed: Reconstructed signal [B, bands, D, T].
        """
        B, bands = indices.shape[:2]
        assert bands == self.num_bands
        
        reconstructed = []
        for i in range(bands):
            # Call the correct decoding method on the adapter: `from_indices`.
            # The input shape [B, num_quantizers, T] is what `from_indices` expects.
            codes = self.layers[i].indices_to_codes(indices[:, i,:])
            # The output `codes` will be [B, D, T], which is the correct shape.
            reconstructed.append(codes)
        
        return torch.stack(reconstructed, dim=1)
