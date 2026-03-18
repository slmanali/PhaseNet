"""
Complex-Valued PhaseNet for Video Frame Interpolation

This is a complex-valued neural network implementation of PhaseNet,
which operates directly on complex steerable pyramid coefficients
without separating into amplitude and phase.

Key advantages of complex-valued approach:
1. Preserves phase relationships naturally
2. More efficient representation of oriented features
3. Better handles circular/periodic data (phase)
4. Reduced parameter count while maintaining expressiveness

Reference:
- Original PhaseNet: Meyer & Djelouah (CVPR 2018)
- Complex Networks: Trabelsi et al. (ICLR 2018)
"""

from torch import nn
import torch
from torch.nn import functional as F
import numpy as np

from .complex_nn import (
    ComplexConv2d,
    ComplexBatchNorm2d,
    ComplexLeakyReLU,
    ComplexTanh,
    ComplexSequential,
    complex_magnitude,
    complex_phase,
    polar2cart
)

pi = np.pi


class ComplexPhaseNetBlock(nn.Module):
    """
    Complex-valued PhaseNet building block.
    
    Consists of:
    - Two complex convolution layers
    - Complex batch normalization
    - Complex activation function (Leaky ReLU)
    
    Args:
        in_channels (int): Number of input channels (complex)
        out_channels (int): Number of output channels (complex)
        kernel_size (int): Size of convolution kernel
        padding (int): Padding size
    """
    
    def __init__(self, in_channels=44, out_channels=32, kernel_size=3, padding=1):
        super(ComplexPhaseNetBlock, self).__init__()
        
        self.conv1 = ComplexConv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = ComplexConv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn = ComplexBatchNorm2d(out_channels)
        self.activation = ComplexLeakyReLU(negative_slope=0.2)
    
    def forward(self, real, imag):
        """
        Forward pass through complex block.
        
        Args:
            real (torch.Tensor): Real part of input
            imag (torch.Tensor): Imaginary part of input
        
        Returns:
            tuple: (output_real, output_imag)
        """
        real, imag = self.conv1(real, imag)
        real, imag = self.conv2(real, imag)
        real, imag = self.bn(real, imag)
        real, imag = self.activation(real, imag)
        
        return real, imag


class ComplexPred(nn.Module):
    """
    Complex-valued prediction layer.
    
    Outputs complex-valued predictions that can be converted to
    amplitude and phase or used directly as complex coefficients.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of convolution kernel
    """
    
    def __init__(self, in_channels=32, out_channels=4, kernel_size=1):
        super(ComplexPred, self).__init__()
        self.conv = ComplexConv2d(in_channels, out_channels, kernel_size)
        self.activation = ComplexTanh()
    
    def forward(self, real, imag):
        """
        Forward pass for prediction.
        
        Args:
            real (torch.Tensor): Real part of input
            imag (torch.Tensor): Imaginary part of input
        
        Returns:
            tuple: (output_real, output_imag)
        """
        real, imag = self.conv(real, imag)
        real, imag = self.activation(real, imag)
        return real, imag


class ComplexPhaseNet(nn.Module):
    """
    Complex-valued PhaseNet for video frame interpolation.
    
    This network operates directly on complex steerable pyramid coefficients,
    maintaining the complex structure throughout the network.
    
    Architecture:
    - Input: Complex steerable pyramid coefficients from start and end frames
    - Processing: Hierarchical complex convolution blocks
    - Output: Complex coefficients for interpolated middle frame
    
    The network naturally handles phase wrapping and amplitude modulation
    through complex arithmetic operations.
    """
    
    def __init__(self, feature_dim=32):
        super(ComplexPhaseNet, self).__init__()
        
        # Learnable interpolation parameters
        # For residual (low-pass) level
        self.alpha = nn.Parameter(torch.rand(1))
        
        # For orientation bands (amplitude and phase)
        self.beta = nn.Parameter(torch.rand(1))
        
        self.feature_dim = feature_dim
        
        # Create network layers
        self.layer = nn.ModuleList()
        self.pred = nn.ModuleList()
        
        # Layer 0: Process residual (low-frequency) component
        # Input: 2 real channels (start and end frame residuals)
        # We treat this as 1 complex channel
        self.layer.append(ComplexPhaseNetBlock(1, feature_dim, 1, 0))
        self.pred.append(ComplexPred(feature_dim, 1))
        
        # Layer 1: First orientation band level
        # Input: 4 orientations × 2 frames = 8 complex channels + features from previous level
        # Total: ~40 channels (8 complex input + upsampled features + prediction)
        # input_ch_1 = 4 * 2 + feature_dim + 1  # 4 orientations, 2 frames, features, prediction
        input_ch_1 = 4 * 2 * 2 + feature_dim + 1  # 4 orientations × 2 frames × 2 (amp+phase) + features + prediction
        print(input_ch_1)
        # = 16 + 32 + 1 = 49 channels
        self.layer.append(ComplexPhaseNetBlock(input_ch_1, feature_dim, 1, 0))
        self.pred.append(ComplexPred(feature_dim, 4))
        input_ch_2 = 16 + feature_dim + 4 
        # Layer 2: Second orientation band level
        self.layer.append(ComplexPhaseNetBlock(input_ch_2, feature_dim, 1, 0))
        self.pred.append(ComplexPred(feature_dim, 4))
        
        # Layers 3-10: Remaining orientation band levels
        for _ in range(8):
            self.layer.append(ComplexPhaseNetBlock(input_ch_2, feature_dim))
            self.pred.append(ComplexPred(feature_dim, 4))
    
    def normalize_complex(self, real, imag, level_type='band'):
        """
        Normalize complex-valued inputs.
        
        For residuals: normalize by maximum magnitude
        For bands: normalize amplitudes by max, phases by pi
        
        Args:
            real (torch.Tensor): Real part
            imag (torch.Tensor): Imaginary part
            level_type (str): 'residual' or 'band'
        
        Returns:
            tuple: (normalized_real, normalized_imag)
        """
        if level_type == 'residual':
            # For residual, just normalize by max magnitude
            for i in range(real.shape[0]):
                mag = torch.sqrt(real[i]**2 + imag[i]**2).max()
                if mag > 0:
                    real[i] = real[i] / mag
                    imag[i] = imag[i] / mag
        else:
            # For bands, normalize amplitude and phase separately
            for i in range(real.shape[0]):
                # Get magnitude and phase
                mag = torch.sqrt(real[i]**2 + imag[i]**2 + 1e-8)
                phase = torch.atan2(imag[i], real[i])
                
                # Normalize magnitude by max
                mag = mag / (mag.max() + 1e-8)
                
                # Normalize phase by pi
                phase = phase / pi
                
                # Convert back to Cartesian
                real[i] = mag * torch.cos(phase * pi)
                imag[i] = mag * torch.sin(phase * pi)
        
        return real, imag
    
    def forward(self, x_real, x_imag):
        """
        Forward pass through ComplexPhaseNet.
        
        Args:
            x_real (list of torch.Tensor): Real parts of input pyramid levels
                x_real[0]: Residual level [N, 2, H, W]
                x_real[1:]: Band levels [N, 16, H, W] (4 orientations × 4 values)
            x_imag (list of torch.Tensor): Imaginary parts of input pyramid levels
        
        Returns:
            tuple: (output_real, output_imag) - lists of predicted coefficients
        """
        feature_map_real = []
        feature_map_imag = []
        pred_map_real = []
        pred_map_imag = []
        output_real = []
        output_imag = []
        
        # Process residual level (level 0)
        # Input is 2 real channels (start and end), treat as complex
        norm_real, norm_imag = self.normalize_complex(
            x_real[0], x_imag[0], 'residual'
        )
        
        # Reshape [N, 2, H, W] to [N, 1, H, W] for real and imag
        # Take mean as the complex representation
        residual_real = norm_real.mean(dim=1, keepdim=True)
        residual_imag = norm_imag.mean(dim=1, keepdim=True) if norm_imag.abs().sum() > 0 else torch.zeros_like(residual_real)
        
        feat_r, feat_i = self.layer[0](residual_real, residual_imag)
        feature_map_real.append(feat_r)
        feature_map_imag.append(feat_i)
        
        pred_r, pred_i = self.pred[0](feat_r, feat_i)
        pred_map_real.append(pred_r)
        pred_map_imag.append(pred_i)
        
        # Linear interpolation for residual
        out_r = self.alpha * x_real[0][:, 0:1, :, :] + (1 - self.alpha) * x_real[0][:, 1:2, :, :]
        out_i = self.alpha * x_imag[0][:, 0:1, :, :] + (1 - self.alpha) * x_imag[0][:, 1:2, :, :]
        output_real.append(out_r)
        output_imag.append(out_i)
        
        # Process band levels (levels 1+)
        for i in range(1, len(x_real)):
            img_shape = (x_real[i].shape[2], x_real[i].shape[3])
            
            # Normalize input
            norm_real, norm_imag = self.normalize_complex(
                x_real[i], x_imag[i], 'band'
            )
            
            # Upsample previous features and predictions
            feat_up_r = F.interpolate(feature_map_real[i-1], img_shape, mode='bilinear')
            feat_up_i = F.interpolate(feature_map_imag[i-1], img_shape, mode='bilinear')
            pred_up_r = F.interpolate(pred_map_real[i-1], img_shape, mode='bilinear')
            pred_up_i = F.interpolate(pred_map_imag[i-1], img_shape, mode='bilinear')
            
            # Concatenate all inputs
            concat_real = torch.cat([norm_real, feat_up_r, pred_up_r], dim=1)
            concat_imag = torch.cat([norm_imag, feat_up_i, pred_up_i], dim=1)
            
            # Forward through layer
            feat_r, feat_i = self.layer[i](concat_real, concat_imag)
            feature_map_real.append(feat_r)
            feature_map_imag.append(feat_i)
            
            pred_r, pred_i = self.pred[i](feat_r, feat_i)
            pred_map_real.append(pred_r)
            pred_map_imag.append(pred_i)
            
            # For bands: interpolate amplitude, predict phase
            # x_real/imag[i] has shape [N, 16, H, W] representing:
            # [amp0_start, amp1_start, amp2_start, amp3_start,
            #  phase0_start, phase1_start, phase2_start, phase3_start,
            #  amp0_end, amp1_end, amp2_end, amp3_end,
            #  phase0_end, phase1_end, phase2_end, phase3_end]
            
            # Linear interpolation for amplitude (first and third quarters)
            amp_r = self.beta * x_real[i][:, 0:4, :, :] + (1 - self.beta) * x_real[i][:, 8:12, :, :]
            amp_i = self.beta * x_imag[i][:, 0:4, :, :] + (1 - self.beta) * x_imag[i][:, 8:12, :, :]
            
            # Use predicted phase (network output)
            phase_r = pred_r
            phase_i = pred_i
            
            # Combine amplitude and phase
            out_r = torch.cat([amp_r, phase_r], dim=1)
            out_i = torch.cat([amp_i, phase_i], dim=1)
            
            output_real.append(out_r)
            output_imag.append(out_i)
        
        return output_real, output_imag


class ComplexTotalLoss(nn.Module):
    """
    Loss function for ComplexPhaseNet.
    
    Combines:
    1. Image reconstruction loss (L1)
    2. Complex-valued phase difference loss
    
    Args:
        v (float): Weight for phase loss term
    """
    
    def __init__(self, v=0.1):
        super(ComplexTotalLoss, self).__init__()
        self.v = v
    
    def forward(self, truth_real, truth_imag, pred_real, pred_imag, 
                truth_img, pred_img):
        """
        Compute total loss.
        
        Args:
            truth_real, truth_imag: Ground truth complex coefficients (lists)
            pred_real, pred_imag: Predicted complex coefficients (lists)
            truth_img: Ground truth image
            pred_img: Reconstructed image
        
        Returns:
            torch.Tensor: Total loss value
        """
        # Image reconstruction loss
        img_loss = nn.L1Loss()(truth_img, pred_img)
        
        # Phase difference loss for band levels
        phase_loss = 0
        num_bands = 0
        
        for i in range(1, len(truth_real)):
            # Extract phase channels (second half of each level)
            n_orient = truth_real[i].shape[1] // 2
            
            # Ground truth phase (complex representation)
            truth_phase_r = truth_real[i][:, n_orient:, :, :]
            truth_phase_i = truth_imag[i][:, n_orient:, :, :]
            
            # Predicted phase (complex representation)
            pred_phase_r = pred_real[i][:, n_orient:, :, :]
            pred_phase_i = pred_imag[i][:, n_orient:, :, :]
            
            # Compute phase angles
            truth_angle = torch.atan2(truth_phase_i, truth_phase_r)
            pred_angle = torch.atan2(pred_phase_i, pred_phase_r)
            
            # Phase difference (wrapped to [-pi, pi])
            dphase = truth_angle - pred_angle
            dphase_wrapped = torch.atan2(torch.sin(dphase), torch.cos(dphase))
            
            # L1 loss on wrapped phase difference
            phase_loss += nn.L1Loss()(dphase_wrapped, torch.zeros_like(dphase_wrapped))
            num_bands += 1
        
        # Average phase loss over all band levels
        if num_bands > 0:
            phase_loss = phase_loss / num_bands
        
        return self.v * phase_loss + img_loss

def _select_frame_coeff(level_coeff, frame_idx):
    """Return a single frame from a pyramid coefficient tensor.

    Band tensors produced by ``extract_complex_coefficients`` are usually
    shaped ``[frames, H, W]`` after splitting real/imaginary parts, but older
    code paths may still carry a singleton channel dimension as
    ``[frames, 1, H, W]``. This helper accepts both layouts.
    """
    if level_coeff.dim() == 4:
        return level_coeff[frame_idx, 0, :, :]
    if level_coeff.dim() == 3:
        return level_coeff[frame_idx, :, :]
    raise ValueError(
        f"Expected coefficient tensor with 3 or 4 dims, got shape {tuple(level_coeff.shape)}"
    )

def complex_input_convert(Tri_coeff_real, Tri_coeff_imag):
    """
    Convert triplet of pyramid coefficients to training format.
    
    Args:
        Tri_coeff_real: Real parts of coefficients [level][orientation][frame]
        Tri_coeff_imag: Imaginary parts of coefficients [level][orientation][frame]
    
    Returns:
        tuple: (train_real, train_imag, truth_real, truth_imag)
               train contains start and end frames
               truth contains middle frame
    """
    # Reverse to go from coarse to fine
    Tri_coeff_real_inv = Tri_coeff_real[::-1]
    Tri_coeff_imag_inv = Tri_coeff_imag[::-1]
    
    train_real = []
    train_imag = []
    truth_real = []
    truth_imag = []
    
    # Low level residual
    train_real.append(torch.stack([Tri_coeff_real_inv[0][0], Tri_coeff_real_inv[0][2]]))
    train_imag.append(torch.stack([Tri_coeff_imag_inv[0][0], Tri_coeff_imag_inv[0][2]]))
    truth_real.append(Tri_coeff_real_inv[0][1].unsqueeze(0))
    truth_imag.append(Tri_coeff_imag_inv[0][1].unsqueeze(0))
    
    # Band levels
    for i in range(1, len(Tri_coeff_real_inv) - 1):
        start_amps = []
        start_phases_r = []
        start_phases_i = []
        end_amps = []
        end_phases_r = []
        end_phases_i = []
        mid_amps = []
        mid_phases_r = []
        mid_phases_i = []

        n_orient = len(Tri_coeff_real_inv[i])

        for orient_idx in range(n_orient):
            orient_real = Tri_coeff_real_inv[i][orient_idx]
            orient_imag = Tri_coeff_imag_inv[i][orient_idx]

            start_r = _select_frame_coeff(orient_real, 0)
            start_i = _select_frame_coeff(orient_imag, 0)
            mid_r = _select_frame_coeff(orient_real, 1)
            mid_i = _select_frame_coeff(orient_imag, 1)
            end_r = _select_frame_coeff(orient_real, 2)
            end_i = _select_frame_coeff(orient_imag, 2)

            start_amp = torch.sqrt(start_r**2 + start_i**2 + 1e-8)
            start_phase = torch.atan2(start_i, start_r)
            mid_amp = torch.sqrt(mid_r**2 + mid_i**2 + 1e-8)
            mid_phase = torch.atan2(mid_i, mid_r)
            end_amp = torch.sqrt(end_r**2 + end_i**2 + 1e-8)
            end_phase = torch.atan2(end_i, end_r)

            start_amps.append(start_amp)
            start_phases_r.append(torch.cos(start_phase))
            start_phases_i.append(torch.sin(start_phase))
            end_amps.append(end_amp)
            end_phases_r.append(torch.cos(end_phase))
            end_phases_i.append(torch.sin(end_phase))
            mid_amps.append(mid_amp)
            mid_phases_r.append(torch.cos(mid_phase))
            mid_phases_i.append(torch.sin(mid_phase))

        train_real.append(torch.stack(start_amps + start_phases_r + end_amps + end_phases_r))
        train_imag.append(torch.stack(
            [torch.zeros_like(amp) for amp in start_amps]
            + start_phases_i
            + [torch.zeros_like(amp) for amp in end_amps]
            + end_phases_i
        ))
        truth_real.append(torch.stack(mid_amps + mid_phases_r))
        truth_imag.append(torch.stack(
            [torch.zeros_like(amp) for amp in mid_amps] + mid_phases_i
        ))
    
    return train_real, train_imag, truth_real, truth_imag


if __name__ == "__main__":
    # Test the complex-valued PhaseNet
    model = ComplexPhaseNet(feature_dim=32)
    print(model)
    print("\nModel parameters:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test with dummy input
    test_input_real = []
    test_input_imag = []
    test_input_real.append(torch.randn(2, 2, 8, 8))
    test_input_imag.append(torch.randn(2, 2, 8, 8))
    
    for size in [16, 24, 32, 46, 64, 90, 128, 182, 256]:
        test_input_real.append(torch.randn(2, 16, size, size))
        test_input_imag.append(torch.randn(2, 16, size, size))
    
    print("\nTesting forward pass...")
    with torch.no_grad():
        out_r, out_i = model(test_input_real, test_input_imag)
    
    print(f"Output levels: {len(out_r)}")
    for i, (r, im) in enumerate(zip(out_r, out_i)):
        print(f"Level {i}: Real shape {r.shape}, Imag shape {im.shape}")
