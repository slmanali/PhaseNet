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
    
    def __init__(self, feature_dim=64):
        super(ComplexPhaseNet, self).__init__()
        
        # Learnable interpolation parameters
        # For residual (low-pass) level
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        # For orientation bands (amplitude and phase)
        self.beta = nn.Parameter(torch.tensor(0.5))
        
        # NEW: learnable per-level amplitude gain (critical for pyramid scales)
        self.amp_gain = nn.Parameter(torch.ones(11))   # 10 pyramid levels

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
        # = 16 + 32 + 1 = 49 channels
        self.layer.append(ComplexPhaseNetBlock(input_ch_1, feature_dim, 1, 0))
        self.pred.append(ComplexPred(feature_dim, 8))
        input_ch_2 = 16 + feature_dim + 8
        # Layer 2: Second orientation band level
        self.layer.append(ComplexPhaseNetBlock(input_ch_2, feature_dim, 1, 0))
        self.pred.append(ComplexPred(feature_dim, 8))
        
        # Layers 3-10: Remaining orientation band levels
        for _ in range(8):
            self.layer.append(ComplexPhaseNetBlock(input_ch_2, feature_dim))
            self.pred.append(ComplexPred(feature_dim, 8))
    
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
    
    def normalize_unit_complex(self, real, imag, eps=1e-6):
        """Safe version that prevents division by zero."""
        mag = torch.sqrt(real ** 2 + imag ** 2 + eps)
        mag = torch.clamp(mag, min=eps)
        return real / mag, imag / mag
    
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
        # Save original scale before normalization (non-destructive)
        residual_scale = torch.zeros(x_real[0].shape[0], 1, 1, 1, device=x_real[0].device)
        for b in range(x_real[0].shape[0]):
            mag = torch.sqrt(x_real[0][b]**2 + x_imag[0][b]**2 + 1e-8).max()
            residual_scale[b, 0, 0, 0] = mag

        norm_real, norm_imag = self.normalize_complex(
            x_real[0], x_imag[0], 'residual'
        )
        
        # Reshape to treat as single complex channel
        residual_real = norm_real.mean(dim=1, keepdim=True)
        residual_imag = norm_imag.mean(dim=1, keepdim=True) if norm_imag.abs().sum() > 0 else torch.zeros_like(residual_real)
        
        feat_r, feat_i = self.layer[0](residual_real, residual_imag)
        feature_map_real.append(feat_r)
        feature_map_imag.append(feat_i)
        
        pred_r, pred_i = self.pred[0](feat_r, feat_i)
        pred_map_real.append(pred_r)
        pred_map_imag.append(pred_i)

        # Base interpolation and correction — all in normalized space
        base_r = self.alpha * norm_real[:, 0:1, :, :] + (1 - self.alpha) * norm_real[:, 1:2, :, :]
        base_i = self.alpha * norm_imag[:, 0:1, :, :] + (1 - self.alpha) * norm_imag[:, 1:2, :, :]

        out_r_norm = base_r + pred_r
        out_i_norm = base_i + pred_i

        # Denormalize back to original pyramid scale (matching truth and reconstruction)
        out_r = out_r_norm * residual_scale
        out_i = out_i_norm * residual_scale

        output_real.append(out_r)
        output_imag.append(out_i)
        
        # Process band levels (levels 1+)
        for i in range(1, len(x_real)):
            img_shape = (x_real[i].shape[2], x_real[i].shape[3])
            
            # ───────────────────────────────────────────────────────────────
            # Proper per-component normalization for band levels             ← NEW
            # Amplitudes (ch 0–3 start, 8–11 end) → scale to roughly [0,1]   ← NEW
            # Phases   (ch 4–7 start, 12–15 end) → keep on unit circle       ← NEW
            # ───────────────────────────────────────────────────────────────
            amp_channels = [0,1,2,3,8,9,10,11]
            phase_channels_real = [4,5,6,7,12,13,14,15]
            phase_channels_imag = [4,5,6,7,12,13,14,15]

            amp_max = torch.max(
                torch.abs(x_real[i][:, amp_channels]),
                dim=1, keepdim=True
            )[0].clamp(min=1e-6)
            norm_real = x_real[i].clone()
            norm_imag = x_imag[i].clone()
            norm_real[:, amp_channels] = norm_real[:, amp_channels] / amp_max

            norm_real[:, phase_channels_real] = torch.clamp(
                norm_real[:, phase_channels_real], -1.0, 1.0
            )
            norm_imag[:, phase_channels_imag] = torch.clamp(
                norm_imag[:, phase_channels_imag], -1.1, 1.1
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
            
            # Linear interpolation for amplitude
            base_amp = self.beta * norm_real[:, 0:4, :, :] + (1 - self.beta) * norm_real[:, 8:12, :, :]

            amp_delta = pred_r[:, 0:4, :, :]
            amp_delta = torch.clamp(amp_delta, min=-0.3, max=0.3)
            amp_r = torch.relu(base_amp + amp_delta)
            amp_r = amp_r * amp_max
            
            # NEW: per-level gain (this is what was missing to beat Phase Based)
            amp_r = amp_r * self.amp_gain[i].view(1, 1, 1, 1)
            
            amp_i = torch.zeros_like(amp_r)

            # start/end phase from input
            start_phase_r = x_real[i][:, 4:8, :, :]
            start_phase_i = x_imag[i][:, 4:8, :, :]
            end_phase_r = x_real[i][:, 12:16, :, :]
            end_phase_i = x_imag[i][:, 12:16, :, :]

            # midpoint base phase
            base_phase_r = start_phase_r + end_phase_r
            base_phase_i = start_phase_i + end_phase_i
            base_phase_r, base_phase_i = self.normalize_unit_complex(base_phase_r, base_phase_i)

            # predict a SMALL angular residual; zero prediction => identity correction
            delta_theta = np.pi * pred_r[:, 4:8, :, :]
            delta_theta = torch.clamp(delta_theta, min=-np.pi/2, max=np.pi/2)

            cos_d = torch.cos(delta_theta)
            sin_d = torch.sin(delta_theta)

            phase_r = base_phase_r * cos_d - base_phase_i * sin_d
            phase_i = base_phase_r * sin_d + base_phase_i * cos_d
            phase_r, phase_i = self.normalize_unit_complex(phase_r, phase_i)

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
    
    def __init__(
        self,
        v=0.1,
        amp_weight=0.5,
        amp_imag_weight=0.1,
        phase_unit_weight=0.1,
        residual_weight=1.0,
        residual_imag_weight=0.1,
    ):
        super(ComplexTotalLoss, self).__init__()
        self.v = v
        self.amp_weight = amp_weight
        self.amp_imag_weight = amp_imag_weight
        self.phase_unit_weight = phase_unit_weight
        self.residual_weight = residual_weight
        self.residual_imag_weight = residual_imag_weight
    
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
        
        residual_real_loss = nn.L1Loss()(pred_real[0], truth_real[0])
        residual_imag_loss = nn.L1Loss()(pred_imag[0], truth_imag[0])
        # Phase difference loss for band levels
        phase_loss = 0
        amp_loss = 0
        amp_imag_loss = 0
        num_bands = 0
        phase_unit_loss = 0
        
        for i in range(1, len(truth_real)):
            # Extract phase channels (second half of each level)
            n_orient = truth_real[i].shape[1] // 2

            truth_amp = truth_real[i][:, :n_orient, :, :]
            pred_amp = pred_real[i][:, :n_orient, :, :]
            pred_amp_imag = pred_imag[i][:, :n_orient, :, :]
            
            # Ground truth phase (complex representation)
            truth_phase_r = truth_real[i][:, n_orient:, :, :]
            truth_phase_i = truth_imag[i][:, n_orient:, :, :]
            
            # Predicted phase (complex representation)
            pred_phase_r = pred_real[i][:, n_orient:, :, :]
            pred_phase_i = pred_imag[i][:, n_orient:, :, :]

            pred_phase_mag = torch.sqrt(pred_phase_r ** 2 + pred_phase_i ** 2 + 1e-8)
            phase_unit_loss += nn.L1Loss()(pred_phase_mag, torch.ones_like(pred_phase_mag))

                        
            # Compute phase angles
            truth_angle = torch.atan2(truth_phase_i, truth_phase_r)
            pred_angle = torch.atan2(pred_phase_i, pred_phase_r)
            
            # Phase difference (wrapped to [-pi, pi])
            dphase = truth_angle - pred_angle
            dphase_wrapped = torch.atan2(torch.sin(dphase), torch.cos(dphase))
            
            # L1 loss on wrapped phase difference
            amp_loss += nn.L1Loss()(pred_amp, truth_amp)
            amp_imag_loss += nn.L1Loss()(pred_amp_imag, torch.zeros_like(pred_amp_imag))
            phase_loss += nn.L1Loss()(dphase_wrapped, torch.zeros_like(dphase_wrapped))
            num_bands += 1
        
        # Average phase loss over all band levels
        if num_bands > 0:
            phase_loss = phase_loss / num_bands
            amp_loss = amp_loss / num_bands
            amp_imag_loss = amp_imag_loss / num_bands
            phase_unit_loss = phase_unit_loss / num_bands

        total_loss = (
            img_loss
            + self.residual_weight * residual_real_loss
            + self.residual_imag_weight * residual_imag_loss
            + self.v * phase_loss
            + self.phase_unit_weight * phase_unit_loss
            + self.amp_weight * amp_loss
            + self.amp_imag_weight * amp_imag_loss
        )

        self.last_stats = {
            "img": float(img_loss.detach().cpu()),
            "residual_real": float(residual_real_loss.detach().cpu()),
            "residual_imag": float(residual_imag_loss.detach().cpu()),
            "phase": float(phase_loss.detach().cpu()),
            "phase_unit": float(phase_unit_loss.detach().cpu()),
            "amp": float(amp_loss.detach().cpu()),
            "amp_imag": float(amp_imag_loss.detach().cpu()),
            "total": float(total_loss.detach().cpu()),
        }

        return total_loss

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
