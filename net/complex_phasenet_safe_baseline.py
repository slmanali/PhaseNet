from torch import nn
import torch
from torch.nn import functional as F
import numpy as np

from .complex_nn import (
    ComplexConv2d,
    ComplexBatchNorm2d,
    ComplexLeakyReLU,
    ComplexTanh,
)

pi = np.pi


class ComplexPhaseNetBlock(nn.Module):
    def __init__(self, in_channels=44, out_channels=32, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = ComplexConv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = ComplexConv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn = ComplexBatchNorm2d(out_channels)
        self.activation = ComplexLeakyReLU(negative_slope=0.2)

    def forward(self, real, imag):
        real, imag = self.conv1(real, imag)
        real, imag = self.conv2(real, imag)
        real, imag = self.bn(real, imag)
        real, imag = self.activation(real, imag)
        return real, imag


class ComplexPred(nn.Module):
    def __init__(self, in_channels=32, out_channels=4, kernel_size=1):
        super().__init__()
        self.conv = ComplexConv2d(in_channels, out_channels, kernel_size)
        self.activation = ComplexTanh()

    def forward(self, real, imag):
        real, imag = self.conv(real, imag)
        real, imag = self.activation(real, imag)
        return real, imag


class ComplexPhaseNetSafe(nn.Module):
    """
    Safer complex-valued PhaseNet closer to the real safe baseline.

    Key behavior:
    - Residual: base interpolation + small learned complex correction
    - Amplitude: pure interpolation only (no learned amplitude delta)
    - Phase: wrap-aware midpoint + small learned angular correction

    Output format remains compatible with the existing reconstruction pipeline:
    - level 0: residual complex coefficient
    - level i>0: [amp, phase] where amplitude lives in real part and phase is unit complex
    """

    def __init__(
        self,
        feature_dim=64,
        phase_correction_scale=0.1,
        residual_correction_scale=0.1,
    ):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.feature_dim = feature_dim
        self.phase_correction_scale = phase_correction_scale
        self.residual_correction_scale = residual_correction_scale

        self.layer = nn.ModuleList()
        self.pred = nn.ModuleList()

        # Residual level: one complex channel in, one complex channel out
        self.layer.append(ComplexPhaseNetBlock(1, feature_dim, 1, 0))
        self.pred.append(ComplexPred(feature_dim, 1))

        # Band levels: 16 input channels + previous feat + previous pred
        # Prediction only needs 4 channels: one phase correction per orientation
        input_ch_1 = 16 + feature_dim + 1
        self.layer.append(ComplexPhaseNetBlock(input_ch_1, feature_dim, 1, 0))
        self.pred.append(ComplexPred(feature_dim, 4))

        input_ch_n = 16 + feature_dim + 4
        self.layer.append(ComplexPhaseNetBlock(input_ch_n, feature_dim, 1, 0))
        self.pred.append(ComplexPred(feature_dim, 4))

        for _ in range(8):
            self.layer.append(ComplexPhaseNetBlock(input_ch_n, feature_dim))
            self.pred.append(ComplexPred(feature_dim, 4))

    @staticmethod
    def normalize_unit_complex(real, imag, eps=1e-6):
        mag = torch.sqrt(real ** 2 + imag ** 2 + eps)
        mag = torch.clamp(mag, min=eps)
        return real / mag, imag / mag

    @staticmethod
    def normalize_residual(real, imag):
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        scale = mag.amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-4)
        return real / scale, imag / scale, scale

    @staticmethod
    def normalize_band(real, imag):
        # amplitude channels: 0:4 and 8:12, phase channels: 4:8 and 12:16
        amp_channels = [0, 1, 2, 3, 8, 9, 10, 11]
        amp_mag = torch.sqrt(real[:, amp_channels] ** 2 + imag[:, amp_channels] ** 2 + 1e-8)
        scale = amp_mag.mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-4)

        norm_real = real.clone()
        norm_imag = imag.clone()
        norm_real[:, amp_channels] = real[:, amp_channels] / scale
        norm_imag[:, amp_channels] = imag[:, amp_channels] / scale
        return norm_real, norm_imag, scale

    @staticmethod
    def _run_tiled(module, real, imag, tile_size, halo):
        """Run a local complex module in spatial tiles without introducing seams."""
        height, width = real.shape[-2:]
        if not tile_size or (height <= tile_size and width <= tile_size):
            return module(real, imag)

        real_rows = []
        imag_rows = []
        for top in range(0, height, tile_size):
            bottom = min(top + tile_size, height)
            real_tiles = []
            imag_tiles = []
            for left in range(0, width, tile_size):
                right = min(left + tile_size, width)
                expanded_top = max(0, top - halo)
                expanded_bottom = min(height, bottom + halo)
                expanded_left = max(0, left - halo)
                expanded_right = min(width, right + halo)
                tile_real, tile_imag = module(
                    real[..., expanded_top:expanded_bottom, expanded_left:expanded_right],
                    imag[..., expanded_top:expanded_bottom, expanded_left:expanded_right],
                )
                crop_top = top - expanded_top
                crop_left = left - expanded_left
                real_tiles.append(tile_real[..., crop_top:crop_top + bottom - top,
                                            crop_left:crop_left + right - left])
                imag_tiles.append(tile_imag[..., crop_top:crop_top + bottom - top,
                                            crop_left:crop_left + right - left])
            real_rows.append(torch.cat(real_tiles, dim=-1))
            imag_rows.append(torch.cat(imag_tiles, dim=-1))
        return torch.cat(real_rows, dim=-2), torch.cat(imag_rows, dim=-2)

    def forward(self, x_real, x_imag, tile_size=None):
        output_real = []
        output_imag = []

        # Residual level
        norm_real, norm_imag, residual_scale = self.normalize_residual(x_real[0], x_imag[0])
        residual_real = norm_real.mean(dim=1, keepdim=True)
        residual_imag = norm_imag.mean(dim=1, keepdim=True)

        feat_r, feat_i = self._run_tiled(
            self.layer[0], residual_real, residual_imag, tile_size, halo=0
        )
        pred_r, pred_i = self._run_tiled(self.pred[0], feat_r, feat_i, tile_size, halo=0)

        # Only the immediately preceding feature and prediction are consumed by
        # the next level.  Keeping every level alive is particularly expensive
        # for native-resolution evaluation, where the feature maps grow at each
        # step through the pyramid.
        previous_feat_r, previous_feat_i = feat_r, feat_i
        previous_pred_r, previous_pred_i = pred_r, pred_i

        base_r = self.alpha * norm_real[:, 0:1] + (1.0 - self.alpha) * norm_real[:, 1:2]
        base_i = self.alpha * norm_imag[:, 0:1] + (1.0 - self.alpha) * norm_imag[:, 1:2]

        out_r = (base_r + self.residual_correction_scale * pred_r) * residual_scale
        out_i = (base_i + self.residual_correction_scale * pred_i) * residual_scale

        output_real.append(out_r)
        output_imag.append(out_i)

        # Band levels
        for i in range(1, len(x_real)):
            img_shape = (x_real[i].shape[2], x_real[i].shape[3])
            norm_real, norm_imag, amp_scale = self.normalize_band(x_real[i], x_imag[i])

            feat_up_r = F.interpolate(previous_feat_r, img_shape, mode='bilinear', align_corners=False)
            feat_up_i = F.interpolate(previous_feat_i, img_shape, mode='bilinear', align_corners=False)
            pred_up_r = F.interpolate(previous_pred_r, img_shape, mode='bilinear', align_corners=False)
            pred_up_i = F.interpolate(previous_pred_i, img_shape, mode='bilinear', align_corners=False)

            concat_real = torch.cat([norm_real, feat_up_r, pred_up_r], dim=1)
            concat_imag = torch.cat([norm_imag, feat_up_i, pred_up_i], dim=1)

            # Levels 1 and 2 use 1x1 convolutions; later blocks contain two
            # 3x3 convolutions and therefore need a two-pixel input halo.
            feat_r, feat_i = self._run_tiled(
                self.layer[i], concat_real, concat_imag, tile_size,
                halo=0 if i <= 2 else 2,
            )
            pred_r, pred_i = self._run_tiled(
                self.pred[i], feat_r, feat_i, tile_size, halo=0
            )

            previous_feat_r, previous_feat_i = feat_r, feat_i
            previous_pred_r, previous_pred_i = pred_r, pred_i

            # Pure amplitude interpolation only, like safe real baseline
            base_amp = self.beta * norm_real[:, 0:4] + (1.0 - self.beta) * norm_real[:, 8:12]
            amp_r = torch.clamp(base_amp, min=1e-4) * amp_scale
            amp_i = torch.zeros_like(amp_r)

            # Wrap-aware base phase midpoint
            start_phase_r = norm_real[:, 4:8]
            start_phase_i = norm_imag[:, 4:8]
            end_phase_r = norm_real[:, 12:16]
            end_phase_i = norm_imag[:, 12:16]

            base_phase_r = start_phase_r + end_phase_r
            base_phase_i = start_phase_i + end_phase_i
            base_phase_r, base_phase_i = self.normalize_unit_complex(base_phase_r, base_phase_i)

            # Small learned angular correction only from pred real branch
            delta_theta = self.phase_correction_scale * pi * pred_r[:, 0:4]
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
    Safer complex loss closer to the safe real baseline.

    Default behavior:
    - modest image and gradient terms
    - residual real/imag supervision
    - phase supervision on unit-complex phase
    - amplitude supervision optional and off by default
    """

    def __init__(
        self,
        img_weight=0.05,
        residual_weight=0.1,
        residual_imag_weight=0.05,
        phase_weight=0.2,
        amp_weight=0.0,
        amp_imag_weight=0.0,
        phase_unit_weight=0.01,
        grad_weight=0.1,
    ):
        super().__init__()
        self.img_weight = img_weight
        self.residual_weight = residual_weight
        self.residual_imag_weight = residual_imag_weight
        self.phase_weight = phase_weight
        self.amp_weight = amp_weight
        self.amp_imag_weight = amp_imag_weight
        self.phase_unit_weight = phase_unit_weight
        self.grad_weight = grad_weight
        self.last_stats = {}

    @staticmethod
    def gradient_loss(x, y):
        dx = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        dy = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        dx_gt = torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
        dy_gt = torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])
        return torch.mean(torch.abs(dx - dx_gt)) + torch.mean(torch.abs(dy - dy_gt))

    def forward(self, truth_real, truth_imag, pred_real, pred_imag, truth_img, pred_img):
        img_loss = F.l1_loss(pred_img, truth_img)
        grad_loss = self.gradient_loss(pred_img, truth_img)

        residual_real_loss = F.l1_loss(pred_real[0], truth_real[0])
        residual_imag_loss = F.l1_loss(pred_imag[0], truth_imag[0])

        phase_loss = pred_img.new_tensor(0.0)
        amp_loss = pred_img.new_tensor(0.0)
        amp_imag_loss = pred_img.new_tensor(0.0)
        phase_unit_loss = pred_img.new_tensor(0.0)
        num_bands = max(len(truth_real) - 1, 1)

        for i in range(1, len(truth_real)):
            n_orient = truth_real[i].shape[1] // 2

            truth_amp = truth_real[i][:, :n_orient]
            pred_amp = pred_real[i][:, :n_orient]
            pred_amp_imag = pred_imag[i][:, :n_orient]

            truth_phase_r = truth_real[i][:, n_orient:]
            truth_phase_i = truth_imag[i][:, n_orient:]
            pred_phase_r = pred_real[i][:, n_orient:]
            pred_phase_i = pred_imag[i][:, n_orient:]

            pred_phase_mag = torch.sqrt(pred_phase_r ** 2 + pred_phase_i ** 2 + 1e-8)
            phase_unit_loss = phase_unit_loss + F.l1_loss(pred_phase_mag, torch.ones_like(pred_phase_mag))

            truth_angle = torch.atan2(truth_phase_i+ 1e-8, truth_phase_r+ 1e-8)
            pred_angle = torch.atan2(pred_phase_i+ 1e-8, pred_phase_r+ 1e-8)
            dphase = torch.atan2(torch.sin(truth_angle - pred_angle)+ 1e-8, torch.cos(truth_angle - pred_angle)+ 1e-8)
            phase_loss = phase_loss + F.l1_loss(dphase, torch.zeros_like(dphase))

            if self.amp_weight > 0.0:
                amp_loss = amp_loss + F.l1_loss(pred_amp, truth_amp)
            if self.amp_imag_weight > 0.0:
                amp_imag_loss = amp_imag_loss + F.l1_loss(pred_amp_imag, torch.zeros_like(pred_amp_imag))

        phase_loss = phase_loss / num_bands
        phase_unit_loss = phase_unit_loss / num_bands
        if self.amp_weight > 0.0:
            amp_loss = amp_loss / num_bands
        if self.amp_imag_weight > 0.0:
            amp_imag_loss = amp_imag_loss / num_bands

        total_loss = (
            self.img_weight * img_loss
            + self.grad_weight * grad_loss
            + self.residual_weight * residual_real_loss
            + self.residual_imag_weight * residual_imag_loss
            + self.phase_weight * phase_loss
            + self.phase_unit_weight * phase_unit_loss
            + self.amp_weight * amp_loss
            + self.amp_imag_weight * amp_imag_loss
        )

        self.last_stats = {
            'img': float(img_loss.detach().cpu()),
            'grad': float(grad_loss.detach().cpu()),
            'residual_real': float(residual_real_loss.detach().cpu()),
            'residual_imag': float(residual_imag_loss.detach().cpu()),
            'phase': float(phase_loss.detach().cpu()),
            'phase_unit': float(phase_unit_loss.detach().cpu()),
            'amp': float(amp_loss.detach().cpu()),
            'amp_imag': float(amp_imag_loss.detach().cpu()),
            'total': float(total_loss.detach().cpu()),
        }

        return total_loss


def _select_frame_coeff(level_coeff, frame_idx):
    if level_coeff.dim() == 4:
        return level_coeff[frame_idx, 0, :, :]
    if level_coeff.dim() == 3:
        return level_coeff[frame_idx, :, :]
    raise ValueError(f"Expected 3D or 4D coeff tensor, got {tuple(level_coeff.shape)}")



def complex_input_convert(Tri_coeff_real, Tri_coeff_imag):
    Tri_coeff_real_inv = Tri_coeff_real[::-1]
    Tri_coeff_imag_inv = Tri_coeff_imag[::-1]

    train_real = []
    train_imag = []
    truth_real = []
    truth_imag = []

    highpass_start = Tri_coeff_real[0][0]
    highpass_end = Tri_coeff_real[0][2]

    train_real.append(torch.stack([Tri_coeff_real_inv[0][0], Tri_coeff_real_inv[0][2]]))
    train_imag.append(torch.stack([Tri_coeff_imag_inv[0][0], Tri_coeff_imag_inv[0][2]]))
    truth_real.append(Tri_coeff_real_inv[0][1].unsqueeze(0))
    truth_imag.append(Tri_coeff_imag_inv[0][1].unsqueeze(0))

    for i in range(1, len(Tri_coeff_real_inv) - 1):
        start_amps, start_phases_r, start_phases_i = [], [], []
        end_amps, end_phases_r, end_phases_i = [], [], []
        mid_amps, mid_phases_r, mid_phases_i = [], [], []

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

            start_amp = torch.sqrt(start_r ** 2 + start_i ** 2 + 1e-8)
            mid_amp = torch.sqrt(mid_r ** 2 + mid_i ** 2 + 1e-8)
            end_amp = torch.sqrt(end_r ** 2 + end_i ** 2 + 1e-8)

            start_phase = torch.atan2(start_i+ 1e-8, start_r+ 1e-8)
            mid_phase = torch.atan2(mid_i+ 1e-8, mid_r+ 1e-8)
            end_phase = torch.atan2(end_i+ 1e-8, end_r+ 1e-8)

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
        truth_imag.append(torch.stack([torch.zeros_like(amp) for amp in mid_amps] + mid_phases_i))

    return train_real, train_imag, truth_real, truth_imag, highpass_start, highpass_end
