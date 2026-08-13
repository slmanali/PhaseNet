"""Metric definitions shared by real- and complex-valued evaluation."""

import math
import torch
import torch.nn.functional as F

AMP_EPS = 1e-4


def compute_l1(pred, target):
    return torch.mean(torch.abs(pred - target)).item()


def compute_mse(pred, target):
    return torch.mean((pred - target) ** 2).item()


def compute_psnr(pred, target):
    mse = compute_mse(pred, target)
    return float("inf") if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)


def compute_ssim(pred, target, window_size=11, sigma=1.5, data_range=1.0):
    coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device)
    coords -= (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); g /= g.sum()
    kernel = (g.view(1, 1, 1, -1) * g.view(1, 1, -1, 1)).repeat(pred.shape[1], 1, 1, 1)
    conv = lambda x: F.conv2d(x, kernel, padding=window_size // 2, groups=pred.shape[1])
    mu1, mu2 = conv(pred), conv(target)
    s1 = conv(pred * pred) - mu1 ** 2
    s2 = conv(target * target) - mu2 ** 2
    s12 = conv(pred * target) - mu1 * mu2
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    value = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 ** 2 + mu2 ** 2 + c1) * (s1 + s2 + c2))
    return value.mean().item()


def compute_lpips(pred, target, loss_fn):
    """LPIPS with its required [-1, 1] input convention."""
    pred_lpips = 2 * pred - 1
    truth_lpips = 2 * target - 1
    return loss_fn(pred_lpips, truth_lpips).mean().item()


def compute_real_pce(truth_coeff, pred_coeff, amp_eps=AMP_EPS):
    total, count = 0.0, 0
    for truth, pred in zip(truth_coeff[1:], pred_coeff[1:]):
        n_orient = truth.shape[1] // 2
        truth_amp = truth[:, :n_orient]
        truth_phase = truth[:, n_orient:]
        pred_phase = pred[:, n_orient:]
        phase_diff = torch.atan2(torch.sin(pred_phase - truth_phase),
                                 torch.cos(pred_phase - truth_phase)).abs()
        mask = truth_amp > amp_eps
        total += phase_diff[mask].sum().item(); count += mask.sum().item()
    return total / count if count else 0.0


def compute_complex_pce(truth_real, truth_imag, pred_real, pred_imag, amp_eps=AMP_EPS):
    total, count = 0.0, 0
    for tr, ti, pr, pi in zip(truth_real[1:], truth_imag[1:], pred_real[1:], pred_imag[1:]):
        n_orient = tr.shape[1] // 2
        truth_amp = torch.sqrt(tr[:, :n_orient] ** 2 + ti[:, :n_orient] ** 2)
        truth_angle = torch.atan2(ti[:, n_orient:], tr[:, n_orient:])
        pred_angle = torch.atan2(pi[:, n_orient:], pr[:, n_orient:])
        diff = torch.atan2(torch.sin(pred_angle - truth_angle), torch.cos(pred_angle - truth_angle)).abs()
        mask = truth_amp > amp_eps
        total += diff[mask].sum().item(); count += mask.sum().item()
    return total / count if count else 0.0
