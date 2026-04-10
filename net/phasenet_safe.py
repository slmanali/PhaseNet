from torch import nn
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
import torchvision

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import os
import os.path
import sys

pi = np.pi


def normalize(input_):
    '''
    Normalize phases by π, but preserve amplitude scale.
    Residual inputs (2 channels) are normalized by per-channel max.
    '''
    temp = input_.clone()
    if input_.shape[1] == 2:
        for i in range(temp.shape[0]):
            for j in range(2):
                max_val = temp[i, j, :, :].abs().max()
                if max_val > 1e-8:
                    temp[i, j, :, :] = temp[i, j, :, :] / max_val
    else:
        bands = int(input_.shape[1] / 4)
        for i in range(bands):
            temp[:, i + bands, :, :] = temp[:, i + bands, :, :] / pi
            temp[:, i + 3 * bands, :, :] = temp[:, i + 3 * bands, :, :] / pi
    return temp


def AmpPhase(complex_input):
    real = torch.unbind(complex_input, -1)[0]
    imag = torch.unbind(complex_input, -1)[1]
    return torch.sqrt(real * real + imag * imag + 1e-8), torch.atan2(imag, real)


def input_convert(Tri_coeff):
    Tri_coeff_inv = Tri_coeff[::-1]
    train = []
    truth = []
    amp_scales = []

    train.append(torch.stack([Tri_coeff_inv[0][0], Tri_coeff_inv[0][2]]))
    truth.append(Tri_coeff_inv[0][1].unsqueeze(0))
    amp_scales.append(None)

    for i in range(1, len(Tri_coeff_inv) - 1):
        AP = [AmpPhase(item) for item in Tri_coeff_inv[i]]
        amp_max = torch.stack([
            torch.max(AP[0][0][0, :, :], AP[0][0][1, :, :]),
            torch.max(AP[1][0][0, :, :], AP[1][0][1, :, :]),
            torch.max(AP[2][0][0, :, :], AP[2][0][1, :, :]),
            torch.max(AP[3][0][0, :, :], AP[3][0][1, :, :]),
        ])
        amp_scales.append(amp_max)

        train.append(torch.stack([
            AP[0][0][0, :, :], AP[1][0][0, :, :], AP[2][0][0, :, :], AP[3][0][0, :, :],
            AP[0][1][0, :, :], AP[1][1][0, :, :], AP[2][1][0, :, :], AP[3][1][0, :, :],
            AP[0][0][2, :, :], AP[1][0][2, :, :], AP[2][0][2, :, :], AP[3][0][2, :, :],
            AP[0][1][2, :, :], AP[1][1][2, :, :], AP[2][1][2, :, :], AP[3][1][2, :, :],
        ]))

        truth.append(torch.stack([
            AP[0][0][1, :, :], AP[1][0][1, :, :], AP[2][0][1, :, :], AP[3][0][1, :, :],
            AP[0][1][1, :, :], AP[1][1][1, :, :], AP[2][1][1, :, :], AP[3][1][1, :, :],
        ]))

    return train, truth, amp_scales



def get_input(batch_coeff_list):
    res = [input_convert(Tri_coeff) for Tri_coeff in batch_coeff_list]
    train = []
    truth = []
    amp_scales_batch = []

    for i in range(len(res[0][0])):
        train.append(torch.stack([tt[0][i] for tt in res]).float())
        truth.append(torch.stack([tt[1][i] for tt in res]).float())

    for level_idx in range(1, len(res[0][2])):
        if res[0][2][level_idx] is None:
            amp_scales_batch.append(None)
        else:
            amp_scales_batch.append(torch.stack([tt[2][level_idx] for tt in res]))

    return train, truth, amp_scales_batch



def output_convert(pred_coeff, amp_scales=None):
    coeff = []
    bands_num = int(pred_coeff[1].shape[1] / 2)
    device = pred_coeff[0].device

    coeff.append(pred_coeff[0].squeeze(1))

    for i in range(1, len(pred_coeff)):
        band = []
        for j in range(bands_num):
            amp = pred_coeff[i][:, j, :, :]
            phase = pred_coeff[i][:, j + bands_num, :, :]

            scale_idx = i - 1
            if amp_scales is not None and scale_idx < len(amp_scales) and amp_scales[scale_idx] is not None:
                scale = amp_scales[scale_idx][:, j, :, :].to(device)
                amp = amp * scale

            real = amp * torch.cos(phase)
            imag = amp * torch.sin(phase)
            band.append(torch.stack([real, imag], -1))

        coeff.insert(0, band)

    hi0_shape = (pred_coeff[-1].shape[0], pred_coeff[-1].shape[2], pred_coeff[-1].shape[3])
    coeff.insert(0, torch.zeros(size=hi0_shape, device=device))
    return coeff



def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class Triplets(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes, self.class_to_idx = self._find_classes()
        self.sample = self._make_sample()
        self.transform = transform

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        imgs_list = self.sample[idx]
        sample = [pil_loader(img[0]) for img in imgs_list]
        if self.transform:
            sample = [self.transform(item) for item in sample]
        return {'start': sample[0], 'inter': sample[1], 'end': sample[2], 'class_index': imgs_list[0][1]}

    def _find_classes(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self.root_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _make_sample(self):
        sample = []
        dir = os.path.expanduser(self.root_dir)
        for target in sorted(self.class_to_idx.keys()):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            images = []
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    images.append((path, self.class_to_idx[target]))
            for i in range(len(images) - 2):
                sample.append(images[i:i + 3])
        return sample


class PhaseNetBlock(nn.Module):
    def __init__(self, in_channels=88, out_channels=64, kernel_size=3, padding=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class Pred(nn.Module):
    def __init__(self, in_channels=64, out_channels=8, kernel_size=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return torch.tanh(self.conv(x))


class PhaseNet(nn.Module):
    '''
    Safe real-valued baseline close to the original PhaseNet logic:
    - residual: alpha * start + (1-alpha) * end
    - amplitude: beta * start_amp + (1-beta) * end_amp
    - phase: predicted by network
    No learned amplitude delta branch.
    '''
    def __init__(self, feature_dim=64):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.feature_dim = feature_dim

        self.layer = nn.ModuleList()
        self.pred = nn.ModuleList()

        self.layer.append(PhaseNetBlock(2, feature_dim, 1, 0))
        self.pred.append(Pred(feature_dim, 1))

        self.layer.append(PhaseNetBlock(81, feature_dim, 1, 0))
        self.pred.append(Pred(feature_dim, 4))

        self.layer.append(PhaseNetBlock(16 + feature_dim + 4, feature_dim, 1, 0))
        self.pred.append(Pred(feature_dim, 4))

        for _ in range(8):
            self.layer.append(PhaseNetBlock(16 + feature_dim + 4, feature_dim))
            self.pred.append(Pred(feature_dim, 4))

    def forward(self, x):
        feature_map = []
        pred_map = []
        output = []

        x0 = normalize(x[0])
        feature_map.append(self.layer[0](x0))
        pred_map.append(self.pred[0](feature_map[0]))

        base_residual = self.alpha * x[0][:, 0:1, :, :] + (1.0 - self.alpha) * x[0][:, 1:2, :, :]
        output.append(base_residual)

        for i in range(1, len(x)):
            img_shape = (x[i].shape[2], x[i].shape[3])
            prev_feat = F.interpolate(feature_map[i - 1], img_shape, mode='bilinear', align_corners=False)
            prev_pred = F.interpolate(pred_map[i - 1], img_shape, mode='bilinear', align_corners=False)

            xin = torch.cat([normalize(x[i]), prev_feat, prev_pred], dim=1)
            feat = self.layer[i](xin)
            phase_pred = self.pred[i](feat)

            feature_map.append(feat)
            pred_map.append(phase_pred)

            amp = self.beta * x[i][:, 0:4, :, :] + (1.0 - self.beta) * x[i][:, 8:12, :, :]
            phase = pi * phase_pred
            output.append(torch.cat([amp, phase], dim=1))

        return output


class Total_loss(nn.Module):
    def __init__(self, img_weight=1.0, phase_weight=0.1, residual_weight=0.1, grad_weight=0.2, amp_weight=0.0):
        super().__init__()
        self.img_weight = img_weight
        self.phase_weight = phase_weight
        self.residual_weight = residual_weight
        self.grad_weight = grad_weight
        self.amp_weight = amp_weight
        self.last_stats = {}

    @staticmethod
    def gradient_loss(pred, truth):
        dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        dx_gt = torch.abs(truth[:, :, :, :-1] - truth[:, :, :, 1:])
        dy_gt = torch.abs(truth[:, :, :-1, :] - truth[:, :, 1:, :])
        return torch.mean(torch.abs(dx - dx_gt)) + torch.mean(torch.abs(dy - dy_gt))

    def forward(self, truth_coeff, pred_coeff, truth_img, pred_img):
        if truth_img.dim() == 3:
            truth_img = truth_img.unsqueeze(1)
        if pred_img.dim() == 3:
            pred_img = pred_img.unsqueeze(1)

        img_loss = F.l1_loss(pred_img, truth_img)
        grad_loss = self.gradient_loss(pred_img, truth_img)
        residual_loss = F.l1_loss(pred_coeff[0], truth_coeff[0])

        phase_loss = pred_img.new_tensor(0.0)
        amp_loss = pred_img.new_tensor(0.0)
        num_bands = max(len(truth_coeff) - 1, 1)

        for i in range(1, len(truth_coeff)):
            truth_amp = truth_coeff[i][:, :4, :, :]
            truth_phase = truth_coeff[i][:, 4:, :, :]
            pred_amp = pred_coeff[i][:, :4, :, :]
            pred_phase = pred_coeff[i][:, 4:, :, :]

            dphase = truth_phase - pred_phase
            wrapped = torch.atan2(torch.sin(dphase), torch.cos(dphase))
            phase_loss = phase_loss + F.l1_loss(wrapped, torch.zeros_like(wrapped))

            if self.amp_weight > 0.0:
                amp_loss = amp_loss + F.l1_loss(pred_amp, truth_amp)

        phase_loss = phase_loss / num_bands
        if self.amp_weight > 0.0:
            amp_loss = amp_loss / num_bands

        total = (
            self.img_weight * img_loss
            + self.grad_weight * grad_loss
            + self.residual_weight * residual_loss
            + self.phase_weight * phase_loss
            + self.amp_weight * amp_loss
        )

        self.last_stats = {
            'img': float(img_loss.detach().cpu()),
            'grad': float(grad_loss.detach().cpu()),
            'residual': float(residual_loss.detach().cpu()),
            'phase': float(phase_loss.detach().cpu()),
            'amp': float(amp_loss.detach().cpu()),
            'total': float(total.detach().cpu()),
        }
        return total
