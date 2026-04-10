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
    Normalize phases by π, but PRESERVE amplitude scale for reconstruction.
    '''
    temp = input_.clone()
    if input_.shape[1] == 2:  # Residual - normalize by max
        for i in range(temp.shape[0]):
            for j in range(2):
                max_val = temp[i, j, :, :].max()
                if max_val > 1e-8:
                    temp[i, j, :, :] = temp[i, j, :, :] / max_val
    else:  # Band levels - normalize phases, keep amplitude scale
        bands = int(input_.shape[1] / 4)
        # Normalize phases by π
        for i in range(bands):
            temp[:, i+bands, :, :] = temp[:, i+bands, :, :] / pi
            temp[:, i+3*bands, :, :] = temp[:, i+3*bands, :, :] / pi
        # ✓ DO NOT normalize amplitudes - keep original scale for reconstruction
    return temp


def AmpPhase(complex_input):
    '''
    Args: 
        complex_input: type~torch.tensor [N,H,W,2]
    Return:
        (amplitude, phase)
        torch.tensor
    '''
    real = torch.unbind(complex_input, -1)[0]
    imag = torch.unbind(complex_input, -1)[1]
    return torch.sqrt(real*real+imag*imag),torch.atan2(imag,real)


def input_convert(Tri_coeff):
    '''
    Return: (train_list, truth_list, amp_scales_list)
    where amp_scales_list[i] = max amplitude per orientation for level i
          shape: [n_orient, H, W] for band levels
    '''
    Tri_coeff_inv = Tri_coeff[::-1]
    train = []
    truth = []
    amp_scales = []  # ✓ Per-level max amplitudes
    
    # Low level residual
    train.append(torch.stack([Tri_coeff_inv[0][0], Tri_coeff_inv[0][2]]))
    truth.append(Tri_coeff_inv[0][1].unsqueeze(0))
    amp_scales.append(None)  # ✓ Residual has no per-orientation decomposition
    
    # Bands
    for i in range(1, len(Tri_coeff_inv)-1):
        AP = [AmpPhase(item) for item in Tri_coeff_inv[i]]
        
        # ✓ Extract amplitude maxima per orientation
        # AP[j][0] is amplitude tensor [H, W], j indexes 4 orientations
        # We take max across frames (start, inter, end) → [H, W] per orientation
        amp_max = torch.stack([
            torch.max(AP[0][0][0, :, :], AP[0][0][1, :, :]),
            torch.max(AP[1][0][0, :, :], AP[1][0][1, :, :]),
            torch.max(AP[2][0][0, :, :], AP[2][0][1, :, :]),
            torch.max(AP[3][0][0, :, :], AP[3][0][1, :, :]),
        ])  # Shape: [4, H, W]
        amp_scales.append(amp_max)
        
        train.append(torch.stack([
            AP[0][0][0,:,:], AP[1][0][0,:,:],
            AP[2][0][0,:,:], AP[3][0][0,:,:],
            AP[0][1][0,:,:], AP[1][1][0,:,:],
            AP[2][1][0,:,:], AP[3][1][0,:,:],
            AP[0][0][2,:,:], AP[1][0][2,:,:],
            AP[2][0][2,:,:], AP[3][0][2,:,:],
            AP[0][1][2,:,:], AP[1][1][2,:,:],
            AP[2][1][2,:,:], AP[3][1][2,:,:]
        ]))
        
        truth.append(torch.stack([
            AP[0][0][1,:,:], AP[1][0][1,:,:],
            AP[2][0][1,:,:], AP[3][0][1,:,:],
            AP[0][1][1,:,:], AP[1][1][1,:,:],
            AP[2][1][1,:,:], AP[3][1][1,:,:]
        ]))

    return train, truth, amp_scales  # ✓ Return scales


def get_input(batch_coeff_list):
    res = [input_convert(Tri_coeff) for Tri_coeff in batch_coeff_list]
    train = []
    truth = []
    amp_scales_batch = []  # ✓ Stack scales per level
    
    for i in range(len(res[0][0])):
        train.append(torch.stack([tt[0][i] for tt in res]).float())
        truth.append(torch.stack([tt[1][i] for tt in res]).float())
    
    # ✓ Stack amplitude scales: each entry is [B, n_orient, H, W]
    # Skip level 0 (residual has no per-orientation scales)
    for level_idx in range(1, len(res[0][2])):
        if res[0][2][level_idx] is None:
            amp_scales_batch.append(None)
        else:
            # Stack across batch: list of [4, H, W] → [B, 4, H, W]
            stacked = torch.stack([tt[2][level_idx] for tt in res])
            amp_scales_batch.append(stacked)
    
    return train, truth, amp_scales_batch


def output_convert(pred_coeff, amp_scales=None):
    """
    Convert model predictions (normalized) back to pyramid coefficient format.
    
    Args:
        pred_coeff: list of model outputs (normalized phases + normalized amps)
        amp_scales: list of original amplitude ranges for denormalization
                    amp_scales[0] = scales for pred_coeff[1] (first band level)
                    amp_scales[1] = scales for pred_coeff[2] (second band level)
                    etc.
    """
    coeff = []
    batch_size = pred_coeff[0].shape[0]
    bands_num = int(pred_coeff[1].shape[1] / 2)
    device = pred_coeff[0].device

    # Residual (level 0)
    coeff.append(pred_coeff[0].squeeze(1))
    
    # Band levels
    for i in range(1, len(pred_coeff)):
        band = []
        for j in range(bands_num):
            amp = pred_coeff[i][:, j, :, :]
            phase = pred_coeff[i][:, j + bands_num, :, :]
            
            # ✓ FIX: Index into amp_scales correctly
            # amp_scales[0] corresponds to pred_coeff[1], so offset by -1
            scale_idx = i - 1
            if amp_scales is not None and scale_idx < len(amp_scales) and amp_scales[scale_idx] is not None:
                scale = amp_scales[scale_idx][:, j, :, :].to(device)  # ✓ Index [batch, orient, H, W]
                amp = amp * scale
            
            # Reconstruct complex coefficient
            real = amp * torch.cos(phase)
            imag = amp * torch.sin(phase)
            band.append(torch.stack([real, imag], -1))
        
        coeff.insert(0, band)
    
    # High-pass residual
    hi0_shape = (pred_coeff[-1].shape[0], pred_coeff[-1].shape[2], pred_coeff[-1].shape[3])
    coeff.insert(0, torch.zeros(size=hi0_shape, device=device))
    
    return coeff

# def get_phase(complex_input):
#     '''
#     Args: 
#         complex_input: type~np.array
#                         dtype=np.complex
#     Return:
#         torch.tensor
#     '''
#     return torch.from_numpy(np.arctan2(complex_input.imag, complex_input.real))


# def get_amplitude(complex_input):
#     '''
#     Args: 
#         complex_input: type~np.array
#                        dtype=np.complex
#     Return:
#         torch.tensor
#     '''
#     return torch.from_numpy(np.abs(complex_input))


# def convert(coeff_start, coeff_inter, coeff_end):
#     '''
#     train: coeff_start and coeff_end

#     truth: coeff_inter

#     '''
#     coeff_start_inv = coeff_start[::-1]
#     coeff_inter_inv = coeff_inter[::-1]
#     coeff_end_inv = coeff_end[::-1]
#     train = []
#     truth = []

#     # low level residual
#     train.append(torch.stack([torch.from_numpy(np.fft.ifft2(np.fft.ifftshift(coeff_start_inv[0])).real),
#                               torch.from_numpy(np.fft.ifft2(np.fft.ifftshift(coeff_end_inv[0])).real)]))
#     truth.append(torch.unsqueeze(torch.from_numpy(
#         np.fft.ifft2(np.fft.ifftshift(coeff_inter_inv[0])).real), 0))

#     # bands
#     for i in range(1, len(coeff_start)-1):
#         train.append(torch.stack([get_amplitude(coeff_start_inv[i][0]), get_amplitude(coeff_start_inv[i][1]),
#                                   get_amplitude(coeff_start_inv[i][2]), get_amplitude(coeff_start_inv[i][3]),
#                                   get_phase(coeff_start_inv[i][0]), get_phase(coeff_start_inv[i][1]),
#                                   get_phase(coeff_start_inv[i][2]), get_phase(coeff_start_inv[i][3]),
#                                   get_amplitude(coeff_end_inv[i][0]), get_amplitude(coeff_end_inv[i][1]),
#                                   get_amplitude(coeff_end_inv[i][2]), get_amplitude(coeff_end_inv[i][3]),
#                                   get_phase(coeff_end_inv[i][0]), get_phase(coeff_end_inv[i][1]),
#                                   get_phase(coeff_end_inv[i][2]), get_phase(coeff_end_inv[i][3])]))
#         truth.append(torch.stack([get_amplitude(coeff_inter_inv[i][0]), get_amplitude(coeff_inter_inv[i][1]),
#                                   get_amplitude(coeff_inter_inv[i][2]), get_amplitude(coeff_inter_inv[i][3]),
#                                   get_phase(coeff_inter_inv[i][0]), get_phase(coeff_inter_inv[i][1]),
#                                   get_phase(coeff_inter_inv[i][2]), get_phase(coeff_inter_inv[i][3])]))

#     return train, truth


# def get_input(batch_coeff):
#     res = [convert(item[0], item[1], item[2]) for item in batch_coeff]
#     train = []
#     truth = []
#     for i in range(len(res[0][0])):
#         train.append(torch.stack([tt[0][i] for tt in res]).float())
#         truth.append(torch.stack([tt[1][i] for tt in res]).float())
#     return train, truth


def pil_loader(path):
    # copy from torchvision.datasets.ImageFolder source code
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def show_Triplets_batch(Triplets_batch):
    '''
    Added by Lijie

    Show Triplets_batch

    Args:
        Triplets_batch ({}): a image batch from Triplets

    Return: 
        image batch (numpy)
    '''
    batch_size = len(Triplets_batch['start'])
    img_list = []
    for i in range(batch_size):
        img_list.append(Triplets_batch['start'][i])
        img_list.append(Triplets_batch['inter'][i])
        img_list.append(Triplets_batch['end'][i])
    im_batch = torchvision.utils.make_grid(img_list, nrow=3).numpy()
    im_batch = np.transpose(im_batch, (1, 2, 0))
    plt.imshow(im_batch)
    plt.show()
    return im_batch


class Triplets(Dataset):
    '''
    Added by Lijie

    Generate dataset as it's showed below:

    {'start':[class1_first, class2_first, ...],
     'inter':[class1_inter, class2_inter, ...],
     'end':  [class1_end,class2_end,...],
     'class_index':[class1_index, class2_index, ...]}

    the len of each value in the Dataset dict : batch_size
    '''

    def __init__(self, root_dir, transform=None):
        """
        dir:
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

        Args:
            root_dir (string): Directory with all the images.

        """
        self.root_dir = root_dir
        self.classes, self.class_to_idx = self._find_classes()
        self.sample = self._make_sample()
        self.transform = transform

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        # [(path,index),(path,index),(path,index)]
        imgs_list = self.sample[idx]
        pil_loader(imgs_list[0][0])
        sample = [pil_loader(img[0]) for img in imgs_list]
        if self.transform:
            sample = [self.transform(item) for item in sample]
        sample = {'start': sample[0], 'inter': sample[1],
                  'end': sample[2], 'class_index': imgs_list[0][1]}
        return sample

    def _find_classes(self):
        """
        Copy from torchvision.datasets.ImageFolder source code

        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(self.root_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _make_sample(self):
        # a modified version from torchvision.datasets.ImageFolder source code
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
                    item = (path, self.class_to_idx[target])
                    images.append(item)
            for i in range(len(images)-2):
                sample.append(images[i:i+3])
        return sample


# def Resize(input, New_Size):
#     '''
#     Added by Lijie

#     Resize torch.Tensor 's H and W

#     Args:
#         input: torch.Tensor (N,C,H,W)  N:batch_size C:channels
#         New_size: (H_new,W_new)

#     Return torch.Tensor (N,C,H_new,W_new)
#     '''
#     assert isinstance(input, torch.Tensor)
#     input_np = input.numpy()
#     input_size = input_np.shape
#     print(input_size)
#     output = np.empty(
#         shape=(input_size[0], input_size[1], New_Size[0], New_Size[1]))
#     print(output.shape)
#     for i in range(input_size[0]):
#         for j in range(input_size[1]):
#             temp = Image.fromarray(input_np[i, j, :, :])
#             temp = temp.resize(New_Size, Image.BILINEAR)
#             output[i, j, :, :] = np.array(temp)
#     return torch.from_numpy(output)


class PhaseNetBlock(nn.Module):
    """Basic real-valued PhaseNet block."""

    def __init__(self, in_channels=88, out_channels=64, kernel_size=3, padding=1):
        super(PhaseNetBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class Pred(nn.Module):
    """Prediction head for residual or band coefficients."""

    def __init__(self, in_channels=64, out_channels=8, kernel_size=1):
        super(Pred, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return torch.tanh(self.conv(x))


class PhaseNet(nn.Module):
    """
    Real-valued PhaseNet with configurable feature width so it can be trained
    under settings closer to ComplexPhaseNet.
    """

    def __init__(self, feature_dim=64):
        super(PhaseNet, self).__init__()

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.feature_dim = feature_dim

        self.layer = nn.ModuleList()
        self.pred = nn.ModuleList()

        self.layer.append(PhaseNetBlock(2, feature_dim, 1, 0))
        self.pred.append(Pred(feature_dim, 1))

        self.layer.append(PhaseNetBlock(81, feature_dim, 1, 0))
        self.pred.append(Pred(feature_dim, 8))

        self.layer.append(PhaseNetBlock(88, feature_dim, 1, 0))
        self.pred.append(Pred(feature_dim, 8))

        for _ in range(8):
            self.layer.append(PhaseNetBlock(88, feature_dim))
            self.pred.append(Pred(feature_dim, 8))

    def forward(self, x):
        feature_map = []
        pred_map = []
        output = []

        feature_0 = self.layer[0](normalize(x[0]))
        pred_0 = self.pred[0](feature_0)
        feature_map.append(feature_0)
        pred_map.append(pred_0)

        residual = self.alpha * x[0][:, 0, :, :] + (1 - self.alpha) * x[0][:, 1, :, :]
        residual = residual + pred_0.squeeze(1)
        output.append(residual.unsqueeze(1))

        for i in range(1, len(x)):
            img_shape = (x[i].shape[2], x[i].shape[3])

            feat_up = F.interpolate(feature_map[i - 1], img_shape, mode='bilinear', align_corners=False)
            pred_up = F.interpolate(pred_map[i - 1], img_shape, mode='bilinear', align_corners=False)

            block_input = torch.cat([normalize(x[i]), feat_up, pred_up], dim=1)
            feat_i = self.layer[i](block_input)
            pred_i = self.pred[i](feat_i)

            feature_map.append(feat_i)
            pred_map.append(pred_i)

            base_amp = self.beta * x[i][:, 0:4, :, :] + (1 - self.beta) * x[i][:, 8:12, :, :]
            amp_delta = 0.5 * pred_i[:, 0:4, :, :]
            amp = torch.clamp(base_amp + amp_delta, min=1e-4)

            base_phase = 0.5 * (x[i][:, 4:8, :, :] + x[i][:, 12:16, :, :])
            delta_phase = np.pi * pred_i[:, 4:8, :, :]
            phase = base_phase + delta_phase

            output.append(torch.cat([amp, phase], dim=1))

        return output


class Total_loss(nn.Module):
    """
    Weighted real-valued PhaseNet loss, aligned with the complex training style.

    Components:
      - image L1 reconstruction loss
      - image gradient loss
      - residual coefficient loss
      - amplitude coefficient loss
      - wrapped phase loss
    """

    def __init__(
        self,
        img_weight=3.0,
        residual_weight=2.0,
        phase_weight=0.2,
        amp_weight=1.2,
        grad_weight=2.0,
    ):
        super(Total_loss, self).__init__()
        self.img_weight = img_weight
        self.residual_weight = residual_weight
        self.phase_weight = phase_weight
        self.amp_weight = amp_weight
        self.grad_weight = grad_weight
        self.last_stats = None

    @staticmethod
    def gradient_loss(x, y):
        dx = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        dy = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])

        dx_gt = torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
        dy_gt = torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])

        return torch.abs(dx - dx_gt).mean() + torch.abs(dy - dy_gt).mean()

    def forward(self, truth_coeff, pred_coeff, truth_img, pred_img):
        if truth_img.dim() == 3:
            truth_img = truth_img.unsqueeze(1)
        if pred_img.dim() == 3:
            pred_img = pred_img.unsqueeze(1)

        img_loss = nn.L1Loss()(pred_img, truth_img)
        grad_loss = self.gradient_loss(pred_img, truth_img)
        residual_loss = nn.L1Loss()(pred_coeff[0], truth_coeff[0])

        phase_loss = 0.0
        amp_loss = 0.0
        num_bands = 0

        for i in range(1, len(truth_coeff)):
            n_orient = truth_coeff[i].shape[1] // 2

            truth_amp = truth_coeff[i][:, :n_orient, :, :]
            pred_amp = pred_coeff[i][:, :n_orient, :, :]
            amp_loss += nn.L1Loss()(pred_amp, truth_amp)

            truth_phase = truth_coeff[i][:, n_orient:, :, :]
            pred_phase = pred_coeff[i][:, n_orient:, :, :]
            dphase = truth_phase - pred_phase
            wrapped = torch.atan2(torch.sin(dphase), torch.cos(dphase))
            phase_loss += torch.abs(wrapped).mean()

            num_bands += 1

        if num_bands > 0:
            phase_loss = phase_loss / num_bands
            amp_loss = amp_loss / num_bands

        total_loss = (
            self.img_weight * img_loss
            + self.grad_weight * grad_loss
            + self.residual_weight * residual_loss
            + self.phase_weight * phase_loss
            + self.amp_weight * amp_loss
        )

        self.last_stats = {
            "img": float(img_loss.detach().cpu()),
            "grad": float(grad_loss.detach().cpu()),
            "residual": float(residual_loss.detach().cpu()),
            "phase": float(phase_loss.detach().cpu() if isinstance(phase_loss, torch.Tensor) else phase_loss),
            "amp": float(amp_loss.detach().cpu() if isinstance(amp_loss, torch.Tensor) else amp_loss),
            "total": float(total_loss.detach().cpu()),
        }

        return total_loss


if __name__ == "__main__":

    model = PhaseNet()
    print(model)
    temp =0
    for i, j in model.named_parameters():
        # temp+=1
        # if temp<=2:
        #     print(i,j)
        print(i)


    # # test input
    # input=[]
    # input.append(torch.autograd.Variable(torch.randn(2, 2, 8, 8)))
    # input.append(torch.autograd.Variable(torch.randn(2, 16, 12, 12)))
    # input.append(torch.autograd.Variable(torch.randn(2, 16, 16, 16)))
    # input.append(torch.autograd.Variable(torch.randn(2, 16, 22, 22)))
    # input.append(torch.autograd.Variable(torch.randn(2, 16, 32, 32)))
    # input.append(torch.autograd.Variable(torch.randn(2, 16, 46, 46)))
    # input.append(torch.autograd.Variable(torch.randn(2, 16, 64, 64)))
    # input.append(torch.autograd.Variable(torch.randn(2, 16, 90, 90)))
    # input.append(torch.autograd.Variable(torch.randn(2, 16, 128, 128)))
    # input.append(torch.autograd.Variable(torch.randn(2, 16, 182, 182)))
    # input.append(torch.autograd.Variable(torch.randn(2, 16, 256, 256)))
    # o = model(input)
