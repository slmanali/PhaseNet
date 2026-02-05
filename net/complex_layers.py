import torch
from torch import nn


def _split_complex(x):
    real, imag = torch.chunk(x, 2, dim=1)
    return real, imag


def _combine_complex(real, imag):
    return torch.cat([real, imag], dim=1)


class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=True):
        super().__init__()
        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)
        self.imag_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

    def forward(self, x):
        real, imag = _split_complex(x)
        real_out = self.real_conv(real) - self.imag_conv(imag)
        imag_out = self.real_conv(imag) + self.imag_conv(real)
        return _combine_complex(real_out, imag_out)


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.real_bn = nn.BatchNorm2d(num_features)
        self.imag_bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        real, imag = _split_complex(x)
        return _combine_complex(self.real_bn(real), self.imag_bn(imag))


class ComplexLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2, inplace=True):
        super().__init__()
        self.act = nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)

    def forward(self, x):
        real, imag = _split_complex(x)
        return _combine_complex(self.act(real), self.act(imag))


class ComplexSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
