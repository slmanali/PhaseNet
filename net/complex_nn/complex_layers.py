"""
Complex-valued neural network layers.

This module implements complex-valued convolution, batch normalization,
and activation functions for deep learning with complex numbers.

References:
- Trabelsi et al. (2018): Deep Complex Networks
- Arjovsky et al. (2016): Unitary Evolution Recurrent Neural Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .complex_utils import (
    complex_multiply, 
    modReLU, 
    CReLU, 
    zReLU,
    apply_complex_activation
)


class ComplexConv2d(nn.Module):
    """
    Complex-valued 2D convolution layer.
    
    Implements convolution for complex-valued inputs and weights:
    (A + Bi) * (W_r + W_i*i) = (A*W_r - B*W_i) + (A*W_i + B*W_r)i
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): Zero-padding added to both sides
        dilation (int or tuple): Spacing between kernel elements
        groups (int): Number of blocked connections
        bias (bool): If True, adds a learnable bias
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Create separate real and imaginary weight matrices
        self.conv_real = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias=False
        )
        
        self.conv_imag = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, groups, bias=False
        )
        
        # Bias is complex-valued
        if bias:
            self.bias_real = nn.Parameter(torch.Tensor(out_channels))
            self.bias_imag = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using He initialization."""
        nn.init.kaiming_uniform_(self.conv_real.weight, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.conv_imag.weight, a=np.sqrt(5))
        
        if self.bias_real is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv_real.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias_real, -bound, bound)
            nn.init.uniform_(self.bias_imag, -bound, bound)
    
    def forward(self, input_real, input_imag):
        """
        Forward pass for complex convolution.
        
        Args:
            input_real (torch.Tensor): Real part of input [N, C_in, H, W]
            input_imag (torch.Tensor): Imaginary part of input [N, C_in, H, W]
        
        Returns:
            tuple: (output_real, output_imag)
        """
        # Complex multiplication: (A + Bi) * (W_r + W_i*i)
        # Real part: A*W_r - B*W_i
        output_real = self.conv_real(input_real) - self.conv_imag(input_imag)
        
        # Imaginary part: A*W_i + B*W_r
        output_imag = self.conv_real(input_imag) + self.conv_imag(input_real)
        
        # Add complex bias
        if self.bias_real is not None:
            output_real = output_real + self.bias_real.view(1, -1, 1, 1)
            output_imag = output_imag + self.bias_imag.view(1, -1, 1, 1)
        
        return output_real, output_imag


class ComplexBatchNorm2d(nn.Module):
    """
    Complex-valued 2D Batch Normalization.
    
    Normalizes both real and imaginary parts using complex-valued statistics.
    Based on "Deep Complex Networks" (Trabelsi et al., 2018).
    
    Args:
        num_features (int): Number of features (channels)
        eps (float): Small value for numerical stability
        momentum (float): Momentum for running statistics
        affine (bool): If True, learnable affine parameters
        track_running_stats (bool): If True, track running mean and variance
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(ComplexBatchNorm2d, self).__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            # Learnable parameters for affine transformation
            self.weight_real = nn.Parameter(torch.Tensor(num_features))
            self.weight_imag = nn.Parameter(torch.Tensor(num_features))
            self.bias_real = nn.Parameter(torch.Tensor(num_features))
            self.bias_imag = nn.Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight_real', None)
            self.register_parameter('weight_imag', None)
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
        
        if self.track_running_stats:
            self.register_buffer('running_mean_real', torch.zeros(num_features))
            self.register_buffer('running_mean_imag', torch.zeros(num_features))
            self.register_buffer('running_var_rr', torch.ones(num_features))
            self.register_buffer('running_var_ii', torch.ones(num_features))
            self.register_buffer('running_var_ri', torch.zeros(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean_real', None)
            self.register_parameter('running_mean_imag', None)
            self.register_parameter('running_var_rr', None)
            self.register_parameter('running_var_ii', None)
            self.register_parameter('running_var_ri', None)
            self.register_parameter('num_batches_tracked', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        if self.track_running_stats:
            self.running_mean_real.zero_()
            self.running_mean_imag.zero_()
            self.running_var_rr.fill_(1)
            self.running_var_ii.fill_(1)
            self.running_var_ri.zero_()
            self.num_batches_tracked.zero_()
        
        if self.affine:
            nn.init.constant_(self.weight_real, 1.0)
            nn.init.constant_(self.weight_imag, 0.0)
            nn.init.constant_(self.bias_real, 0.0)
            nn.init.constant_(self.bias_imag, 0.0)
    
    def forward(self, input_real, input_imag):
        """
        Forward pass for complex batch normalization.
        
        Args:
            input_real (torch.Tensor): Real part [N, C, H, W]
            input_imag (torch.Tensor): Imaginary part [N, C, H, W]
        
        Returns:
            tuple: (normalized_real, normalized_imag)
        """
        exponential_average_factor = 0.0
        
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum
        
        # Calculate mean
        # Shape: [N, C, H, W] -> [C]
        if self.training:
            mean_real = input_real.mean([0, 2, 3])
            mean_imag = input_imag.mean([0, 2, 3])
        else:
            mean_real = self.running_mean_real
            mean_imag = self.running_mean_imag
        
        # Center the data
        input_real_centered = input_real - mean_real.view(1, -1, 1, 1)
        input_imag_centered = input_imag - mean_imag.view(1, -1, 1, 1)
        
        # Calculate covariance matrix components
        if self.training:
            var_rr = (input_real_centered ** 2).mean([0, 2, 3])
            var_ii = (input_imag_centered ** 2).mean([0, 2, 3])
            var_ri = (input_real_centered * input_imag_centered).mean([0, 2, 3])
        else:
            var_rr = self.running_var_rr
            var_ii = self.running_var_ii
            var_ri = self.running_var_ri
        
        # Update running statistics
        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_mean_real = (1 - exponential_average_factor) * self.running_mean_real + \
                                        exponential_average_factor * mean_real
                self.running_mean_imag = (1 - exponential_average_factor) * self.running_mean_imag + \
                                        exponential_average_factor * mean_imag
                self.running_var_rr = (1 - exponential_average_factor) * self.running_var_rr + \
                                     exponential_average_factor * var_rr
                self.running_var_ii = (1 - exponential_average_factor) * self.running_var_ii + \
                                     exponential_average_factor * var_ii
                self.running_var_ri = (1 - exponential_average_factor) * self.running_var_ri + \
                                     exponential_average_factor * var_ri
        
        # Compute the inverse square root of the covariance matrix
        # Using Cholesky decomposition for numerical stability
        tau = var_rr + var_ii + self.eps
        delta = (var_rr * var_ii - var_ri ** 2) + self.eps
        
        s = torch.sqrt(delta)
        t = torch.sqrt(tau + 2 * s)
        
        inverse_st = 1.0 / (s * t)
        Wrr = (var_ii + s) * inverse_st
        Wii = (var_rr + s) * inverse_st
        Wri = -var_ri * inverse_st
        
        # Normalize
        output_real = Wrr.view(1, -1, 1, 1) * input_real_centered + \
                     Wri.view(1, -1, 1, 1) * input_imag_centered
        output_imag = Wri.view(1, -1, 1, 1) * input_real_centered + \
                     Wii.view(1, -1, 1, 1) * input_imag_centered
        
        # Apply affine transformation
        if self.affine:
            output_real_scaled = self.weight_real.view(1, -1, 1, 1) * output_real - \
                               self.weight_imag.view(1, -1, 1, 1) * output_imag + \
                               self.bias_real.view(1, -1, 1, 1)
            output_imag_scaled = self.weight_real.view(1, -1, 1, 1) * output_imag + \
                               self.weight_imag.view(1, -1, 1, 1) * output_real + \
                               self.bias_imag.view(1, -1, 1, 1)
            return output_real_scaled, output_imag_scaled
        
        return output_real, output_imag


class ComplexReLU(nn.Module):
    """Complex ReLU activation: applies ReLU independently to real and imaginary parts."""
    
    def __init__(self):
        super(ComplexReLU, self).__init__()
    
    def forward(self, input_real, input_imag):
        return CReLU(input_real, input_imag)


class ComplexLeakyReLU(nn.Module):
    """Complex Leaky ReLU activation."""
    
    def __init__(self, negative_slope=0.01):
        super(ComplexLeakyReLU, self).__init__()
        self.negative_slope = negative_slope
    
    def forward(self, input_real, input_imag):
        return (F.leaky_relu(input_real, negative_slope=self.negative_slope),
                F.leaky_relu(input_imag, negative_slope=self.negative_slope))


class ComplexModReLU(nn.Module):
    """
    Modified ReLU for complex numbers (Arjovsky et al., 2016).
    
    modReLU(z) = ReLU(|z| + b) * (z / |z|)
    """
    
    def __init__(self, num_features):
        super(ComplexModReLU, self).__init__()
        self.bias = nn.Parameter(torch.Tensor(num_features))
        nn.init.constant_(self.bias, 0.0)
    
    def forward(self, input_real, input_imag):
        return modReLU(input_real, input_imag, self.bias.view(1, -1, 1, 1))


class ComplexTanh(nn.Module):
    """Complex Tanh activation: applies tanh independently to real and imaginary parts."""
    
    def __init__(self):
        super(ComplexTanh, self).__init__()
    
    def forward(self, input_real, input_imag):
        return torch.tanh(input_real), torch.tanh(input_imag)


class ComplexZReLU(nn.Module):
    """Phase-preserving ReLU for complex numbers."""
    
    def __init__(self):
        super(ComplexZReLU, self).__init__()
    
    def forward(self, input_real, input_imag):
        return zReLU(input_real, input_imag)


class ComplexSequential(nn.Module):
    """
    Sequential container for complex-valued layers.
    
    All layers in this container must accept and return (real, imag) tuples.
    """
    
    def __init__(self, *layers):
        super(ComplexSequential, self).__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, input_real, input_imag):
        """
        Forward pass through all layers.
        
        Args:
            input_real (torch.Tensor): Real part of input
            input_imag (torch.Tensor): Imaginary part of input
        
        Returns:
            tuple: (output_real, output_imag)
        """
        real, imag = input_real, input_imag
        
        for layer in self.layers:
            real, imag = layer(real, imag)
        
        return real, imag
    
    def __len__(self):
        return len(self.layers)
    
    def __getitem__(self, idx):
        return self.layers[idx]
