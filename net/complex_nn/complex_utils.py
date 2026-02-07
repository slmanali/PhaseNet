"""
Complex-valued utility functions for neural networks.

This module provides utility functions for working with complex-valued tensors
in PyTorch, including conversions between Cartesian and polar representations.
"""

import torch
import numpy as np


def complex_magnitude(real, imag):
    """
    Compute magnitude of complex number.
    
    Args:
        real (torch.Tensor): Real part
        imag (torch.Tensor): Imaginary part
    
    Returns:
        torch.Tensor: Magnitude |z| = sqrt(real^2 + imag^2)
    """
    return torch.sqrt(real**2 + imag**2 + 1e-8)


def complex_phase(real, imag):
    """
    Compute phase (angle) of complex number.
    
    Args:
        real (torch.Tensor): Real part
        imag (torch.Tensor): Imaginary part
    
    Returns:
        torch.Tensor: Phase angle = atan2(imag, real)
    """
    return torch.atan2(imag, real)


def complex_multiply(r1, i1, r2, i2):
    """
    Multiply two complex numbers: (r1 + i1*j) * (r2 + i2*j)
    
    Formula: (r1*r2 - i1*i2) + (r1*i2 + i1*r2)j
    
    Args:
        r1, i1: Real and imaginary parts of first complex number
        r2, i2: Real and imaginary parts of second complex number
    
    Returns:
        tuple: (real_part, imag_part) of the product
    """
    real_part = r1 * r2 - i1 * i2
    imag_part = r1 * i2 + i1 * r2
    return real_part, imag_part


def complex_conjugate(real, imag):
    """
    Compute complex conjugate: z* = real - imag*j
    
    Args:
        real (torch.Tensor): Real part
        imag (torch.Tensor): Imaginary part
    
    Returns:
        tuple: (real, -imag)
    """
    return real, -imag


def cart2polar(real, imag):
    """
    Convert Cartesian coordinates to polar coordinates.
    
    Args:
        real (torch.Tensor): Real part (x-coordinate)
        imag (torch.Tensor): Imaginary part (y-coordinate)
    
    Returns:
        tuple: (magnitude, phase)
    """
    magnitude = complex_magnitude(real, imag)
    phase = complex_phase(real, imag)
    return magnitude, phase


def polar2cart(magnitude, phase):
    """
    Convert polar coordinates to Cartesian coordinates.
    
    Args:
        magnitude (torch.Tensor): Magnitude (radius)
        phase (torch.Tensor): Phase (angle in radians)
    
    Returns:
        tuple: (real, imag)
    """
    real = magnitude * torch.cos(phase)
    imag = magnitude * torch.sin(phase)
    return real, imag


def complex_normalize(real, imag, eps=1e-8):
    """
    Normalize complex number to unit magnitude.
    
    Args:
        real (torch.Tensor): Real part
        imag (torch.Tensor): Imaginary part
        eps (float): Small value to avoid division by zero
    
    Returns:
        tuple: (normalized_real, normalized_imag)
    """
    mag = complex_magnitude(real, imag)
    return real / (mag + eps), imag / (mag + eps)


def apply_complex_activation(real, imag, activation_fn):
    """
    Apply a real-valued activation function to both components independently.
    
    Args:
        real (torch.Tensor): Real part
        imag (torch.Tensor): Imaginary part
        activation_fn: Activation function to apply
    
    Returns:
        tuple: (activated_real, activated_imag)
    """
    return activation_fn(real), activation_fn(imag)


def modReLU(real, imag, bias=None):
    """
    Modified ReLU for complex numbers (Arjovsky et al., 2016).
    
    modReLU(z) = ReLU(|z| + b) * (z / |z|)
    
    Args:
        real (torch.Tensor): Real part
        imag (torch.Tensor): Imaginary part
        bias (torch.Tensor or None): Learnable bias parameter
    
    Returns:
        tuple: (activated_real, activated_imag)
    """
    magnitude = complex_magnitude(real, imag)
    
    if bias is not None:
        magnitude = magnitude + bias
    
    # Apply ReLU to magnitude
    activated_magnitude = torch.relu(magnitude)
    
    # Normalize and rescale
    norm_real, norm_imag = complex_normalize(real, imag)
    
    return activated_magnitude * norm_real, activated_magnitude * norm_imag


def CReLU(real, imag):
    """
    Complex ReLU: Apply ReLU to both real and imaginary parts independently.
    
    Args:
        real (torch.Tensor): Real part
        imag (torch.Tensor): Imaginary part
    
    Returns:
        tuple: (ReLU(real), ReLU(imag))
    """
    return torch.relu(real), torch.relu(imag)


def zReLU(real, imag):
    """
    Phase-preserving ReLU for complex numbers.
    
    zReLU(z) = z if Re(z) >= 0 and Im(z) >= 0, else 0
    
    Args:
        real (torch.Tensor): Real part
        imag (torch.Tensor): Imaginary part
    
    Returns:
        tuple: (activated_real, activated_imag)
    """
    mask = (real >= 0) & (imag >= 0)
    return real * mask, imag * mask
