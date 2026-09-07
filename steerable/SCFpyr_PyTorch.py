# Modified by Lijie, scale_factor can be 2**(1/2), 2**(1/4), ...
# Email：glee1018@buaa.edu.cn
# Date Modified： 2019-04-15 11:33

# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-12-04

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import torch

import steerable.math_utils as math_utils
pointOp = math_utils.pointOp

def _fft2_realimag(x):
    return torch.view_as_real(torch.fft.fft2(x, dim=(-2, -1)))


def _ifft2_realimag(x):
    return torch.view_as_real(torch.fft.ifft2(torch.view_as_complex(x.contiguous()), dim=(-2, -1)))

################################################################################
################################################################################


class SCFpyr_PyTorch(object):
    '''
    Modified by Lijie, scale_factor can be 2**(1/2), 2**(1/4), ...

    This is a modified version of buildSFpyr, that constructs a
    complex-valued steerable pyramid  using Hilbert-transform pairs
    of filters. Note that the imaginary parts will *not* be steerable.

    Description of this transform appears in: Portilla & Simoncelli,
    International Journal of Computer Vision, 40(1):49-71, Oct 2000.
    Further information: http://www.cns.nyu.edu/~eero/STEERPYR/

    Modified code from the perceptual repository:
      https://github.com/andreydung/Steerable-filter

    This code looks very similar to the original Matlab code:
      https://github.com/LabForComputationalVision/matlabPyrTools/blob/master/buildSCFpyr.m

    Also looks very similar to the original Python code presented here:
      https://github.com/LabForComputationalVision/pyPyrTools/blob/master/pyPyrTools/SCFpyr.py

    '''

    def __init__(self, height=5, nbands=4, scale_factor=2, device=None):
        self.height = height  # including low-pass and high-pass
        self.nbands = nbands  # number of orientation bands
        self.scale_factor = scale_factor
        self.device = torch.device('cpu') if device is None else device

        # Cache constants
        self.lutsize = 1024
        self.Xcosn = np.pi * np.array(range(-(2*self.lutsize+1), (self.lutsize+2)))/self.lutsize
        self.alpha = (self.Xcosn + np.pi) % (2*np.pi) - np.pi
        self.complex_fact_construct   = np.power(complex(0, -1), self.nbands-1)
        self.complex_fact_reconstruct = np.power(complex(0, 1), self.nbands-1)
        
    ################################################################################
    # Construction of Steerable Pyramid

    def build(self, im_batch, pyr_type):
        ''' Decomposes a batch of images into a complex steerable pyramid. 
        The pyramid typically has ~4 levels and 4-8 orientations. 
        
        Args:
            im_batch (torch.Tensor): Batch of images of shape [N,C,H,W]
            pyr_type (int): 0 for band=ifft(banddft)  
                            1 for band=dft
        Returns:
            pyramid: list containing torch.Tensor objects storing the pyramid
        '''
        
        assert im_batch.device == self.device, 'Devices invalid (pyr = {}, batch = {})'.format(self.device, im_batch.device)
        assert im_batch.dtype == torch.float32, 'Image batch must be torch.float32'
        assert im_batch.dim() == 4, 'Image batch must be of shape [N,C,H,W]'
        assert im_batch.shape[1] == 1, 'Second dimension must be 1 encoding grayscale image'
        assert pyr_type==0 or pyr_type==1, 'pyr_type must be 0 or 1'
        im_batch = im_batch.squeeze(1)  # flatten channels dim
        # After removing the channel axis the tensor layout is [N, H, W].
        # Keep the spatial dimensions in that order so non-square inputs use
        # masks with the same shape as their Fourier transforms.
        height, width = im_batch.shape[1:3]
        
        # Check whether image size is sufficient for number of levels
        if self.height > int(np.floor(np.log2(min(width, height))/np.log2(self.scale_factor)) - 2):
            raise RuntimeError('Cannot build {} levels, image too small.'.format(self.height))
        
        # Prepare a grid
        log_rad, angle = math_utils.prepare_grid(height, width)

        # Radial transition function (a raised cosine in log-frequency):
        Xrcos, Yrcos = math_utils.rcosFn(1, -0.5)
        Yrcos = np.sqrt(Yrcos)

        YIrcos = np.sqrt(1 - Yrcos**2)

        lo0mask = pointOp(log_rad, YIrcos, Xrcos)
        hi0mask = pointOp(log_rad, Yrcos, Xrcos)

        # Note that we expand dims to support broadcasting later
        lo0mask = torch.from_numpy(lo0mask).float()[None,:,:,None].to(self.device)
        hi0mask = torch.from_numpy(hi0mask).float()[None,:,:,None].to(self.device)

        # Fourier transform (2D) and shifting
        batch_dft = _fft2_realimag(im_batch)
        batch_dft = math_utils.batch_fftshift2d(batch_dft)

        # Low-pass
        lo0dft = batch_dft * lo0mask

        # Start recursively building the pyramids
        coeff = self._build_levels(lo0dft, log_rad, angle, Xrcos, Yrcos, self.height-1, np.array((height, width)), pyr_type, self.device)

        # High-pass
        hi0dft = batch_dft * hi0mask
        hi0 = math_utils.batch_ifftshift2d(hi0dft)
        hi0 = _ifft2_realimag(hi0)
        hi0_real = torch.unbind(hi0, -1)[0]
        coeff.insert(0,hi0_real)
        return coeff

    def _build_levels(self, lodft, log_rad, angle, Xrcos, Yrcos, height, img_dims, pyr_type, device):
        # ↑ Add device parameter
        
        if height <= 1:
            lo0 = math_utils.batch_ifftshift2d(lodft)
            lo0 = _ifft2_realimag(lo0)
            lo0_real = torch.unbind(lo0, -1)[0]
            coeff = [lo0_real]
        else:
            Xrcos = Xrcos - np.log2(self.scale_factor)
            ####################################################################
            ####################### Orientation bandpass #######################
            ####################################################################
            himask = pointOp(log_rad, Yrcos, Xrcos)
            himask = torch.from_numpy(himask[None,:,:,None]).float().to(device)  # ✓ Use device param
            order = self.nbands - 1
            const = np.power(2, 2*order) * np.square(math.factorial(order)) / (self.nbands * math.factorial(2*order))
            Ycosn = 2*np.sqrt(const) * np.power(np.cos(self.Xcosn), order) * (np.abs(self.alpha) < np.pi/2)
            orientations = []
            for b in range(self.nbands):
                anglemask = pointOp(angle, Ycosn, self.Xcosn + np.pi*b/self.nbands)
                anglemask = anglemask[None,:,:,None]
                anglemask = torch.from_numpy(anglemask).float().to(device)  # ✓ Use device param
                banddft = lodft * anglemask * himask
                banddft = torch.unbind(banddft, -1)
                banddft_real = self.complex_fact_construct.real*banddft[0] - self.complex_fact_construct.imag*banddft[1]
                banddft_imag = self.complex_fact_construct.real*banddft[1] + self.complex_fact_construct.imag*banddft[0]
                banddft = torch.stack((banddft_real, banddft_imag), -1)
                if pyr_type==0:
                    band = math_utils.batch_ifftshift2d(banddft)
                    band = _ifft2_realimag(band)
                    orientations.append(band)
                else:
                    orientations.append(banddft)
            ####################################################################
            ######################## Subsample lowpass #########################
            ####################################################################
            dims = np.array(lodft.shape[1:3])
            ctr=np.ceil((dims+0.5)/2)
            lodims=np.round(img_dims/(self.scale_factor**(self.height-height)))
            loctr=np.ceil((lodims+0.5)/2)
            lostart=(ctr-loctr).astype(int)
            loend=(lostart+lodims).astype(int)
            log_rad = log_rad[lostart[0]:loend[0],lostart[1]:loend[1]]
            angle = angle[lostart[0]:loend[0],lostart[1]:loend[1]]
            lodft = lodft[:,lostart[0]:loend[0],lostart[1]:loend[1],:]
            YIrcos = np.abs(np.sqrt(1 - Yrcos**2))
            lomask = pointOp(log_rad, YIrcos, Xrcos)
            lomask = torch.from_numpy(lomask[None,:,:,None]).float().to(device)  # ✓ Use device param
            lodft = lomask * lodft
            ####################################################################
            ####################### Recursion next level #######################
            ####################################################################
            # ✓ Pass device to recursive call
            coeff = self._build_levels(lodft, log_rad, angle, Xrcos, Yrcos, height-1, img_dims, pyr_type, device)
            coeff.insert(0, orientations)
        return coeff

    ############################################################################
    ########################### RECONSTRUCTION #################################
    ############################################################################

    def reconstruct(self, coeff, pyr_type):

        if self.nbands != len(coeff[1]):
            raise Exception("Unmatched number of orientations")
        
        device = coeff[0].device
        # The high-pass residual uses the [N, H, W] layout produced by build.
        height, width = coeff[0].shape[1:3]
        log_rad, angle = math_utils.prepare_grid(height, width)

        Xrcos, Yrcos = math_utils.rcosFn(1, -0.5)
        Yrcos  = np.sqrt(Yrcos)
        YIrcos = np.sqrt(np.abs(1 - Yrcos**2))

        lo0mask = pointOp(log_rad, YIrcos, Xrcos)
        hi0mask = pointOp(log_rad, Yrcos, Xrcos)

        # Note that we expand dims to support broadcasting later
        lo0mask = torch.from_numpy(lo0mask).float()[None,:,:,None].to(device)
        hi0mask = torch.from_numpy(hi0mask).float()[None,:,:,None].to(device)

        # Start recursive reconstruction
        lo0dft = self._reconstruct_levels(coeff[1:], log_rad, Xrcos, Yrcos, angle, np.array((height, width)), pyr_type, device)

        hidft = _fft2_realimag(coeff[0])
        hidft = math_utils.batch_fftshift2d(hidft)

        outdft = lo0dft * lo0mask + hidft * hi0mask

        reconstruction = math_utils.batch_ifftshift2d(outdft)
        reconstruction = _ifft2_realimag(reconstruction)
        reconstruction = torch.unbind(reconstruction, -1)[0]  # real

        return reconstruction

    def _reconstruct_levels(self, coeff, log_rad, Xrcos, Yrcos, angle, img_dims, pyr_type, device=None):
            if device is None:
                device = self.device 
                
            if len(coeff) == 1:
                # ✓ FIX: Explicitly move the lowest-level residual to the GPU before FFT
                c0 = coeff[0].to(device)
                dft = _fft2_realimag(c0)
                dft = math_utils.batch_fftshift2d(dft)
                return dft

            Xrcos = Xrcos - np.log2(self.scale_factor)

            ####################################################################
            ####################### Orientation Residue ########################
            ####################################################################

            himask = pointOp(log_rad, Yrcos, Xrcos)
            himask = torch.from_numpy(himask[None,:,:,None]).float().to(device)

            lutsize = 1024
            Xcosn = np.pi * np.array(range(-(2*lutsize+1), (lutsize+2)))/lutsize
            order = self.nbands - 1
            const = np.power(2, 2*order) * np.square(math.factorial(order)) / (self.nbands * math.factorial(2*order))
            Ycosn = np.sqrt(const) * np.power(np.cos(Xcosn), order)

            # Extract reference tensor for shape & dtype
            ref_tensor = coeff[0][0]
            orientdft = torch.zeros(ref_tensor.shape, dtype=ref_tensor.dtype, device=device)
            
            for b in range(self.nbands):
                anglemask = pointOp(angle, Ycosn, Xcosn + np.pi * b/self.nbands)
                anglemask = anglemask[None,:,:,None]
                anglemask = torch.from_numpy(anglemask).float().to(device)

                if pyr_type==0:
                    banddft = _fft2_realimag(coeff[0][b].to(device))
                    banddft = math_utils.batch_fftshift2d(banddft)
                else:
                    banddft = coeff[0][b].to(device)
                    
                # Explicit device alignment before complex multiplication
                anglemask = anglemask.to(device)
                himask = himask.to(device)
                
                banddft = banddft * anglemask * himask
                banddft = torch.unbind(banddft, -1)
                banddft_real = self.complex_fact_reconstruct.real*banddft[0] - self.complex_fact_reconstruct.imag*banddft[1]
                banddft_imag = self.complex_fact_reconstruct.real*banddft[1] + self.complex_fact_reconstruct.imag*banddft[0]
                banddft = torch.stack((banddft_real, banddft_imag), -1)

                orientdft = orientdft + banddft

            ####################################################################
            ########## Lowpass component are upsampled and convoluted ##########
            ####################################################################
            
            dims = np.array(coeff[0][0].shape[1:3])
            ctr=np.ceil((dims+0.5)/2)
            lodims=np.round(img_dims/(self.scale_factor**(self.height-len(coeff))))
            loctr=np.ceil((lodims+0.5)/2)
            lostart=(ctr-loctr).astype(int)
            loend=(lostart+lodims).astype(int)

            nlog_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
            nangle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
            YIrcos = np.sqrt(np.abs(1 - Yrcos**2))
            lomask = pointOp(nlog_rad, YIrcos, Xrcos)
            lomask = torch.from_numpy(lomask[None,:,:,None]).float().to(device)

            # Recursive call for image reconstruction         
            nresdft = self._reconstruct_levels(coeff[1:], nlog_rad, Xrcos, Yrcos, nangle, img_dims, pyr_type, device)
            
            # ✓ FIX: Ensure recursive return matches target device
            nresdft = nresdft.to(device)

            resdft = torch.zeros(ref_tensor.shape, dtype=ref_tensor.dtype, device=device)
            resdft[:,lostart[0]:loend[0], lostart[1]:loend[1],:] = nresdft * lomask

            return resdft + orientdft
