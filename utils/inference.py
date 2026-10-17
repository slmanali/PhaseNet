"""Endpoint-only RGB inference used by evaluators and computational benchmarks."""
import torch

from train_complex_safe_baseline import (extract_complex_coefficients,
                                         output_convert_complex)
from net.phasenet import get_input, output_convert


def _triplet_coefficients(start_coeff, end_coeff):
    """Combine two pyramids into the legacy [start, dummy, end] representation."""
    combined = []
    for start, end in zip(start_coeff, end_coeff):
        if isinstance(start, list):
            combined.append([torch.cat((a, torch.zeros_like(a), b), dim=0)
                             for a, b in zip(start, end)])
        else:
            combined.append(torch.cat((start, torch.zeros_like(start), end), dim=0))
    return combined


def prepare_complex_channel(start, end, pyramid, pyr_type=1):
    combined = _triplet_coefficients(pyramid.build(start, pyr_type=pyr_type),
                                     pyramid.build(end, pyr_type=pyr_type))
    real, imag = extract_complex_coefficients(combined)
    from net.complex_phasenet_safe_baseline import complex_input_convert
    train_r, train_i, _, _, hp_start, hp_end = complex_input_convert(real, imag)
    return ([item.unsqueeze(0) for item in train_r],
            [item.unsqueeze(0) for item in train_i],
            0.5 * (hp_start + hp_end).unsqueeze(0))


def prepare_real_channel(start, end, pyramid, pyr_type=1):
    combined = _triplet_coefficients(pyramid.build(start, pyr_type=pyr_type),
                                     pyramid.build(end, pyr_type=pyr_type))
    inputs, _, scales = get_input([combined])
    return inputs, scales


@torch.inference_mode()
def interpolate_endpoints(model, pyramid, start_rgb, end_rgb, *, complex_model,
                          tile_size=None, pyr_type=1, return_prepared=False):
    """Predict an RGB midpoint from endpoints only; inputs/outputs stay on device."""
    outputs, prepared = [], []
    for channel in range(3):
        start = start_rgb[:, channel:channel + 1]
        end = end_rgb[:, channel:channel + 1]
        if complex_model:
            train_r, train_i, highpass = prepare_complex_channel(start, end, pyramid, pyr_type)
            train_r = [x.to(start_rgb.device).float() for x in train_r]
            train_i = [x.to(start_rgb.device).float() for x in train_i]
            prediction_r, prediction_i = model(train_r, train_i, tile_size=tile_size)
            output_coeff = output_convert_complex(prediction_r, prediction_i, highpass=highpass)
            prepared.append((train_r, train_i, highpass))
        else:
            inputs, scales = prepare_real_channel(start, end, pyramid, pyr_type)
            inputs = [x.to(start_rgb.device).float() for x in inputs]
            output_coeff = output_convert(model(inputs), amp_scales=scales)
            prepared.append((inputs, scales))
        reconstructed = pyramid.reconstruct(output_coeff, pyr_type=pyr_type)
        if reconstructed.ndim == 3:
            reconstructed = reconstructed.unsqueeze(1)
        outputs.append(reconstructed)
    result = torch.cat(outputs, dim=1).clamp(0, 1)
    return (result, prepared) if return_prepared else result
