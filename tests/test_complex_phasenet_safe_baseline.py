import torch

from net.complex_phasenet_safe_baseline import ComplexPhaseNetSafe


def test_tiled_inference_matches_full_frame_inference():
    torch.manual_seed(0)
    model = ComplexPhaseNetSafe(feature_dim=2).eval()
    real = [torch.randn(1, 2, 4, 5)]
    imag = [torch.randn(1, 2, 4, 5)]
    for height, width in ((7, 9), (11, 13), (17, 19), (25, 27)):
        real.append(torch.randn(1, 16, height, width))
        imag.append(torch.randn(1, 16, height, width))

    with torch.no_grad():
        full_real, full_imag = model(real, imag)
        tiled_real, tiled_imag = model(real, imag, tile_size=8)

    for full, tiled in zip(full_real + full_imag, tiled_real + tiled_imag):
        torch.testing.assert_close(tiled, full, rtol=1e-5, atol=1e-6)
