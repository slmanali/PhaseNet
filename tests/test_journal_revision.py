import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from utils.journal_diagnostics import JournalDiagnostics, masked_statistics
from tools.analyze_journal_evaluation import paired_cluster_bootstrap
from tools.compare_evaluation_protocols import compare
from tools.evaluate_journal_external import predict_float
from utils.metrics import compute_real_pce


def test_real_pce_masks_using_original_pyramid_amplitudes():
    residual = torch.zeros(1, 1, 1, 1)
    truth = torch.tensor([.5, .5, 0., 0.]).reshape(1, 4, 1, 1)
    prediction = torch.tensor([.5, .5, 1., .2]).reshape(1, 4, 1, 1)
    scales = torch.tensor([1e-5, 1.]).reshape(1, 2, 1, 1)
    result = compute_real_pce([residual, truth], [residual, prediction], amp_scales=[scales])
    assert result == pytest.approx(.2)
    assert compute_real_pce([residual, truth], [residual, prediction]) == pytest.approx(.6)


def test_wrapping_and_amplitude_mask_are_explicit():
    result = masked_statistics(torch.tensor([-np.pi + .1, 2.0]),
                               torch.tensor([np.pi - .1, 0.0]), torch.tensor([1.0, 1e-5]))
    assert result["error_sum"] == pytest.approx(.2, abs=1e-6)
    assert result["valid"] == 1 and result["total"] == 2


def test_diagnostics_separate_internal_midpoint_and_reconstructed_error(tmp_path):
    residual = torch.zeros(1, 1, 2, 2)
    train = [torch.zeros(1, 2, 2, 2), torch.zeros(1, 16, 2, 2)]
    truth = [residual, torch.cat((torch.ones(1, 4, 2, 2), torch.zeros(1, 4, 2, 2)), 1)]
    prediction = [residual, torch.cat((torch.ones(1, 4, 2, 2), torch.full((1, 4, 2, 2), .2)), 1)]
    class Pyramid:
        def build(self, images, pyr_type):
            band = torch.zeros(2, 2, 2, 2)
            band[0, ..., 0], band[0, ..., 1] = np.cos(.3), np.sin(.3)
            band[1, ..., 0] = 1
            return [residual, [band] * 4, residual]
    diagnostic = JournalDiagnostics("cpu")
    for channel in range(3):
        diagnostic.channel(0, channel, train, truth, prediction, amp_scales=[torch.ones(1, 4, 2, 2)])
    frame = torch.zeros(1, 3, 2, 2)
    diagnostic.reconstructed(0, frame, frame, Pyramid())
    row = diagnostic.samples[0]
    assert row["internal_pce"] == pytest.approx(.2)
    assert row["circular_midpoint_pce"] == pytest.approx(0)
    assert row["reconstructed_pce"] == pytest.approx(.3)
    assert row["internal_valid_fraction"] == 1
    assert row["internal_valid_coefficients"] == 48
    assert row["phase_unit_loss_unweighted"] is None


def test_cluster_bootstrap_preserves_triplet_weighting():
    groups = ["a", "b", "b", "b"]
    result = paired_cluster_bootstrap([[2, 0, 0, 0], [4, 0, 0, 0]], groups, 200)
    assert result["mean_difference"] == .75  # not an equal-video mean of 1.5
    constant = paired_cluster_bootstrap([[.2] * 4] * 3, groups, 200)
    assert constant["ci95_low"] == pytest.approx(.2)
    assert constant["ci95_high"] == pytest.approx(.2)
    with pytest.raises(ValueError, match="two independent"):
        paired_cluster_bootstrap([[1, 2]], ["same", "same"], 200)


def test_before_after_comparison_rejects_a_different_checkpoint(tmp_path):
    old, new = tmp_path / "before", tmp_path / "after"
    suffix = Path("model/seed-11/davis/val/summary.json")
    row = dict(status="completed", variant="model", seed=11, dataset="davis", subset="val",
               checkpoint_sha256="a", sample_manifest_sha256="data", tile_size=0,
               psnr=25.0, ssim=.8, lpips=.2, pce=.9)
    for root in (old, new):
        (root / suffix).parent.mkdir(parents=True)
        (root / suffix).write_text(json.dumps(row))
    result = compare(old, new)
    assert all(item["corrected_minus_legacy"] == 0 for item in result)
    (new / suffix).write_text(json.dumps(dict(row, checkpoint_sha256="b")))
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        compare(old, new)


@pytest.mark.parametrize("kind", ["rife", "film"])
def test_external_path_keeps_fractional_rgb_inputs(kind):
    first = torch.full((1, 3, 7, 9), .12345)
    second = torch.full_like(first, .65432)
    calls = []
    if kind == "rife":
        class Model:
            def inference(self, a, b, scale):
                calls.append(a[..., :7, :9].clone())
                return .5 * (a + b)
        backend = SimpleNamespace(device=torch.device("cpu"), scale=1, model=Model())
    else:
        def model(a, b, time):
            calls.append(torch.from_numpy(a.copy()).permute(0, 3, 1, 2))
            return .5 * (a + b)
        backend = SimpleNamespace(model=model)
    result = predict_float(backend, kind, first, second)
    torch.testing.assert_close(calls[0], first)
    torch.testing.assert_close(result, .5 * (first + second))


@pytest.mark.parametrize("architecture", ["real", "complex"])
def test_corrected_worker_runs_on_synthetic_frames_without_metric_downloads(tmp_path, monkeypatch, architecture):
    import lpips
    from tools import run_saved_evaluation as driver
    from net.phasenet import PhaseNet
    from net.complex_phasenet_safe_baseline import ComplexPhaseNetSafe
    class MetricStub(torch.nn.Module):
        def forward(self, a, b):
            return (a - b).abs().mean().reshape(1, 1, 1, 1)
    monkeypatch.setattr(lpips, "LPIPS", lambda **kw: MetricStub())
    torch.manual_seed(4)
    name = "synthetic-" + architecture
    folder = tmp_path / "runs" / name / "seed-11"
    folder.mkdir(parents=True)
    model = PhaseNet(2) if architecture == "real" else ComplexPhaseNetSafe(2)
    torch.save(model.state_dict(), folder / "model_final_state_dict.pth")
    (folder / "metadata.json").write_text(json.dumps({"status": "completed", "completed_epochs": 1,
        "seed": 11, "configuration": name, "arguments": {"feature_dim": 2}}))
    images = torch.rand(3, 3, 256, 256)
    samples = [{"start": images[0], "inter": images[1], "end": images[2]}]
    manifest = [{"sample_index": 0, "sample_key": "fixture", "sequence_id": "fixture-video",
                 "sequence_grouping_verified": True, "frames": ["a", "b", "c"]}]
    monkeypatch.setattr(driver, "make_dataset", lambda *a: (samples, manifest, {"sample_manifest_sha256": "fixture"}))
    plan = {"stage": "corrected", "study_id": "synthetic_test_only", "evaluation_sources": [],
            "run_root": str(tmp_path / "runs"), "expected_epochs": 1, "deterministic": "strict",
            "models": {name: {"architecture": architecture, "feature_dim": 2}}, "datasets": {"fixture": {}}}
    job = {"variant": name, "seed": 11, "dataset": "fixture", "subset": "test"}
    result = driver.run_job(plan, job, tmp_path / "output", "cpu")
    assert result["status"] == "completed" and result["samples"] == 1
    diagnostic = json.loads((tmp_path / "output/diagnostics.json").read_text())
    assert diagnostic["internal_pce"] == pytest.approx(result["pce"], abs=2e-6)
    assert np.isfinite(diagnostic["reconstructed_pce"])
    if architecture == "complex":
        assert np.isfinite(diagnostic["phase_unit_loss_unweighted"])
    # A repeated call must skip, without downloading metrics or changing output.
    assert driver.run_job(plan, job, tmp_path / "output", "cpu")["checkpoint_sha256"] == result["checkpoint_sha256"]
