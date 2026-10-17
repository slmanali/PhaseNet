"""Controls that protect the scientific meaning of the component experiment."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from net.complex_nn.complex_layers import ComplexBatchNorm2d, ComplexModReLU, ComplexTanh
from net.complex_phasenet_safe_baseline import ComplexPhaseNetSafe
from tools.benchmark_interpolation import measure
from tools.evaluate_reepe_ablations import summarize
from tools.reepe_common import completed_checkpoint, select_runs, write_csv, write_json
from tools.run_experiments import command
from train_complex_safe_baseline import create_training_model
from utils.model_loading import load_inference_model

ROOT = Path(__file__).resolve().parents[1]


def settings(**changes):
    return SimpleNamespace(**(dict(feature_dim=4, convolution="complex", normalization="full",
        activation="leaky_relu", phase_correction_scale=.1, residual_correction_scale=.1) | changes))


def test_plan_contains_fifteen_fresh_one_component_runs(tmp_path):
    plan = json.loads((ROOT / "experiments/reepe_10epoch_ablations.json").read_text())
    assert plan["common"]["epochs"] == 10
    assert set(plan["seeds"]).isdisjoint({11, 22, 33})
    runs = select_runs(plan)
    assert len(runs) == 15
    args = SimpleNamespace(run_root=tmp_path, dataset_path="DAVIS", deterministic="strict")
    for name, seed in runs:
        config = plan["configurations"][name]
        assert len(set(config) - {"entrypoint"}) == (0 if name == "reepe10-reference" else 1)
        cmd = command(config["entrypoint"], plan["common"], config, name, seed, args)
        assert "--resume" not in cmd
        assert cmd[cmd.index("--epochs") + 1] == "10"


def test_diagonal_bn_removes_covariance_mixing_with_identical_parameters():
    full = ComplexBatchNorm2d(2)
    diagonal = ComplexBatchNorm2d(2, whitening="diagonal")
    diagonal.load_state_dict(full.state_dict(), strict=True)
    assert list(full.state_dict()) == list(diagonal.state_dict())
    real = torch.randn(3, 2, 5, 5, requires_grad=True)
    imag = real.detach() * .8 + torch.randn_like(real) * .2
    imag.requires_grad_()
    actual = diagonal(real, imag)
    expected = full(real, imag)
    assert not torch.allclose(actual[0], expected[0])
    (actual[0].square().mean() + actual[1].square().mean()).backward()
    assert torch.isfinite(real.grad).all() and torch.isfinite(imag.grad).all()
    for value in diagonal.parameters():
        assert torch.isfinite(value.grad).all()
    assert torch.equal(full.running_var_ri, diagonal.running_var_ri)


@pytest.mark.parametrize("zero", [False, True])
def test_modrelu_preserves_phase_and_has_finite_gradients(zero):
    activation = ComplexModReLU(2)
    real = (torch.zeros if zero else torch.randn)(2, 2, 3, 3).requires_grad_()
    imag = (torch.zeros if zero else torch.randn)(2, 2, 3, 3).requires_grad_()
    out_r, out_i = activation(real, imag)
    assert torch.allclose(out_r * imag, out_i * real, atol=1e-6)
    (out_r.sum() + out_i.sum()).backward()
    assert torch.isfinite(real.grad).all() and torch.isfinite(imag.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in activation.parameters())


@pytest.mark.parametrize("change", [{"normalization": "diagonal"}, {"activation": "modrelu"},
                                   {"phase_correction_scale": 0.0}, {"convolution": "unrestricted_real"}])
def test_component_initialization_preserves_shared_state_and_rng(change):
    torch.manual_seed(101)
    reference = create_training_model(settings())
    rng = torch.get_rng_state().clone()
    torch.manual_seed(101)
    variant = create_training_model(settings(**change))
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in variant.state_dict().items():
        if key in reference.state_dict():
            assert torch.equal(value, reference.state_dict()[key]), key
    if change.get("activation") == "modrelu":
        assert all(isinstance(block.activation, ComplexModReLU) for block in variant.layer)
        assert all(isinstance(head.activation, ComplexTanh) for head in variant.pred)


@pytest.mark.parametrize("change", [{"normalization": "diagonal"}, {"activation": "modrelu"},
                                   {"phase_correction_scale": 0.0}, {"convolution": "unrestricted_real"}])
def test_inference_restores_non_tensor_flags_without_restoring_training_rng(tmp_path, change):
    args = settings(**change)
    model = create_training_model(args)
    path = tmp_path / "checkpoint.pth"
    torch.save({"model_state_dict": model.state_dict(), "configuration": vars(args), "epoch": 10}, path)
    before = torch.get_rng_state().clone()
    restored, info = load_inference_model(path)
    assert torch.equal(before, torch.get_rng_state())
    assert info["epoch"] == 10
    assert restored.normalization == args.normalization
    assert restored.feature_activation == args.activation
    assert restored.phase_correction_scale == args.phase_correction_scale
    assert restored.convolution == args.convolution
    for a, b in zip(model.state_dict().values(), restored.state_dict().values()):
        assert torch.equal(a, b)
    with pytest.raises(ValueError, match="conflicts"):
        load_inference_model(path, feature_dim=64)


def test_quality_study_rejects_journal_checkpoint(tmp_path):
    folder = tmp_path / "reference" / "seed-101"
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(json.dumps({"status": "completed", "completed_epochs": 40,
        "configuration": "reference", "seed": 101, "study_id": "journal"}))
    with pytest.raises(ValueError, match="epoch 10"):
        completed_checkpoint(tmp_path, "reference", 101, 10, "reepe_component_10ep_v1")


def test_seed_aggregation_rejects_duplicates_and_marks_incomplete():
    row = dict(variant="ref", seed=101, dataset="davis", subset="val", sample_manifest_sha256="a",
               psnr=2.0, ssim=.5, lpips=.4, pce=.2)
    with pytest.raises(ValueError, match="duplicate"):
        summarize([row, row], [101, 202, 303])
    second = dict(row, seed=202, psnr=4.0)
    summary = summarize([row, second], [101, 202, 303])[0]
    assert summary["psnr_mean"] == 3.0
    assert summary["psnr_sample_std"] == pytest.approx(2**.5)
    assert summary["missing_seeds"] == [303] and not summary["complete"]


def test_cpu_measurement_never_reports_gpu_memory():
    result = measure(lambda: torch.ones(2, 2) + 1, torch.device("cpu"), 1, 2, True)
    assert result["peak_allocated_mib"] is None
    assert result["incremental_peak_allocated_mib"] is None
    assert len(result["latency_samples_ms"]) == 2
    with pytest.raises(ValueError):
        measure(lambda: None, torch.device("cpu"), 0, 2, False)


def test_cost_plan_selects_all_twelve_finished_checkpoints():
    plan = json.loads((ROOT / "experiments/reepe_profile_12models.json").read_text())
    assert len(select_runs(plan)) == 12
    assert plan["expected_epochs"] == 40
    assert plan["seeds"] == [11, 22, 33]


def test_phase_off_keeps_the_circular_midpoint_when_head_changes():
    model = ComplexPhaseNetSafe(2, phase_correction_scale=0.0).eval()
    real = [torch.randn(1, 2, 4, 4), torch.randn(1, 16, 8, 8)]
    imag = [torch.randn_like(value) for value in real]
    base_r, base_i = model.normalize_unit_complex(real[1][:, 4:8] + real[1][:, 12:16],
                                                 imag[1][:, 4:8] + imag[1][:, 12:16])
    expected = model.normalize_unit_complex(base_r, base_i)
    with torch.no_grad():
        for parameter in model.pred[1].parameters():
            parameter.add_(2)
        output_r, output_i = model(real, imag)
    torch.testing.assert_close(output_r[1][:, 4:], expected[0])
    torch.testing.assert_close(output_i[1][:, 4:], expected[1])


def test_profile_serializes_paths_in_historical_checkpoint_arguments(tmp_path):
    row = {"model": {"training_arguments": {"run_root": Path("runs_seeded")}}}
    write_json(tmp_path / "result.json", row)
    write_csv(tmp_path / "result.csv", [row])
    assert json.loads((tmp_path / "result.json").read_text())["model"]["training_arguments"]["run_root"] == "runs_seeded"


def test_quality_driver_exports_and_skips_completed_runs(tmp_path, monkeypatch):
    import sys
    import tools.evaluate_reepe_ablations as driver

    plan_path = ROOT / "experiments/reepe_10epoch_ablations.json"
    plan = json.loads(plan_path.read_text())
    name, seed = "reepe10-reference", 101
    arguments = dict(plan["common"], config_name=name, seed=seed)
    info = {"epoch": 10, "seed": seed, "configuration": name, "study_id": plan["study_id"],
            "checkpoint_sha256": "fixture", "training_arguments": arguments}
    monkeypatch.setattr(driver, "completed_checkpoint", lambda *a: (tmp_path / "fixture.pth", {"split_sample_sha256": "train-fixture"}))
    monkeypatch.setattr(driver, "load_inference_model", lambda *a, **kw: (torch.nn.Identity(), info))
    monkeypatch.setattr(driver, "make_dataset", lambda *a: ([{"start": torch.ones(3, 4, 4)}],
        [{"sample_index": 0, "sequence_id": "test-sequence", "frames": ["0.jpg", "1.jpg", "2.jpg"]}]))
    calls = []
    def fixture_evaluate(model, loader, device, sample_callback):
        calls.append(1)
        scores = {"psnr": 20.0, "ssim": .7, "lpips": .2, "pce": .4}
        sample_callback(0, dict(scores, height=4, width=4))
        return dict(scores, samples=1)
    monkeypatch.setattr(driver, "evaluate", fixture_evaluate)
    monkeypatch.setattr(driver, "environment_metadata", lambda: {"fixture": True})
    monkeypatch.setattr(sys, "argv", ["evaluate", "--plan", str(plan_path), "--dataset", "davis",
        "--dataset-path", str(tmp_path), "--device", "cpu", "--output-root", str(tmp_path / "results"),
        "--configs", name, "--seeds", str(seed)])
    driver.main()
    destination = tmp_path / "results/davis-val-256"
    assert (destination / name / "seed-101/samples.csv").is_file()
    assert (destination / name / "seed-101/sequences.csv").is_file()
    aggregate = json.loads((destination / "aggregates.json").read_text())[0]
    assert aggregate["missing_seeds"] == [202, 303]
    driver.main()
    assert len(calls) == 1
