import json
from pathlib import Path
import pytest

from tools.run_saved_evaluation import aggregate, jobs_from_plan, check_checkpoint
from utils.evaluation_protocol import sha256, verify_source_lock

ROOT = Path(__file__).resolve().parents[1]


def test_plan_has_twelve_models_and_seven_dataset_subsets():
    plan = json.loads((ROOT / "experiments/computer_optics_eval_12_legacy.json").read_text())
    jobs = jobs_from_plan(plan)
    assert len(jobs) == 84
    assert len({(job["variant"], job["seed"]) for job in jobs}) == 12
    assert {job["subset"] for job in jobs if job["dataset"] == "davis"} == {"val"}
    assert len(jobs_from_plan(plan, seeds=[33], datasets=["snufilm"])) == 16


def test_legacy_source_guard_refuses_modified_evaluator(tmp_path):
    path = tmp_path / "test.py"
    path.write_text("original\n")
    plan = {"source_lock": {"test.py": sha256(path)}}
    verify_source_lock(plan, tmp_path)
    path.write_text("corrected\n")
    with pytest.raises(ValueError, match="source changed"):
        verify_source_lock(plan, tmp_path)


def test_only_completed_exported_weights_can_be_evaluated(tmp_path):
    folder = tmp_path / "model" / "seed-11"
    folder.mkdir(parents=True)
    metadata = {"status": "completed", "completed_epochs": 40, "seed": 11,
                "configuration": "model", "arguments": {"feature_dim": 64}}
    (folder / "metadata.json").write_text(json.dumps(metadata))
    plan = {"run_root": str(tmp_path), "expected_epochs": 40, "models": {"model": {"feature_dim": 64}}}
    job = {"variant": "model", "seed": 11}
    with pytest.raises(FileNotFoundError, match="final exported"):
        check_checkpoint(plan, job)
    (folder / "model_final_state_dict.pth").touch()
    assert check_checkpoint(plan, job)[1]["completed_epochs"] == 40


def test_seed_summary_does_not_pool_frames_as_training_runs():
    rows = [{"status": "completed", "variant": "m", "dataset": "d", "subset": "v", "seed": seed,
             "sample_manifest_sha256": "same", "psnr": value, "ssim": .5, "lpips": .2, "pce": .3}
            for seed, value in [(11, 1.), (22, 3.)]]
    result = aggregate(rows, [11, 22, 33])[0]
    assert result["psnr_mean"] == 2
    assert result["psnr_sample_std"] == pytest.approx(2**.5)
    assert result["missing_seeds"] == [33]
