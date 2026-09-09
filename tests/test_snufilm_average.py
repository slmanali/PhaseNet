import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

from utils.average_metrics import METRICS, MODELS, MODES, mode_averages, representative_candidates


def load_figure_tool():
    path = Path(__file__).parents[1] / "tools/make_snufilm_average_paper_figure.py"
    spec = importlib.util.spec_from_file_location("average_figure", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def rows():
    output = []
    for mode in MODES:
        for model_number, model in enumerate(MODELS):
            for index, delta in enumerate((-1.0, 0.0, 2.0)):
                output.append({"model": model, "mode": mode, "sample_index": index,
                               **{metric: 10 * model_number + j + delta for j, metric in enumerate(METRICS)}})
    return output


def test_means_population_standard_deviations_and_normalized_distance():
    averages = mode_averages(rows())
    easy = next(r for r in averages if r["model"] == MODELS[0] and r["mode"] == "easy")
    assert easy["mean_psnr"] == np.mean([-1, 0, 2])
    assert easy["std_psnr"] == np.std([-1, 0, 2], ddof=0)
    candidate = next(r for r in representative_candidates(rows(), averages, 1) if r["mode"] == "easy")
    expected = abs(0 - np.mean([-1, 0, 2])) / (np.std([-1, 0, 2]) + 1e-8)
    assert np.isclose(candidate["phasenet_default_psnr_deviation"], expected)


def test_joint_minimum_is_one_common_sample_and_k_variants():
    averages = mode_averages(rows())
    one = representative_candidates(rows(), averages, 1)
    five_requested = representative_candidates(rows(), averages, 5)
    assert len(one) == 4 and all(r["sample_index"] == 1 for r in one)
    # Only three synthetic samples exist; k is an upper bound.
    assert len(five_requested) == 12
    assert set(one[0]) >= {f"{model}_{metric}" for model in MODELS for metric in METRICS}
    assert not any("selected_sample" in key for key in one[0])  # no model-specific selection field


def test_models_must_share_exact_sample_set():
    broken = rows()[:-1]
    try: representative_candidates(broken, mode_averages(broken), 1)
    except ValueError as error: assert "common sample-index set" in str(error)
    else: raise AssertionError("independent model-specific samples accepted")


def make_case(tmp_path, mismatch=False):
    selections = {}
    for mode in MODES:
        selections[mode] = {"sample_index": 7, "representative_score": .1}
        source = np.zeros((120, 180, 3), dtype=np.uint8); source[25:80, 50:100:2] = 255
        for number, model in enumerate(MODELS):
            directory = tmp_path / model / mode / "samples" / "7"; directory.mkdir(parents=True)
            triplet = source.copy()
            if mismatch and mode == "easy" and model == MODELS[-1]: triplet[0, 0] = 1
            for filename in ("input_1.png", "ground_truth.png", "input_2.png"):
                Image.fromarray(triplet).save(directory / filename)
            Image.fromarray(np.roll(source, number, axis=1)).save(directory / "prediction.png")
            (directory / "metrics.json").write_text(json.dumps({m: number for m in METRICS}))
    return selections


def test_native_resolution_sha_and_shared_valid_crop(tmp_path):
    tool = load_figure_tool(); selections = make_case(tmp_path)
    cases = tool.load_cases(tmp_path, selections)
    assert cases["easy"]["ground_truth"].shape[:2] == (120, 180)
    crop = tool.method_neutral_crop(cases["easy"]); tool.validate_crop(crop, (120, 180, 3))
    assert crop[2] - crop[0] == 96  # native-aware 20%, bounded to 96 minimum
    selections["easy"]["crop"] = crop
    # One crop field is mode-level, never attached to a model.
    assert all("crop" not in cases["easy"][model] for model in MODELS)


def test_sha_rejects_nonidentical_triplet(tmp_path):
    tool = load_figure_tool(); selections = make_case(tmp_path, mismatch=True)
    try: tool.load_cases(tmp_path, selections)
    except ValueError as error: assert "SHA256 mismatch" in str(error)
    else: raise AssertionError("different model triplets accepted")
