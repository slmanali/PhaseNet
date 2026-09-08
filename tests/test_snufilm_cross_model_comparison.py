import importlib.util
import json
from pathlib import Path

import numpy as np


def _load_tool():
    path = Path(__file__).parents[1] / "tools" / "make_snufilm_cross_model_comparison.py"
    spec = importlib.util.spec_from_file_location("cross_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_suggested_crops_are_shared_valid_and_nonoverlapping():
    tool = _load_tool()
    gt = np.zeros((64, 80, 3), dtype=np.uint8)
    gt[20:44, 25:50] = 255
    big = gt.copy(); big[18:48, 22:54] = 80
    full = gt.copy(); full[20:44, 25:50] = 240
    case = {name: {"ground_truth": gt, "prediction": full}
            for name in tool.MODELS}
    case["phasenet_big"]["prediction"] = big
    boxes = tool.suggest_crops(case, count=2)
    assert len(boxes) == 2
    for x0, y0, x1, y1 in boxes:
        assert 0 <= x0 < x1 <= 80
        assert 0 <= y0 < y1 <= 64
    assert boxes[0] != boxes[1]


def test_blur_crops_are_method_neutral_shared_and_saved(tmp_path):
    tool = _load_tool()
    gt = np.zeros((64, 80, 3), dtype=np.uint8)
    gt[18:46, 24:54:2] = 255
    case = {}
    for offset, model in enumerate(tool.MODELS):
        prediction = gt.copy()
        prediction[18:46, 24 + offset:54] = 100
        case[model] = {"ground_truth": gt, "prediction": prediction}
    boxes = tool.crop_config(tmp_path, case, "hard", 19, 2, "blur")
    payload = json.loads((tmp_path / "crop_coordinates.json").read_text())
    assert payload["crop_strategy"] == "blur"
    assert payload["crops"] == boxes
    # A single coordinate list is produced for all model panels, never per-model crops.
    assert all(len(box) == 4 for box in boxes)


def test_same_triplet_loader_rejects_missing_model_outputs(tmp_path):
    tool = _load_tool()
    try:
        tool.load_case(tmp_path, "easy", 7)
    except FileNotFoundError as error:
        assert "every model output for the same sample" in str(error)
    else:
        raise AssertionError("missing model outputs were accepted")
