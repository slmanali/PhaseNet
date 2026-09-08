import importlib.util
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
