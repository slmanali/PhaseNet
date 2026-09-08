import importlib.util
from pathlib import Path

import numpy as np


def _load_tool(name):
    path = Path(__file__).parents[1] / "tools" / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_crop_suggestions_are_valid_and_nonidentical():
    tool = _load_tool("export_snufilm_qualitative.py")
    gt = np.zeros((80, 100, 3), dtype=np.float32)
    big = gt.copy(); full = gt.copy()
    big[5:20, 5:20] = 1
    big[55:70, 75:90] = .8
    result = tool.boxes(big, full, gt, size=24, count=2)
    assert len(result) == 2
    assert result[0] != result[1]
    for x0, y0, x1, y1 in result:
        assert 0 <= x0 < x1 <= 100
        assert 0 <= y0 < y1 <= 80


def test_figure_panel_combines_full_frame_and_shared_crops():
    tool = _load_tool("make_snufilm_qualitative_figure.py")
    frame = np.zeros((60, 120, 3), dtype=np.uint8)
    result = tool.panel(frame, [[0, 0, 20, 20], [80, 40, 100, 60]])
    assert result.shape[0] == 130
    assert result.shape[1] > 260
