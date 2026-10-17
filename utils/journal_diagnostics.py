"""Phase diagnostics requested in review; these never change predictions."""
from collections import defaultdict
import statistics

import torch

from utils.evaluation_protocol import write_csv, write_json
from utils.metrics import AMP_EPS


def wrapped_error(left, right):
    difference = left - right
    return torch.atan2(torch.sin(difference), torch.cos(difference)).abs()


def masked_statistics(prediction, truth, amplitude, threshold=AMP_EPS):
    mask = amplitude > threshold
    error = wrapped_error(prediction, truth)
    return {"error_sum": float(error[mask].double().sum().item()),
            "valid": int(mask.sum().item()), "total": mask.numel()}


class JournalDiagnostics:
    def __init__(self, device):
        self.device = device
        self.samples = []
        self.bands = defaultdict(lambda: {"error_sum": 0.0, "valid": 0, "total": 0})
        self.current = None

    def _record(self, kind, channel, level, orientation, values):
        for destination in (self.bands[(kind, channel, level, orientation)],
                            self.current["totals"][(kind, channel)]):
            for key in ("error_sum", "valid", "total"):
                destination[key] += values[key]

    @torch.no_grad()
    def channel(self, index, channel, inputs, truth, prediction, amp_scales=None, complex_model=False):
        if channel == 0:
            self.current = {"index": index, "unit": [], "lowpass_imaginary_l1": [],
                            "totals": defaultdict(lambda: {"error_sum": 0.0, "valid": 0, "total": 0})}
        if self.current is None or self.current["index"] != index:
            raise ValueError("diagnostic sample/channel ordering differs from evaluation")
        if complex_model:
            train_r, train_i = inputs
            truth_r, truth_i = truth
            pred_r, pred_i = prediction
            self.current["lowpass_imaginary_l1"].append(float((pred_i[0] - truth_i[0]).abs().mean()))
            levels = len(truth_r)
        else:
            levels = len(truth)
        for level in range(1, levels):
            if complex_model:
                n = truth_r[level].shape[1] // 2
                amplitude = torch.sqrt(truth_r[level][:, :n]**2 + truth_i[level][:, :n]**2)
                target = torch.atan2(truth_i[level][:, n:], truth_r[level][:, n:])
                estimate = torch.atan2(pred_i[level][:, n:], pred_r[level][:, n:])
                midpoint = torch.atan2(train_i[level][:, n:2*n] + train_i[level][:, 3*n:4*n],
                                       train_r[level][:, n:2*n] + train_r[level][:, 3*n:4*n])
                # Exact expression used by the historical ComplexTotalLoss.
                norm = torch.sqrt(pred_r[level][:, n:]**2 + pred_i[level][:, n:]**2 + 1e-8)
                self.current["unit"].append(float((norm - 1).abs().mean()))
            else:
                n = truth[level].shape[1] // 2
                amplitude = truth[level][:, :n] * amp_scales[level - 1]
                target, estimate = truth[level][:, n:], prediction[level][:, n:]
                start, end = inputs[level][:, n:2*n], inputs[level][:, 3*n:4*n]
                midpoint = torch.atan2(torch.sin(start) + torch.sin(end), torch.cos(start) + torch.cos(end))
            for orientation in range(n):
                amp, target_band = amplitude[:, orientation], target[:, orientation]
                self._record("internal", channel, level, orientation,
                             masked_statistics(estimate[:, orientation], target_band, amp))
                self._record("circular_midpoint", channel, level, orientation,
                             masked_statistics(midpoint[:, orientation], target_band, amp))

    @torch.no_grad()
    def reconstructed(self, index, prediction, truth, pyramid, pyr_type=1):
        if self.current is None or self.current["index"] != index:
            raise ValueError("internal phase diagnostics missing for this reconstructed frame")
        for channel in range(3):
            coefficients = pyramid.build(torch.cat((prediction[:, channel:channel+1],
                                                    truth[:, channel:channel+1]), dim=0), pyr_type=pyr_type)
            # Reverse oriented levels to match the model's coarse-to-fine indices.
            for level, bands in enumerate(coefficients[1:-1][::-1], 1):
                for orientation, band in enumerate(bands):
                    if band.ndim != 4 or band.shape[-1] != 2:
                        raise ValueError("expected a complex oriented pyramid band")
                    estimate = torch.atan2(band[0, ..., 1], band[0, ..., 0])
                    target = torch.atan2(band[1, ..., 1], band[1, ..., 0])
                    amplitude = torch.sqrt(band[1, ..., 0]**2 + band[1, ..., 1]**2 + 1e-8)
                    self._record("reconstructed", channel, level, orientation,
                                 masked_statistics(estimate, target, amplitude))
        row = {"sample_index": index}
        for kind in ("internal", "circular_midpoint", "reconstructed"):
            channels = [self.current["totals"][(kind, channel)] for channel in range(3)]
            row[kind + "_pce"] = statistics.mean(value["error_sum"] / value["valid"] if value["valid"] else 0.0
                                                  for value in channels)
            valid, total = sum(value["valid"] for value in channels), sum(value["total"] for value in channels)
            row[kind + "_valid_coefficients"] = valid
            row[kind + "_total_coefficients"] = total
            row[kind + "_valid_fraction"] = valid / total if total else None
        row["phase_unit_loss_unweighted"] = statistics.mean(self.current["unit"]) if self.current["unit"] else None
        row["lowpass_imaginary_l1"] = (statistics.mean(self.current["lowpass_imaginary_l1"])
                                      if self.current["lowpass_imaginary_l1"] else None)
        self.samples.append(row)
        self.current = None

    def save(self, destination, sample_rows):
        if len(self.samples) != len(sample_rows):
            raise ValueError("diagnostic and quality sample counts differ")
        rows = []
        for identity, diagnostic in zip(sample_rows, self.samples):
            if identity["sample_index"] != diagnostic["sample_index"]:
                raise ValueError("diagnostic sample IDs differ")
            rows.append({key: identity[key] for key in
                         ("variant", "seed", "dataset", "subset", "sample_key", "sequence_id")}
                        | diagnostic)
        write_csv(destination / "phase_diagnostics.csv", rows)
        bands = []
        for (kind, channel, level, orientation), values in sorted(self.bands.items()):
            bands.append({"kind": kind, "rgb_channel": channel, "coarse_to_fine_level": level,
                          "orientation": orientation, **values,
                          "valid_fraction": values["valid"] / values["total"] if values["total"] else None,
                          "coefficient_pooled_pce": values["error_sum"] / values["valid"] if values["valid"] else None})
        write_csv(destination / "phase_by_band.csv", bands)
        summary = {"samples": len(rows), "amplitude_threshold": AMP_EPS,
                   "reconstructed_input": "final clamped RGB prediction, decomposed again",
                   "aggregation": "valid-coefficient mean per RGB channel; RGB mean; equal image mean",
                   "empty_mask_rule": "channel error zero; retained counts expose empty masks",
                   "phase_unit_definition": "mean abs(sqrt(real_phase^2 + imag_phase^2 + 1e-8)-1); equal band/RGB mean",
                   "interpretation": "coefficient phase agreement; not video temporal consistency"}
        for key in rows[0]:
            if key.endswith(("_pce", "_valid_fraction")) or key in {"phase_unit_loss_unweighted", "lowpass_imaginary_l1"}:
                values = [row[key] for row in rows if row[key] is not None]
                summary[key] = statistics.mean(values) if values else None
        write_json(destination / "diagnostics.json", summary)
