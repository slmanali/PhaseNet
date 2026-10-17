"""Export saved training settings, checkpoint identity and honest legacy log curves."""
import argparse
import json
import math
from pathlib import Path
import re

import numpy as np

from utils.evaluation_protocol import sha256, write_csv, write_json


def parse_log(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        match = re.match(r"Step\s+(\d+)\s*\|\s*Loss\s+(.*)", line)
        if not match:
            continue
        values = {key: float(value) for key, value in re.findall(r"(\w+)\s*=\s*([-+\d.eE]+)", match.group(2))}
        rows.append({"step": int(match.group(1)), **values})
    steps = [row["step"] for row in rows]
    if not rows or steps != sorted(set(steps)):
        raise ValueError(f"log has missing, duplicate or out-of-order step records: {path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("experiments/computer_optics_eval_12_corrected.json"))
    parser.add_argument("--log-map", type=Path, default=Path("experiments/computer_optics_log_map.json"))
    parser.add_argument("--output", type=Path, default=Path("results/computer_optics/training_audit"))
    parser.add_argument("--logs-only", action="store_true")
    args = parser.parse_args()
    plan, mapping = json.loads(args.plan.read_text()), json.loads(args.log_map.read_text())
    records, curves = [], []
    for name, config in plan["models"].items():
        for seed in plan["seeds"]:
            folder = Path(plan["run_root"]) / name / f"seed-{seed}"
            metadata = json.loads((folder / "metadata.json").read_text())
            row = {"variant": name, "seed": seed, "completed_epochs": metadata.get("completed_epochs"),
                   "completed_steps": metadata.get("completed_steps"), "status": metadata.get("status"),
                   "training_triplets": metadata.get("sample_count"), "split_sha256": metadata.get("split_sample_sha256"),
                   "arguments": metadata.get("arguments"), "optimizer": metadata.get("optimizer"),
                   "scheduler": metadata.get("scheduler"), "environment": metadata.get("environment"),
                   "checkpoint_selection": metadata.get("checkpoint_selection")}
            if not args.logs_only:
                from utils.model_loading import load_inference_model
                checkpoint = folder / "model_final_state_dict.pth"
                model, info = load_inference_model(checkpoint, "cpu", config["architecture"], feature_dim=config["feature_dim"])
                row.update(checkpoint_sha256=info["checkpoint_sha256"],
                           parameters=sum(p.numel() for p in model.parameters()),
                           alpha=float(model.alpha.detach()), beta=float(model.beta.detach()),
                           alpha_beta_constraint="none; unconstrained learned real scalars initialized at 0.5")
                row["parameters_finite"] = all(bool(np.isfinite(value.detach().cpu().numpy()).all()) for value in model.parameters())
                del model
            log = Path(mapping["runs"][name][str(seed)])
            points = parse_log(log)
            steps_per_epoch = math.ceil(metadata["sample_count"] / metadata["arguments"]["batch_size"])
            row.update(log_path=str(log), log_sha256=sha256(log), logged_batches=len(points),
                       first_logged_step=points[0]["step"], last_logged_step=points[-1]["step"],
                       log_step_coverage_complete=points[-1]["step"] == metadata["completed_steps"],
                       log_identity="explicit legacy mapping; source log has no embedded run identity")
            for point in points:
                curves.append({"variant": name, "seed": seed, "epoch_position": point["step"] / steps_per_epoch,
                               "record_kind": "rounded logged minibatch, last RGB channel; not epoch mean", **point})
            records.append(row)
    write_json(args.output / "runs.json", records)
    write_csv(args.output / "runs.csv", records)
    write_csv(args.output / "logged_training_values.csv", curves)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for axis, metric in zip(axes.flat, ("total", "img", "grad", "phase")):
        for name, color in (("complex-full-64", "#2166ac"), ("complex-loss-matched-64", "#b2182b")):
            for seed in plan["seeds"]:
                points = [row for row in curves if row["variant"] == name and metric in row and row["seed"] == seed]
                values = np.array([row[metric] for row in points])
                window = min(101, len(values))
                smooth = np.convolve(values, np.ones(window) / window, mode="valid")
                x = [row["epoch_position"] for row in points[window - 1:]]
                axis.plot(x, smooth, color=color, alpha=.65, linewidth=1,
                          label=f"{name}, seed {seed}")
        axis.set(xlabel="Epoch position", ylabel=f"Logged {metric} loss")
        axis.grid(alpha=.2)
    axes.flat[0].legend(fontsize=6)
    fig.suptitle("Logged minibatches: 101-record moving mean (not validation curves)")
    fig.savefig(args.output / "complex_training_logs.png", dpi=180)
    fig.savefig(args.output / "complex_training_logs.pdf")
    plt.close(fig)
    write_json(args.output / "limitations.json", {
        "total_loss_comparison": "full and loss-matched models have different weights; totals are not a common objective",
        "historical_logs": "one rounded minibatch every ten steps; complex logger recorded the last RGB channel",
        "phase_unit_zero": "printed 0.000 only establishes rounding; use corrected diagnostics for unrounded values",
        "convergence": "equal epochs and smoothed logs do not establish equal convergence"})
    print(f"Saved training audit and labelled historical curves to {args.output}")


if __name__ == "__main__":
    main()
