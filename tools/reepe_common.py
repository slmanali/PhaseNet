"""Small shared utilities for the two independent REEPE result collections."""
import csv
import json
from pathlib import Path


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported result value: {type(value).__name__}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=json_default) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, default=json_default) if isinstance(value, (dict, list))
                          else value for key, value in row.items()} for row in rows)
    temporary.replace(path)


def select_runs(plan, configs=None, seeds=None):
    configs = configs or list(plan["configurations"])
    seeds = seeds or plan["seeds"]
    if len(set(configs)) != len(configs) or len(set(seeds)) != len(seeds):
        raise ValueError("configuration and seed selections must be unique")
    if set(configs) - set(plan["configurations"]):
        raise ValueError("configuration is not in this study plan")
    if set(seeds) - set(plan["seeds"]):
        raise ValueError("seed is not in this study plan")
    return [(name, seed) for name in configs for seed in seeds]


def completed_checkpoint(root, name, seed, epochs, study_id=None):
    directory = Path(root) / name / f"seed-{seed}"
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("status") != "completed" or metadata.get("completed_epochs") != epochs:
        raise ValueError(f"{directory}: requires completed epoch {epochs}")
    if metadata.get("configuration") != name or metadata.get("seed") != seed:
        raise ValueError(f"{directory}: configuration/seed metadata mismatch")
    if study_id is not None and metadata.get("study_id") != study_id:
        raise ValueError(f"{directory}: wrong study; journal quality results cannot be reused here")
    for filename in ("checkpoint_last.pth", "model_final_state_dict.pth"):
        checkpoint = directory / filename
        if checkpoint.is_file():
            return checkpoint, metadata
    raise FileNotFoundError(f"{directory}: no saved weights; this tool never starts training")
