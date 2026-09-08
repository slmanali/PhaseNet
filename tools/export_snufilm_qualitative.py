#!/usr/bin/env python3
"""Evaluate SNU-FILM and export only the top metric samples for each model."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "phasenet_default": ("test.py", 64),
    "phasenet_big": ("test.py", 93),
    "complex_loss_matched": ("test_complex_safe_baseline.py", 64),
    "complex_full": ("test_complex_safe_baseline.py", 64),
}


def parse_pairs(values):
    result = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--checkpoint must be NAME=PATH: {value}")
        name, path = value.split("=", 1)
        if name not in MODELS:
            raise ValueError(f"unknown model {name!r}; choose from {', '.join(MODELS)}")
        if name in result:
            raise ValueError(f"duplicate checkpoint for {name}")
        result[name] = Path(path).expanduser().resolve()
    return result


def build_command(args, name, checkpoint):
    script, feature_dim = MODELS[name]
    command = [sys.executable, str(ROOT / script),
               "--dataset-type", "snufilm", "--dataset-path", str(args.snu_root),
               "--snu-mode", "all", "--image-size", args.image_size,
               "--batch-size", "1", "--model-path", str(checkpoint),
               "--feature-dim", str(feature_dim), "--model-name", name,
               "--save-dir", str(args.output_dir / name),
               "--metrics-dir", str(args.output_dir / "quantitative_metrics"),
               "--best-k", str(args.best_k)]
    if args.device:
        command.extend(("--device", args.device))
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snu-root", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True,
                        metavar="MODEL=FILE")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("qualitative_snufilm_best"))
    parser.add_argument("--best-k", type=int, default=1)
    parser.add_argument("--image-size", choices=("native", "256"), default="256",
                        help="Use the paper protocol (256) unless native is explicitly requested.")
    parser.add_argument("--device")
    args = parser.parse_args()
    if args.best_k < 1:
        parser.error("--best-k must be at least 1")
    args.snu_root = args.snu_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    checkpoints = parse_pairs(args.checkpoint)
    missing = [name for name in MODELS if name not in checkpoints]
    if missing:
        parser.error("missing checkpoints for: " + ", ".join(missing))
    for name, checkpoint in checkpoints.items():
        if not checkpoint.is_file():
            parser.error(f"checkpoint not found: {checkpoint}")
        subprocess.run(build_command(args, name, checkpoint), cwd=ROOT, check=True)
    print(f"Best-sample CSV: {args.output_dir / 'best_samples.csv'}")
    print(f"Best-sample JSON: {args.output_dir / 'best_samples.json'}")


if __name__ == "__main__":
    main()
