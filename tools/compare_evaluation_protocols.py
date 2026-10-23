"""Pair legacy/corrected scores only when checkpoint, triplets and sizes match."""
import argparse
import json
from pathlib import Path

from utils.evaluation_protocol import write_csv, write_json


def compare(before, after):
    output = []
    for path in sorted(after.glob("*/seed-*/*/*/summary.json")):
        previous_path = before / path.relative_to(after)
        if not previous_path.is_file():
            continue
        old, new = json.loads(previous_path.read_text()), json.loads(path.read_text())
        if old.get("status") != "completed" or new.get("status") != "completed":
            continue
        for field in ("variant", "seed", "dataset", "subset", "checkpoint_sha256", "sample_manifest_sha256", "tile_size"):
            if old.get(field) != new.get(field):
                raise ValueError(f"cannot isolate an evaluator change: {field} differs at {path}")
        for metric in ("psnr", "ssim", "lpips", "pce"):
            output.append({key: new[key] for key in ("variant", "seed", "dataset", "subset")}
                | {"metric": metric, "legacy": old[metric], "corrected": new[metric],
                   "corrected_minus_legacy": new[metric] - old[metric],
                   "checkpoint_sha256": new["checkpoint_sha256"]})
    if not output:
        raise ValueError("no matching completed before/after evaluations")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, default=Path("results/computer_optics/legacy_12models"))
    parser.add_argument("--after", type=Path, default=Path("results/computer_optics/corrected_12models"))
    parser.add_argument("--output", type=Path, default=Path("results/computer_optics/protocol_comparison.csv"))
    args = parser.parse_args()
    rows = compare(args.before, args.after)
    write_csv(args.output, rows)
    write_json(args.output.with_suffix(".json"), rows)
    print(f"Saved {len(rows)} metric differences to {args.output}")


if __name__ == "__main__":
    main()
