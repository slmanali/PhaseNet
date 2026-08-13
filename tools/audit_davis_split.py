#!/usr/bin/env python3
"""Audit official DAVIS train/validation isolation at sequence and frame level."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.davis import (davis_image_root, load_davis_train_val_info,
                         resolve_davis_root)


def frame_paths(dataset):
    return {Path(item[0]).resolve() for triplet in dataset.sample for item in triplet}


def fail_count_audit(label, split, dataset):
    expected = len(split.frames) - 2 * len(split.sequences)
    actual = len(dataset)
    if actual == expected:
        return
    generated_by_sequence = {name: 0 for name in split.sequences}
    for triplet in dataset.sample:
        generated_by_sequence[Path(triplet[0][0]).parent.name] += 1
    print(f"{label} TRIPLET COUNT MISMATCH", file=sys.stderr)
    print(f"  split file: {split.path}", file=sys.stderr)
    print(f"  frame entries: {len(split.frames)}", file=sys.stderr)
    print(f"  unique sequences: {len(split.sequences)}", file=sys.stderr)
    print(f"  expected frames - 2*sequences: {expected}", file=sys.stderr)
    print(f"  generated triplets: {actual}", file=sys.stderr)
    print("  generated triplets by sequence:", file=sys.stderr)
    for name, count in generated_by_sequence.items():
        print(f"    {name}: {count}", file=sys.stderr)
    raise RuntimeError(f"{label.lower()} generated triplet count failed audit")


def main():
    parser = argparse.ArgumentParser()
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--davis-root", help="DAVIS root containing JPEGImages and ImageSets")
    location.add_argument(
        "--dataset-path",
        help="DAVIS root, JPEGImages directory, or JPEGImages/480p directory",
    )
    parser.add_argument(
        "--imageset-root",
        help=("Optional separate DAVIS root containing ImageSets/2017; a "
              "JPEGImages or JPEGImages/480p path is also accepted"),
    )
    args = parser.parse_args()

    root = resolve_davis_root(args.davis_root, args.dataset_path)
    image_root = davis_image_root(root, args.dataset_path)
    if not image_root.is_dir():
        parser.error(f"DAVIS image directory does not exist: {image_root}")
    all_sequences = sorted(p.name for p in image_root.iterdir() if p.is_dir())
    train, val = load_davis_train_val_info(root, args.imageset_root)
    from net.phasenet import Triplets

    overlap = set(train.sequences) & set(val.sequences)
    split_frame_overlap = set(train.frames) & set(val.frames)
    train_data = Triplets(str(image_root), allowed_sequences=train.sequences)
    val_data = Triplets(str(image_root), allowed_sequences=val.sequences)
    frame_overlap = frame_paths(train_data) & frame_paths(val_data)
    total_assigned_sequences = len(train.sequences) + len(val.sequences)
    total_triplets = len(train_data) + len(val_data)
    if not total_triplets:
        raise RuntimeError("The official DAVIS train/val sequences generated no triplets")

    if train.frames:
        fail_count_audit("TRAINING", train, train_data)
    if val.frames:
        fail_count_audit("VALIDATION", val, val_data)

    print("DAVIS source resolution: 480p")
    print(f"DAVIS root: {root}")
    print(f"Image root: {image_root}")
    print(f"Train split file: {train.path}")
    print(f"Val split file: {val.path}")
    print(f"Training frame entries: {len(train.frames)}")
    print(f"Validation frame entries: {len(val.frames)}")
    print(f"Total 480p sequences: {len(all_sequences)}")
    print("All sequence names: " + ", ".join(all_sequences))
    print(f"Training sequences: {len(train.sequences)}")
    print(f"Validation sequences: {len(val.sequences)}")
    print("Train sequence names: " + ", ".join(train.sequences))
    print("Validation sequence names: " + ", ".join(val.sequences))
    print(f"TRAIN/VAL SEQUENCE OVERLAP: {len(overlap)}")
    print(f"Training triplets: {len(train_data)}")
    print(f"Validation triplets: {len(val_data)}")
    print(f"Train sequences: {100 * len(train.sequences) / total_assigned_sequences:.2f}%")
    print(f"Validation sequences: {100 * len(val.sequences) / total_assigned_sequences:.2f}%")
    print(f"Train triplets: {100 * len(train_data) / total_triplets:.2f}%")
    print(f"Validation triplets: {100 * len(val_data) / total_triplets:.2f}%")
    print(f"TRAIN/VAL FRAME OVERLAP: {len(split_frame_overlap | frame_overlap)}")
    if overlap or split_frame_overlap or frame_overlap:
        raise RuntimeError("DAVIS split integrity failed")
    print("Split integrity: PASS")


if __name__ == "__main__":
    main()
