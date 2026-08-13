#!/usr/bin/env python3
"""Audit official DAVIS train/validation isolation at sequence and frame level."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from net.phasenet import Triplets
from utils.davis import davis_image_root, load_davis_train_val, resolve_davis_root


def frame_paths(dataset):
    return {Path(item[0]).resolve() for triplet in dataset.sample for item in triplet}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--davis-root", required=True)
    parser.add_argument("--dataset-path", default=None)
    args = parser.parse_args()

    root = resolve_davis_root(args.davis_root, args.dataset_path)
    image_root = davis_image_root(root, args.dataset_path)
    all_sequences = sorted(p.name for p in image_root.iterdir() if p.is_dir())
    train, val = load_davis_train_val(root)
    overlap = set(train) & set(val)
    train_data = Triplets(str(image_root), allowed_sequences=train)
    val_data = Triplets(str(image_root), allowed_sequences=val)
    frame_overlap = frame_paths(train_data) & frame_paths(val_data)
    total_assigned_sequences = len(train) + len(val)
    total_triplets = len(train_data) + len(val_data)

    print(f"DAVIS root: {root}")
    print(f"Image root: {image_root}")
    print(f"Total 480p sequences: {len(all_sequences)}")
    print("All sequence names: " + ", ".join(all_sequences))
    print(f"Train sequence count: {len(train)}")
    print(f"Validation sequence count: {len(val)}")
    print("Train sequence names: " + ", ".join(train))
    print("Validation sequence names: " + ", ".join(val))
    print(f"TRAIN/VAL SEQUENCE OVERLAP: {len(overlap)}")
    print(f"Generated train triplets: {len(train_data)}")
    print(f"Generated validation triplets: {len(val_data)}")
    print(f"Train sequences: {100 * len(train) / total_assigned_sequences:.2f}%")
    print(f"Validation sequences: {100 * len(val) / total_assigned_sequences:.2f}%")
    print(f"Train triplets: {100 * len(train_data) / total_triplets:.2f}%")
    print(f"Validation triplets: {100 * len(val_data) / total_triplets:.2f}%")
    print(f"TRAIN/VAL FRAME OVERLAP: {len(frame_overlap)}")
    if overlap or frame_overlap:
        raise RuntimeError("DAVIS split integrity failed")
    print("Split integrity: PASS")


if __name__ == "__main__":
    main()
