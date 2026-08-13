import tempfile
import unittest
from pathlib import Path
import subprocess
import importlib.util

from utils.davis import (load_davis_split_info, load_davis_train_val,
                         parse_davis_split, resolve_davis_root)


class DavisSplitTest(unittest.TestCase):
    def test_resolve_root_from_image_directories(self):
        root = Path(tempfile.gettempdir()) / "DAVIS"
        self.assertEqual(resolve_davis_root(dataset_path=root / "JPEGImages" / "480p"), root)
        self.assertEqual(resolve_davis_root(dataset_path=root / "JPEGImages"), root)

    def test_separate_imageset_root(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory)
            split_dir = metadata / "ImageSets" / "2017"
            split_dir.mkdir(parents=True)
            (split_dir / "train.txt").write_text("bear\nblackswan\n")
            (split_dir / "val.txt").write_text("breakdance\n")
            train, val = load_davis_train_val("/images/elsewhere", metadata)
            self.assertEqual(train, ["bear", "blackswan"])
            self.assertEqual(val, ["breakdance"])

    def test_imageset_root_accepts_image_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "DAVIS"
            split_dir = root / "ImageSets" / "2017"
            split_dir.mkdir(parents=True)
            (split_dir / "train.txt").write_text("bear\n")
            (split_dir / "val.txt").write_text("breakdance\n")

            train, val = load_davis_train_val(
                "/images/elsewhere", root / "JPEGImages" / "480p")

            self.assertEqual(train, ["bear"])
            self.assertEqual(val, ["breakdance"])

    def test_missing_metadata_explains_how_to_fix_it(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "complete DAVIS trainval package"):
                load_davis_train_val(directory)

    def test_480p_frame_list_extracts_and_deduplicates_sequences_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_dir = root / "ImageSets" / "480p"
            split_dir.mkdir(parents=True)
            (split_dir / "train.txt").write_text(
                "/JPEGImages/480p/bear/00000.jpg /Annotations/480p/bear/00000.png\n"
                "/JPEGImages/480p/bear/00001.jpg /Annotations/480p/bear/00001.png\n"
                "/JPEGImages/480p/blackswan/00000.jpg /Annotations/480p/blackswan/00000.png\n"
                "/JPEGImages/480p/bear/00002.jpg /Annotations/480p/bear/00002.png\n"
            )
            info = load_davis_split_info(root)
            self.assertEqual(info.sequences, ["bear", "blackswan"])
            self.assertEqual(len(info.frames), 4)
            self.assertEqual(info.path, (split_dir / "train.txt").resolve())

    def test_480p_is_preferred_over_year_sequence_list(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for version in ("480p", "2017"):
                (root / "ImageSets" / version).mkdir(parents=True)
            (root / "ImageSets" / "480p" / "train.txt").write_text(
                "/JPEGImages/480p/bear/00000.jpg /Annotations/480p/bear/00000.png\n")
            (root / "ImageSets" / "2017" / "train.txt").write_text("wrong\n")
            self.assertEqual(load_davis_split_info(root).sequences, ["bear"])

    def test_frame_list_rejects_full_path_as_sequence_and_bad_structure(self):
        with tempfile.TemporaryDirectory() as directory:
            split = Path(directory) / "train.txt"
            split.write_text(
                "/JPEGImages/720p/bear/00000.jpg /Annotations/480p/bear/00000.png\n")
            with self.assertRaisesRegex(RuntimeError, "expected /JPEGImages/480p"):
                parse_davis_split(split)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "audit requires PyTorch")
    def test_audit_reports_frame_list_counts_and_disjoint_triplets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "DAVIS"
            split_dir = root / "ImageSets" / "480p"
            image_root = root / "JPEGImages" / "480p"
            split_dir.mkdir(parents=True)
            splits = {"train": ("bear", 5), "val": ("dance", 4)}
            for split, (sequence, count) in splits.items():
                sequence_dir = image_root / sequence
                sequence_dir.mkdir(parents=True)
                rows = []
                for frame in range(count):
                    name = f"{frame:05}.jpg"
                    (sequence_dir / name).touch()
                    rows.append(
                        f"/JPEGImages/480p/{sequence}/{name} "
                        f"/Annotations/480p/{sequence}/{frame:05}.png\n")
                (split_dir / f"{split}.txt").write_text("".join(rows))
            result = subprocess.run(
                ["python", "tools/audit_davis_split.py", "--dataset-path", str(image_root)],
                cwd=Path(__file__).resolve().parents[1], text=True,
                capture_output=True, check=True)
            self.assertIn("Training frame entries: 5", result.stdout)
            self.assertIn("Validation frame entries: 4", result.stdout)
            self.assertIn("Training triplets: 3", result.stdout)
            self.assertIn("Validation triplets: 2", result.stdout)
            self.assertIn("TRAIN/VAL FRAME OVERLAP: 0", result.stdout)
            self.assertIn("Split integrity: PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
