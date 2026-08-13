import tempfile
import unittest
from pathlib import Path

from utils.davis import load_davis_train_val, resolve_davis_root


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

    def test_missing_metadata_explains_how_to_fix_it(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "complete DAVIS trainval package"):
                load_davis_train_val(directory)


if __name__ == "__main__":
    unittest.main()
