"""检查会污染训练结果或造成训练中断的数据边界。"""
import shutil
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from point_machine.dataset import check_dataset


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for split, color in (("train", (150, 50, 20)), ("val", (20, 100, 50))):
            (self.root / "images" / split).mkdir(parents=True)
            (self.root / "labels" / split).mkdir(parents=True)
            Image.new("RGB", (64, 64), color).save(self.root / "images" / split / "a.png")
            (self.root / "labels" / split / "a.txt").write_text("0 0.5 0.5 0.25 0.25\n")
        self.yaml = self.root / "data.yaml"
        self.yaml.write_text(yaml.safe_dump({"path": ".", "train": "images/train",
                                             "val": "images/val", "names": ["damage"]}))
        self.label = self.root / "labels/train/a.txt"

    def tearDown(self):
        self.temporary.cleanup()

    def test_valid_relative_paths(self):
        config, report = check_dataset(self.yaml)
        self.assertEqual(config["path"], str(self.root))
        self.assertEqual(report["splits"]["train"]["instances"], {"damage": 1})

    def test_empty_negative_label_allowed(self):
        Image.new("RGB", (64, 64), "black").save(self.root / "images/train/normal.png")
        (self.root / "labels/train/normal.txt").touch()
        self.assertEqual(check_dataset(self.yaml)[1]["splits"]["train"]["negative_images"], 1)

    def test_missing_label_rejected(self):
        self.label.unlink()
        with self.assertRaisesRegex(ValueError, "缺少标签"):
            check_dataset(self.yaml)

    def test_invalid_boxes_rejected(self):
        for text in ("1 0.5 0.5 0.2 0.2", "0 nan 0.5 0.2 0.2", "0 0.9 0.5 0.4 0.4",
                     "0 0.5 0.5 0 0.2", "0 0.5 0.5 0.2 0.2 0.1", "0.5 0.5 0.5 0.2 0.2"):
            with self.subTest(label=text):
                self.label.write_text(text)
                with self.assertRaises(ValueError):
                    check_dataset(self.yaml)

    def test_train_val_leakage_rejected(self):
        shutil.copyfile(self.root / "images/train/a.png", self.root / "images/val/a.png")
        with self.assertRaisesRegex(ValueError, "数据泄漏"):
            check_dataset(self.yaml)

    def test_corrupt_image_rejected(self):
        (self.root / "images/train/a.png").write_bytes(b"invalid image")
        with self.assertRaises(OSError):
            check_dataset(self.yaml)

    def test_orphan_label_rejected(self):
        (self.root / "labels/train/orphan.txt").touch()
        with self.assertRaisesRegex(ValueError, "无对应图片"):
            check_dataset(self.yaml)

    def test_empty_training_set_rejected(self):
        self.label.write_text("")
        with self.assertRaisesRegex(ValueError, "全为空标签"):
            check_dataset(self.yaml)


if __name__ == "__main__":
    unittest.main()
