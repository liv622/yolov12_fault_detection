"""重复演示应创建新目录，显式输出目录继续保护旧结果。"""
import tempfile
import unittest
from pathlib import Path

from point_machine.runtime import new_numbered_run, new_run


class OutputDirectoryTests(unittest.TestCase):
    def test_repeated_demo_preserves_previous_results(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "runs/demo"
            base.mkdir(parents=True)
            original = base / "demo_report.json"
            original.write_text('{"status":"PASS"}', encoding="utf-8")
            # 目录或文件已占用相同名字时都应跳过。
            occupied = base.with_name("demo2")
            occupied.write_text("keep", encoding="utf-8")
            first = new_numbered_run(base)
            second = new_numbered_run(base)
            self.assertEqual(first.name, "demo3")
            self.assertEqual(second.name, "demo4")
            self.assertTrue(first.is_dir() and second.is_dir())
            self.assertEqual(original.read_text(encoding="utf-8"), '{"status":"PASS"}')
            self.assertEqual(occupied.read_text(encoding="utf-8"), "keep")

    def test_explicit_output_cannot_overwrite_existing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "chosen_output"
            output.mkdir()
            result = output / "existing.txt"
            result.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "输出目录已非空"):
                new_run(output)
            self.assertEqual(result.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
