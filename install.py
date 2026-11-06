"""在独立虚拟环境中执行：python install.py [--accelerator cpu|cu121]。"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accelerator", choices=["cpu", "cu121"], default="cpu")
    args = parser.parse_args()
    if not (3, 10) <= sys.version_info[:2] <= (3, 12):
        parser.error("请使用 Python 3.10、3.11 或 3.12。")
    if sys.prefix == sys.base_prefix:
        parser.error("请先创建并激活虚拟环境，避免修改系统 Python。详见 README.md。")
    root = Path(__file__).resolve().parent
    def pip(*arguments):
        subprocess.run([sys.executable, "-m", "pip", *arguments], check=True, cwd=root)
    pip("install", "torch==2.5.1", "torchvision==0.20.1", "--index-url",
        f"https://download.pytorch.org/whl/{args.accelerator}")
    pip("install", "-r", str(root / "requirements.txt"))
    # 直接安装随包提供的官方源码，防止误装 PyPI 上的另一个实现。
    pip("install", "--no-deps", "--no-build-isolation", "-e", str(root / "vendor/yolov12"))
    subprocess.run([sys.executable, str(root / "main.py"), "doctor"], check=True, cwd=root)


if __name__ == "__main__":
    main()
