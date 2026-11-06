"""固定加载随包附带的官方实现，并限制 CPU 线程数量。"""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "yolov12"
COMMIT = "64d5ffc6db35a1bde446689ab8450ecd43475e00"


def load_yolo(threads=2):
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("MPLBACKEND", "Agg")
    # 上游检查 Arial 字体；使用 matplotlib 自带 DejaVuSans 作为本地替代，避免下载。
    import matplotlib
    config_dir = Path(os.environ["YOLO_CONFIG_DIR"])
    config_dir.mkdir(parents=True, exist_ok=True)
    font_cache = config_dir / "Arial.ttf"
    if not font_cache.exists():
        shutil.copyfile(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf", font_cache)
    if not (VENDOR / "ultralytics" / "__init__.py").is_file():
        raise RuntimeError("缺少 vendor/yolov12 官方源码，请重新解压完整项目。")
    sys.path.insert(0, str(VENDOR))
    import torch
    import ultralytics
    if not Path(ultralytics.__file__).resolve().is_relative_to(VENDOR):
        raise RuntimeError("加载了其他 ultralytics 安装，请在新 Python 进程中运行本项目。")
    torch.set_num_threads(threads)
    from ultralytics import YOLO, settings
    from ultralytics import utils
    from ultralytics.utils import torch_utils
    # 上游 select_device 会重新设置线程数，保持其设置和 CLI 参数一致。
    utils.NUM_THREADS = threads
    torch_utils.NUM_THREADS = threads
    settings.update({"sync": False})
    return YOLO


def select_device(value):
    import torch
    if value == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if value == "cpu":
        return value
    if not value.isdigit() or not torch.cuda.is_available():
        raise ValueError("device 应为 auto、cpu 或可用 GPU 编号，例如 0。")
    if int(value) >= torch.cuda.device_count():
        raise ValueError(f"GPU {value} 不存在。")
    return value


def model_file(value):
    """只接受实际存在的文件；不悄悄联网下载其他模型。"""
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() not in {".pt", ".yaml", ".yml"}:
        raise ValueError(f"模型文件不存在或格式不支持：{path}")
    return path


def new_run(value):
    path = Path(value).expanduser().resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"输出目录已非空，请换一个 --output，防止覆盖：{path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def new_numbered_run(value):
    """为默认演示原子创建新目录，保留已有结果及并发运行的输出。"""
    base = Path(value).expanduser().resolve()
    base.parent.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        candidate = base if number == 1 else base.with_name(f"{base.name}{number}")
        try:
            candidate.mkdir()
        except FileExistsError:
            number += 1
        else:
            return candidate
