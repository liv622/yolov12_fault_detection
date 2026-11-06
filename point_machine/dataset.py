"""严格检查 YOLO 检测标签、图片可读性、类别范围和跨集合重复数据。"""
import hashlib
import math
from pathlib import Path

import yaml
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_config(filename):
    filename = Path(filename).expanduser().resolve()
    with filename.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("数据 YAML 须为字典。")
    names = data.get("names")
    if isinstance(names, dict):
        if set(names) != set(range(len(names))):
            raise ValueError("names 的类别编号必须从 0 开始连续递增。")
        names = [names[i] for i in range(len(names))]
    if not isinstance(names, list) or not names or any(not isinstance(x, str) or not x.strip() for x in names):
        raise ValueError("names 必须为非空类别名称列表或编号字典。")
    if len(set(names)) != len(names):
        raise ValueError("类别名称不可重复。")
    # 使用 ASCII 类别名，避免推理标注字体及离线字体下载问题。
    if any(not x.isascii() for x in names):
        raise ValueError("names 请使用英文类别标识；中文含义可写在 YAML 注释或说明中。")
    if "nc" in data and data["nc"] != len(names):
        raise ValueError("nc 与 names 数量不一致。")
    root = Path(data.get("path", ".")).expanduser()
    if not root.is_absolute():
        root = filename.parent / root
    root = root.resolve()
    normalized = {"path": str(root), "names": names, "nc": len(names)}
    for split in ("train", "val", "test"):
        if split not in data:
            if split != "test":
                raise ValueError(f"数据配置缺少 {split}。")
            continue
        if not isinstance(data[split], str):
            raise ValueError(f"{split} 仅支持图片目录路径。")
        image_dir = Path(data[split]).expanduser()
        image_dir = (root / image_dir).resolve() if not image_dir.is_absolute() else image_dir.resolve()
        # 严格约定 images/<split>，避免官方图片->标签规则和自定义检查产生偏差。
        if image_dir != root / "images" / split:
            raise ValueError(f"{split} 必须指向 path 下的 images/{split} 目录。")
        normalized[split] = str(image_dir)
    return normalized


def label_rows(filename, nc):
    if not filename.is_file():
        raise ValueError(f"缺少标签：{filename}；正常图片也请提供空 txt。")
    boxes = []
    for line_no, line in enumerate(filename.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        prefix = f"{filename}:{line_no}"
        if len(parts) != 5:
            raise ValueError(f"{prefix}：每行必须是 class_id x_center y_center width height。")
        try:
            cls, x, y, w, h = map(float, parts)
        except ValueError as exc:
            raise ValueError(f"{prefix}：标签含非数字内容。") from exc
        if not all(math.isfinite(v) for v in (cls, x, y, w, h)):
            raise ValueError(f"{prefix}：标签含 NaN/Inf。")
        if not cls.is_integer() or not 0 <= cls < nc:
            raise ValueError(f"{prefix}：类别编号必须为 0~{nc-1} 的整数。")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"{prefix}：框坐标必须归一化到 [0,1]，宽高须大于 0。")
        if min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6:
            raise ValueError(f"{prefix}：标注框超出图片边界。")
        row = (int(cls), x, y, w, h)
        if row in boxes:
            raise ValueError(f"{prefix}：重复标注框。")
        boxes.append(row)
    return boxes


def check_dataset(filename):
    data = read_config(filename)
    root = Path(data["path"])
    seen_content = {}
    summary = {"names": data["names"], "splits": {}}
    for split in ("train", "val", "test"):
        if split not in data:
            continue
        image_dir = Path(data[split])
        images = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f"{split} 没有图片：{image_dir}")
        counts = [0] * data["nc"]
        negatives = 0
        expected_labels = set()
        for image in images:
            with Image.open(image) as opened:
                opened.verify()
            with Image.open(image) as opened:
                if min(opened.size) < 32:
                    raise ValueError(f"图片尺寸过小：{image}")
                # 对解码后的像素做哈希，也拦截改名或仅修改元数据的重复图片。
                rgb = opened.convert("RGB")
                digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            if digest in seen_content and seen_content[digest][0] != split:
                raise ValueError(f"数据泄漏：{image} 与 {seen_content[digest][1]} 重复，请按设备/采集批次划分。")
            seen_content[digest] = (split, image)
            label = root / "labels" / split / image.relative_to(image_dir).with_suffix(".txt")
            if label in expected_labels:
                raise ValueError(f"同名不同后缀图片共用了一个标签：{image}")
            expected_labels.add(label)
            boxes = label_rows(label, data["nc"])
            negatives += not boxes
            for cls, *_ in boxes:
                counts[cls] += 1
        actual_labels = set((root / "labels" / split).rglob("*.txt"))
        if actual_labels - expected_labels:
            raise ValueError(f"存在无对应图片的标签：{sorted(actual_labels - expected_labels)[0]}")
        if not any(counts):
            raise ValueError(f"{split} 全为空标签，至少需要一张有故障标注的图片。")
        if split == "train" and any(n == 0 for n in counts):
            raise ValueError(f"训练集缺少某些类别标注：{dict(zip(data['names'], counts))}")
        summary["splits"][split] = {"images": len(images), "negative_images": negatives,
                                      "instances": dict(zip(data["names"], counts))}
    return data, summary
