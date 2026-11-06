"""绘制明确标记的合成测试图；不模拟真实铁路故障分布。"""
from pathlib import Path

import cv2
import numpy as np
import yaml

NAMES = ["bolt_missing", "rod_damage", "cable_damage", "housing_damage"]


def make_demo(directory, seed=42):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    for split, total in (("train", 12), ("val", 8), ("test", 4)):
        image_dir = directory / "images" / split
        label_dir = directory / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for index in range(total):
            image = rng.integers(28, 65, (256, 256, 3), dtype=np.uint8)
            cv2.rectangle(image, (24, 68), (228, 208), (140, 145, 150), -1)
            cv2.rectangle(image, (70, 20), (90, 230), (95, 95, 95), -1)
            for x, y in ((40, 82), (210, 82), (40, 192), (210, 192)):
                cv2.circle(image, (x, y), 6, (45, 45, 45), -1)
            cls = index % 4
            x, y = int(rng.integers(95, 170)), int(rng.integers(100, 160))
            negative = split != "test" and index == total - 1
            if not negative:
                if cls == 0:
                    cv2.circle(image, (x, y), 12, (15, 15, 15), -1)
                    cv2.circle(image, (x, y), 17, (200, 200, 200), 2)
                elif cls == 1:
                    cv2.line(image, (x-18, y-18), (x+18, y+18), (30, 30, 220), 8)
                    cv2.line(image, (x-18, y+18), (x+18, y-18), (30, 30, 220), 8)
                elif cls == 2:
                    cv2.line(image, (x-18, y), (x+18, y), (20, 180, 230), 10)
                    cv2.line(image, (x, y-18), (x, y+18), (30, 30, 30), 5)
                else:
                    pts = np.array([[x-20,y-12],[x-4,y+4],[x+4,y-10],[x+20,y+18]])
                    cv2.polylines(image, [pts], False, (15, 15, 15), 5)
            cv2.putText(image, f"SYNTHETIC {split} {index}", (8, 245),
                        cv2.FONT_HERSHEY_SIMPLEX, .40, (255, 255, 255), 1)
            name = f"demo_{index:03d}"
            encoded_ok, encoded = cv2.imencode(".jpg", image)
            if not encoded_ok:
                raise OSError("无法保存演示图片。")
            (image_dir / f"{name}.jpg").write_bytes(encoded.tobytes())
            label = "" if negative else f"{cls} {x/256:.6f} {y/256:.6f} {48/256} {48/256}\n"
            (label_dir / f"{name}.txt").write_text(label, encoding="utf-8")
    config = {"path": ".", "train": "images/train", "val": "images/val",
              "test": "images/test", "names": NAMES}
    filename = directory / "data.yaml"
    filename.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    # MJPG/AVI 在本项目已测 CPU 环境可写；真实输入可以是 MP4。
    video = directory / "synthetic.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 6, (256, 256))
    if not writer.isOpened():
        raise RuntimeError("OpenCV 无法创建 MJPG 视频，请检查安装。")
    try:
        images = sorted((directory / "images/test").glob("*.jpg"))
        for image in images:
            frame = cv2.imdecode(np.frombuffer(image.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
            for _ in range(3):
                writer.write(frame)
    finally:
        writer.release()
    return filename


def verify_demo_outputs(directory):
    """独立重读输出，确认 JPG 和视频确实能解码，而非仅生成了文件名。"""
    import json
    directory = Path(directory)
    image_records = [json.loads(line) for line in (directory / "images/detections.jsonl").read_text(encoding="utf-8").splitlines()]
    video_records = [json.loads(line) for line in (directory / "video/detections.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(image_records) != 4 or len(video_records) != 12:
        raise RuntimeError("JSONL 图片/帧数异常。")
    for record in image_records:
        image = cv2.imread(record["visualization"])
        if image is None or image.shape[:2] != (256, 256):
            raise RuntimeError("演示标注图片不可读取或尺寸异常。")
    if [r["frame_index"] for r in video_records] != list(range(12)):
        raise RuntimeError("视频 JSONL 帧号异常。")
    cap = cv2.VideoCapture(video_records[0]["visualization"])
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape[:2] != (256, 256):
                raise RuntimeError("输出视频尺寸异常。")
            count += 1
    finally:
        cap.release()
    if count != 12:
        raise RuntimeError(f"输出视频只能解码 {count}/12 帧。")
    return {"readable_images": 4, "decoded_video_frames": count, "jsonl_records": 16}
