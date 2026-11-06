"""训练、评估与流式推理实现。"""
import json
from pathlib import Path

import cv2
import yaml

from .dataset import IMAGE_EXTENSIONS, check_dataset, read_config
from .runtime import ROOT, VENDOR, load_yolo, model_file, new_run, select_device

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def prepare_data(data_file, output, checked=None):
    config, summary = checked if checked is not None else check_dataset(data_file)
    filename = output / "data.resolved.yaml"
    filename.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(output / "dataset_check.json", summary)
    return filename, config


def train(args):
    # 检查通过后再分配运行目录，防止错误输入留下假训练结果。
    checked = check_dataset(args.data)
    weights = model_file(args.weights) if args.weights else None
    if weights and weights.suffix.lower() != ".pt":
        raise ValueError("迁移学习权重须为 .pt 文件。")
    YOLO = load_yolo(args.threads)
    device = select_device(args.device)
    output = new_run(args.output)
    data_file, config = prepare_data(args.data, output, checked)
    architecture = yaml.safe_load((VENDOR / "ultralytics/cfg/models/v12/yolov12.yaml").read_text(encoding="utf-8"))
    architecture["scale"] = "n"
    architecture["nc"] = config["nc"]
    architecture["names"] = config["names"]
    architecture_file = output / "yolov12n.yaml"
    architecture_file.write_text(yaml.safe_dump(architecture, sort_keys=False), encoding="utf-8")
    model = YOLO(str(architecture_file), task="detect")
    if weights:
        # 始终从 v12 架构创建，再迁移匹配权重，避免误将其他 YOLO 作为训练架构。
        model.load(str(weights))
    model.train(data=str(data_file), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=device, workers=0, seed=args.seed, deterministic=True,
                project=str(output.parent), name=output.name, exist_ok=True,
                pretrained=False, amp=False, optimizer="AdamW", lr0=0.001,
                patience=args.patience, cache=False, plots=False, save=True,
                val=True, mosaic=0.0, mixup=0.0, copy_paste=0.0, close_mosaic=0,
                fliplr=0.0, flipud=0.0, verbose=False)
    best = output / "weights/best.pt"
    if not best.is_file():
        raise RuntimeError("训练结束但 best.pt 未生成，请查看训练日志。")
    write_json(output / "run_info.json", {"task": "visible_defect_detection",
               "weights": str(best), "names": config["names"], "device": device,
               "epochs_requested": args.epochs, "seed": args.seed,
               "initial_weights": str(weights) if weights else "random_initialization"})
    print(f"训练完成，权重：{best}")
    return best


def trained_model(weights, threads, names=None):
    path = model_file(weights)
    if path.suffix.lower() != ".pt":
        raise ValueError("评估和推理必须使用训练后的 .pt 权重。")
    YOLO = load_yolo(threads)
    model = YOLO(str(path), task="detect")
    if model.task != "detect" or "A2C2f" not in {type(m).__name__ for m in model.model.modules()}:
        raise ValueError("权重必须是 YOLOv12 目标检测模型。")
    actual = [model.names[i] for i in range(len(model.names))]
    if names is not None and actual != names:
        raise ValueError(f"权重类别与数据集不一致：权重={actual}，数据集={names}")
    return model


def validate(args):
    checked = check_dataset(args.data)
    if args.split not in checked[0]:
        raise ValueError(f"数据配置没有 {args.split} 集合。")
    model = trained_model(args.weights, args.threads, checked[1]["names"])
    device = select_device(args.device)
    output = new_run(args.output)
    data_file, config = prepare_data(args.data, output, checked)
    metrics = model.val(data=str(data_file), split=args.split, imgsz=args.imgsz,
                        batch=args.batch, device=device, workers=0, plots=False,
                        half=False, project=str(output.parent), name=output.name, exist_ok=True)
    per_class = []
    # 只报告实际参与评估的类别；不把 absent class 的默认填充值当作真实指标。
    for index, cls in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(index)
        per_class.append({"class_id": int(cls), "name": config["names"][int(cls)],
                          "precision": float(p), "recall": float(r),
                          "mAP50": float(ap50), "mAP50_95": float(ap)})
    report = {"split": args.split, "metrics": {k: float(v) for k, v in metrics.results_dict.items()},
              "per_class": per_class, "note": "指标仅反映当前评估数据；合成数据不能证明真实检测效果。"}
    write_json(output / "metrics.json", report)
    print(f"评估结果：{output / 'metrics.json'}")
    return report


def prediction_record(result, source, frame_index, fps, visual_path):
    boxes = result.boxes
    detections = []
    if boxes is not None:
        for xyxy, cls, conf in zip(boxes.xyxy.cpu().tolist(), boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist()):
            detections.append({"class_id": int(cls), "class_name": result.names[int(cls)],
                               "confidence": round(conf, 6), "xyxy": [round(v, 2) for v in xyxy]})
    return {"source": str(source), "frame_index": frame_index,
            "timestamp_seconds": round(frame_index / fps, 6) if fps else None,
            "image_size": {"height": result.orig_shape[0], "width": result.orig_shape[1]},
            "status": "defect_detected" if detections else "no_defect_detected",
            "detections": detections, "visualization": str(visual_path)}


def predict(args):
    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        raise ValueError(f"输入不存在：{source}")
    files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS) if source.is_dir() else [source]
    if not files or any(p.suffix.lower() not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS for p in files):
        raise ValueError("输入须为图片、视频，或包含它们的目录。")
    config = read_config(getattr(args, "data", ROOT / "configs/point_machine.yaml"))
    model = trained_model(args.weights, args.threads, config["names"])
    device = select_device(args.device)
    output = new_run(args.output)
    total_frames, total_detections = 0, 0
    def infer(frame):
        return model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                             device=device, half=False, verbose=False, max_det=100)[0]
    with (output / "detections.jsonl").open("w", encoding="utf-8") as stream:
        for file_index, filename in enumerate(files):
            print(f"处理 {file_index+1}/{len(files)}：{filename.name}", flush=True)
            name = f"{file_index:04d}_{filename.stem}"
            if filename.suffix.lower() in IMAGE_EXTENSIONS:
                frame = cv2.imread(str(filename))
                if frame is None:
                    raise ValueError(f"无法读取图片：{filename}")
                result = infer(frame)
                visual = output / f"{name}.jpg"
                if not cv2.imwrite(str(visual), result.plot()):
                    raise OSError(f"无法写入：{visual}")
                record = prediction_record(result, filename, 0, None, visual)
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                total_frames += 1
                total_detections += len(record["detections"])
            else:
                cap = cv2.VideoCapture(str(filename))
                writer = None
                frame_index = 0
                visual = output / f"{name}.avi"
                try:
                    if not cap.isOpened():
                        raise ValueError(f"无法打开视频：{filename}")
                    fps = float(cap.get(cv2.CAP_PROP_FPS))
                    if not 0 < fps < 1000:
                        raise ValueError(f"视频帧率无效：{filename}")
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        if writer is None:
                            h, w = frame.shape[:2]
                            writer = cv2.VideoWriter(str(visual), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
                            if not writer.isOpened():
                                raise RuntimeError("无法创建 MJPG/AVI 输出视频。")
                        result = infer(frame)
                        writer.write(result.plot())
                        record = prediction_record(result, filename, frame_index, fps, visual)
                        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                        total_detections += len(record["detections"])
                        total_frames += 1
                        frame_index += 1
                        if frame_index % 100 == 0:
                            print(f"  已处理 {frame_index} 帧", flush=True)
                    if not frame_index:
                        raise ValueError(f"视频没有可解码帧：{filename}")
                    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if expected > 0 and frame_index < expected:
                        raise ValueError(f"视频提前解码结束：读取 {frame_index}/{expected} 帧：{filename}")
                finally:
                    cap.release()
                    if writer is not None:
                        writer.release()
    summary = {"files": len(files), "frames": total_frames, "detections": total_detections,
               "confidence_threshold": args.conf, "weights": str(Path(args.weights).resolve()),
               "note": "no_defect_detected 表示未检出目标，不等于设备正常。"}
    write_json(output / "summary.json", summary)
    print(f"推理完成：{total_frames} 张图片/视频帧，结果：{output}")
    return summary
