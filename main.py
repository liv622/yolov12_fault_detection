#!/usr/bin/env python3
"""转辙机可见故障检测：python main.py --help。"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须为正整数。")
    return number


def probability(value):
    number = float(value)
    if not 0 < number <= 1:
        raise argparse.ArgumentTypeError("必须在 (0,1] 范围内。")
    return number


def image_size(value):
    size = positive_int(value)
    if size < 128 or size % 32:
        raise argparse.ArgumentTypeError("imgsz 必须是不小于 128 的 32 倍数。")
    return size


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--debug", action="store_true", help="出错时显示完整堆栈")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="检查依赖、官方源码路径、GPU 和模型构建")
    doctor.add_argument("--threads", type=positive_int, default=2)
    check = sub.add_parser("check", help="训练前严格检查数据")
    check.add_argument("--data", default=str(ROOT / "configs/point_machine.yaml"))
    train = sub.add_parser("train", help="在自定义数据上训练 YOLOv12n")
    val = sub.add_parser("val", help="评估训练权重")
    predict = sub.add_parser("predict", help="图片/视频检测并输出标注图和 JSONL")
    demo = sub.add_parser("demo", help="生成合成数据并实际训练、验证、图片/视频推理")
    for command in (train, val):
        command.add_argument("--data", default=str(ROOT / "configs/point_machine.yaml"))
        command.add_argument("--batch", type=positive_int, default=4)
    for command in (train, val, predict, demo):
        command.add_argument("--output", default=str(ROOT / "runs" / command.prog.split()[-1]))
        command.add_argument("--device", default="auto", help="auto、cpu 或单个 GPU 编号")
        command.add_argument("--threads", type=positive_int, default=2, help="CPU 线程数")
        command.add_argument("--imgsz", type=image_size, default=128 if command is demo else 640)
    train.add_argument("--weights", help="可选：作者发布的 YOLOv12 预训练 .pt 权重")
    train.add_argument("--epochs", type=positive_int, default=100)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--patience", type=positive_int, default=30)
    val.add_argument("--weights", required=True)
    val.add_argument("--split", choices=["val", "test"], default="val")
    predict.add_argument("--weights", required=True)
    predict.add_argument("--data", default=str(ROOT / "configs/point_machine.yaml"), help="用于核对权重类别；无需图片存在")
    predict.add_argument("--source", required=True, help="本地图片、视频或目录")
    predict.add_argument("--conf", type=probability, default=0.25)
    predict.add_argument("--iou", type=probability, default=0.45)
    demo.add_argument("--epochs", type=positive_int, default=1)
    return parser


def doctor(args):
    from point_machine.runtime import COMMIT, VENDOR, load_yolo
    YOLO = load_yolo(args.threads)
    import cv2
    import torch
    import torchvision
    from torchvision.ops import nms
    # 校验 torch/torchvision 二进制是否匹配，而非只做 import。
    selected = nms(torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.]]), torch.tensor([.9, .8]), .5)
    if selected.tolist() != [0]:
        raise RuntimeError("torchvision NMS 自检失败。")
    model = YOLO(str(VENDOR / "ultralytics/cfg/models/v12/yolov12.yaml"), task="detect")
    result = model.predict(__import__("numpy").zeros((128, 128, 3), dtype="uint8"),
                           imgsz=128, device="cpu", verbose=False)
    if len(result) != 1:
        raise RuntimeError("YOLOv12 CPU 前向推理自检失败。")
    print(json.dumps({"python": sys.version.split()[0], "torch": torch.__version__,
                      "torchvision": torchvision.__version__, "opencv": cv2.__version__,
                      "cuda_available": torch.cuda.is_available(), "source": str(VENDOR),
                      "upstream_commit": COMMIT, "cpu_forward_and_nms": "PASS"}, indent=2))


def demo(args):
    from point_machine.demo import make_demo, verify_demo_outputs
    from point_machine.pipeline import predict, train, validate, write_json
    from point_machine.runtime import new_run
    output = new_run(args.output)
    data = make_demo(output / "dataset")
    common = {"device": args.device, "threads": args.threads, "imgsz": args.imgsz}
    best = train(SimpleNamespace(**common, data=str(data), output=str(output / "train"),
                                  weights=None, epochs=args.epochs, batch=2, seed=42, patience=5))
    metrics = validate(SimpleNamespace(**common, data=str(data), output=str(output / "validation"),
                                      weights=str(best), batch=2, split="test"))
    images = predict(SimpleNamespace(**common, data=str(data), weights=str(best), conf=0.25, iou=0.45,
                                    source=str(output / "dataset/images/test"), output=str(output / "images")))
    video = predict(SimpleNamespace(**common, data=str(data), weights=str(best), conf=0.25, iou=0.45,
                                   source=str(output / "dataset/synthetic.avi"), output=str(output / "video")))
    if images["frames"] != 4 or video["frames"] != 12:
        raise RuntimeError("演示输出的图片/视频帧数不符合预期。")
    output_checks = verify_demo_outputs(output)
    report = {"status": "PASS", "data": "SYNTHETIC_ONLY", "epochs": args.epochs,
              "image_frames": images["frames"], "video_frames": video["frames"],
              "weights": str(best), "test_metrics": metrics["metrics"], "output_checks": output_checks,
              "note": "演示仅验证代码执行，不证明转辙机故障检测精度。"}
    write_json(output / "demo_report.json", report)
    print(f"演示通过，报告：{output / 'demo_report.json'}")


def main():
    args = build_parser().parse_args()
    try:
        if args.command == "doctor":
            doctor(args)
        elif args.command == "check":
            from point_machine.dataset import check_dataset
            print(json.dumps(check_dataset(args.data)[1], indent=2, ensure_ascii=False))
        elif args.command == "demo":
            demo(args)
        else:
            from point_machine import pipeline
            getattr(pipeline, {"val": "validate"}.get(args.command, args.command))(args)
    except Exception as exc:
        if args.debug:
            raise
        print(f"错误：{exc}\n可在子命令前添加 --debug 查看完整堆栈。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
