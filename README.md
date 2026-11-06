# YOLOv12 转辙机可见故障检测（完整可运行项目）

适用：图片/视频中能直接看见、能画出边界框的转辙机缺陷。提供官方模型源码、训练、评估、图片/视频推理、严格数据检查、合成演示及测试。

**当前没有真实转辙机数据。随包演示权重只用合成图训练，用于验证代码运行，不能识别真实故障。** 图片检测也不能直接判断内部电气故障、动作电流异常、接点状态或设备整体是否安全。实际使用前，需要采集并标注自己的真实数据，训练后在独立测试集评估。

## 1. 项目使用的 YOLOv12

- 官方作者源码：[sunsmarterjie/yolov12](https://github.com/sunsmarterjie/yolov12)。
- 固定提交：`64d5ffc6db35a1bde446689ab8450ecd43475e00`，已附在 `vendor/yolov12/`，无需再次克隆。
- 网络：官方 `yolov12.yaml` 的 nano 配置（YOLOv12-turbo n）；只改类别数，不改主干、注意力模块和检测头结构。
- 代码中的 `from ultralytics import YOLO` 指向随包附带的作者实现。请用本项目入口运行。
- Python 3.10～3.12；当前实测环境与检查结果见 `TEST_REPORT.md`。CPU 为默认安装方案。GPU 安装入口已提供，当前环境没有 GPU，未做 GPU 实测。

上游 `requirements.txt` 含特定 Python 3.11/CUDA 的 FlashAttention wheel，以及本项目不需要的部署和网页依赖，因此本项目提供独立的固定版本依赖。官方源码支持没有 FlashAttention 时使用 PyTorch SDPA，CPU 可直接运行；此时提示 `FlashAttention is not available ...` 是正常回退提示。

## 2. 安装

在解压后的项目根目录打开终端。建议 64 位 Python 3.11 或 3.12。

Linux / macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
python install.py
```

Windows PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe install.py
```

Windows 后续命令中的 `python` 可直接替换为 `.\.venv\Scripts\python.exe`。不必更改 PowerShell 执行策略。

CPU 安装从 PyTorch 官方 CPU 源获取 torch/torchvision，其他依赖走 pip 当前配置的镜像。首次需要网络和磁盘空间，CPU 包也有数百 MB。安装完成后不需下载额外模型即可运行演示。

有 NVIDIA GPU 且驱动支持 CUDA 12.1 时，可在新的虚拟环境使用：

```bash
python install.py --accelerator cu121
python main.py doctor
```

Apple Silicon 本项目按 CPU 路径运行；Windows、macOS、CUDA 的运行情况未在当前机器实测。默认 `--device auto` 有 CUDA 就用 GPU 0，否则用 CPU；可明确设置 `--device cpu` 或 `--device 0`。

## 3. 一键演示：实际训练后再推理

```bash
python main.py demo --device cpu
```

默认生成 24 张带有 `SYNTHETIC` 标记的合成图（训练 12、验证 8、测试 4），训练 1 个 epoch，独立测试集评估，再检测 4 张图片和一段 12 帧的视频。演示不是一份凭空写出的推理输出，而是调用官方 YOLOv12 完整执行。

默认输出：

```text
runs/demo/
├── dataset/                 合成图片、标签、data.yaml、synthetic.avi
├── train/weights/best.pt     合成演示权重，不用于真实检测
├── train/results.csv        每轮训练/验证记录
├── validation/metrics.json  独立测试集指标
├── images/                  标注后的图片、detections.jsonl、summary.json
├── video/                   标注后的 AVI、逐帧 detections.jsonl、summary.json
└── demo_report.json         流程执行检查结果
```

输出目录非空时程序会拒绝覆盖。重复演示请换目录：

```bash
python main.py demo --device cpu --output runs/demo_second
```

1 轮训练的检测结果通常为空，指标可能为 0，这是演示的合理结果；演示通过只说明代码链路执行成功。

项目附带的 `example_output/` 是本次实际运行的产物副本，方便直接查看图片、视频、日志和演示权重。其 JSON/日志保留测试机器的原始绝对路径；在自己的机器重新执行 `demo` 会生成对应本机的新路径。现成结果不放入 `runs/demo`，首次演示可直接运行。

## 4. 换成你自己的转辙机数据

默认定义 4 类示例故障，可按设备和采集规范修改 `configs/point_machine.yaml`：

| ID | 英文标识 | 需要拍到的可见证据 |
|---|---|---|
| 0 | bolt_missing | 预期螺栓安装位置确实缺失，且位置可辨认 |
| 1 | rod_damage | 连接杆可见断裂、变形 |
| 2 | cable_damage | 线缆绝缘层或连接部位可见破损 |
| 3 | housing_damage | 壳体可见裂纹、缺口、破损 |

这是待你确认和标注的类别方案，不是已经训练好的能力。不把仅凭图片无法确认的“螺栓松动”直接作为确定故障。缺失类需统一标注预期安装区域，不能给完全不可见的对象随意画框；若视角无法定位，应排除该图或增加合适的采集视角。

建立目录：

```text
datasets/point_machine/
├── images/
│   ├── train/    训练图片
│   ├── val/      验证图片
│   └── test/     最终测试图片
└── labels/
    ├── train/    与图片同名的 txt
    ├── val/
    └── test/
```

可用标注工具导出 **YOLO detection** 格式。每张图对应一个同名 txt；每个框独立一行：

```text
class_id x_center y_center width height
```

例如图片 `a.jpg` 的 `a.txt`：

```text
1 0.500000 0.500000 0.200000 0.300000
2 0.250000 0.600000 0.100000 0.150000
```

坐标均除以图片宽/高，归一化到 `[0,1]`。正常图片保留一个**空 txt**，不增加 `normal` 检测类别。类别必须从 0 连续编号，修改类别后需同步所有标签和重新训练。

建议按设备编号、地点、采集批次划分训练/验证/测试（可从约 7:2:1 起步），不能把同一段视频相邻帧随机分到不同集合。代码会检查像素完全相同的跨集合图片，但不能自动判断所有近似重复图。训练集须覆盖每个类别；评估集要尽量覆盖真实光照、角度、遮挡及各类缺陷。

YAML 的 `path` 相对 **YAML 文件所在目录**解析，本项目会生成绝对路径配置再交给官方训练器。目录需使用 `images/train`、`images/val`、`images/test` 及相应 `labels` 结构。若暂时没有独立测试集，可以从 YAML 删除 `test`，训练及 `val --split val` 仍可运行。

先检查：

```bash
python main.py check --data configs/point_machine.yaml
```

检查内容包含可读图片、缺失标签、孤立标签、类别越界、NaN/Inf、边界框越界、训练类别覆盖和跨集合精确重复。

## 5. 训练自己的故障检测权重

推荐迁移学习：先从作者官方 [YOLOv12n-turbo 权重](https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12n.pt) 下载 COCO 预训练权重，保存到自己创建的 `weights/yolov12n.pt`。

```bash
python main.py train --data configs/point_machine.yaml --weights weights/yolov12n.pt --epochs 100 --imgsz 640 --batch 4 --device 0 --output runs/point_machine_v1
```

没有 GPU 时将 `--device 0` 改成 `--device cpu`，但正式训练通常会慢得多。显存不足可先减小 `--batch`，小缺陷较多时保留足够分辨率。

也支持不下载预训练权重，从官方结构随机初始化开始训练：

```bash
python main.py train --data configs/point_machine.yaml --epochs 100 --imgsz 640 --batch 2 --device cpu --output runs/point_machine_scratch
```

示例默认 epoch、batch、图像尺寸只是起点，不是效果保证。默认关闭 AMP，使用保守的几何增强，避免自动 AMP 自检额外下载模型以及不合理翻转破坏设备朝向。训练早停可能在指定 epoch 之前结束。

主要产物：`weights/best.pt`、`weights/last.pt`、`results.csv`、`dataset_check.json`、`run_info.json`、`data.resolved.yaml`。

## 6. 独立评估

```bash
python main.py val --data configs/point_machine.yaml --weights runs/point_machine_v1/weights/best.pt --split test --device 0 --output runs/point_machine_test
```

输出 `metrics.json`，包含 precision、recall、mAP50、mAP50-95 及实际参与评估类别的指标。最终效果须用未参与调参的真实测试集判断；不能用演示图、训练集或单张截图证明有效。

## 7. 图片/视频故障检测

单张图片：

```bash
python main.py predict --weights runs/point_machine_v1/weights/best.pt --source your_image.jpg --conf 0.25 --device cpu --output runs/predict_image
```

图片文件夹或视频：

```bash
python main.py predict --weights runs/point_machine_v1/weights/best.pt --source your_images --output runs/predict_folder
python main.py predict --weights runs/point_machine_v1/weights/best.pt --source your_video.mp4 --output runs/predict_video
```

改过类别或使用另一份数据配置时，推理也传入相同的 `--data your_data.yaml`。推理只读取配置核对类别，不要求训练图片还在。代码会拒绝类别不匹配的权重，避免误把 COCO 通用模型当作转辙机故障模型。

输出包含边框和英文类别标注的 JPG/AVI、逐图片/逐帧 JSONL 和汇总 JSON。视频逐帧处理，不把整段视频留在内存；输出使用 MJPG/AVI，优先保证 CPU 环境可写。

`detections.jsonl` 每行是一张图片或一帧，记录来源、从 0 开始的帧号、视频时间、尺寸、检测框、置信度、可视化路径：

```json
{"status":"defect_detected","detections":[{"class_id":1,"class_name":"rod_damage","confidence":0.91,"xyxy":[100,120,160,190]}]}
```

以上仅为字段示意，不是实测预测。`xyxy` 为原图像素坐标。`no_defect_detected` 只表示在当前阈值下未检出，不能等同于设备正常。视频每帧的框独立预测，没有跨帧跟踪和报警去重。

## 8. 自检与常见问题

```bash
python main.py doctor
python -m unittest discover -s tests -v
python main.py --help
```

- `doctor` 检查源码路径、torch/torchvision NMS 二进制兼容性，并实际执行 YOLOv12 CPU 前向推理。
- 报错时把 `--debug` 放在子命令前，例如 `python main.py --debug train ...`。
- `No module named ...`：确认正在使用创建项目环境的 Python，再执行 `python install.py`。
- `operator torchvision::nms does not exist`：通常是 torch 与 torchvision 来自不同构建，重新建环境并用本项目安装脚本。
- 提示找不到图片：先按第 4 节放入真实图片，或直接运行 `demo`；模板中没有真实数据。
- 只看到空检测框：检查是否误用了演示权重、真实训练是否充分、类别与标注是否一致，以及 `--conf` 是否合适。降低阈值产生框不能证明模型有效。
- 输出目录已有内容：改 `--output`，程序不自动删除旧训练结果。

## 9. 文件与许可证

```text
main.py                命令行入口
install.py             独立环境安装与自检
requirements.txt       固定版本运行依赖（torch 单独安装）
point_machine/         数据检查、训练、推理、演示实现
configs/               真实数据配置模板
tests/                 数据校验回归测试
vendor/yolov12/        固定版本官方源码和上游 LICENSE
TEST_REPORT.md         当前环境的实际运行记录
```

本项目新增代码采用与附带上游源码相同的 AGPL-3.0 许可，完整条款见 `LICENSE`；上游版权和许可保留于 `vendor/yolov12/`。安装时生成的字体缓存是 matplotlib 自带 DejaVuSans 的本地副本，用于避免上游字体检查联网下载。
