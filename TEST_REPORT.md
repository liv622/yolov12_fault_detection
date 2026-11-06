# 实际运行测试报告

测试日期：2026-11-06（Asia/Shanghai）。下列结果来自实际执行，并非仅通过语法检查推断。

## 环境

| 项目 | 实测值 |
|---|---|
| 系统 | Linux x86_64 |
| CPU | Intel Core i7-13650HX |
| Python | 3.12.14 |
| PyTorch | 2.5.1+cpu |
| torchvision | 0.20.1+cpu |
| OpenCV | 4.10.0（opencv-python 4.10.0.84） |
| NumPy | 1.26.4 |
| 官方源码提交 | 64d5ffc6db35a1bde446689ab8450ecd43475e00 |
| 官方库版本 | 8.3.63（来自作者 YOLOv12 仓库） |
| GPU | 无；GPU 路径未实测 |

完整依赖版本记录在 `verification/tested-environment.txt`。

## 通过的检查

| 检查 | 实际结果 | 证据 |
|---|---|---|
| 全新虚拟环境安装 | 使用已下载的 wheel 缓存，完整执行 `install.py`，退出码 0 | `verification/clean_install.log` |
| 依赖一致性 | `pip check`：No broken requirements found | `verification/clean_install.log` |
| 官方模型 CPU 前向、torchvision NMS | PASS | `verification/doctor.log` |
| 数据校验测试 | 8 项全部通过，含多组异常标签子用例 | `verification/clean_install.log` |
| 从结构随机初始化训练 | 合成训练集 12 张、验证集 8 张，128 输入、batch 2、1 epoch，生成 best.pt 和 last.pt | `verification/demo.log` |
| 加载已有权重继续训练 | 从刚生成的演示权重初始化，另跑 1 epoch，退出码 0 | `verification/transfer_train.log` |
| 独立测试集评估 | 4 张图、4 个实例，生成 metrics.json，退出码 0 | `example_output/validation/metrics.json`（ZIP 内） |
| 图片推理 | 4 张输出图能重新解码，均为 256×256 | `example_output/demo_report.json`（ZIP 内） |
| 视频推理 | 输入 12 帧，输出 AVI 能逐帧解码 12 帧，尺寸 256×256 | 同上 |
| JSONL 输出 | 图片 4 行、视频 12 行，视频帧号为 0～11 | 同上 |
| 检测框序列化分支 | 低阈值专门触发 16 个候选框，验证类别、置信度和原图坐标范围 | `verification/inference_checks.json` |
| 类别不匹配 | 有意传入错误类别配置，按预期拒绝，退出码 1 | `verification/class_mismatch.log` |
| 防止覆盖已有结果 | 有意复用非空输出目录，按预期拒绝，原 summary.json 内容保持一致 | `verification/no_overwrite.log` |
| ZIP 完整性 | ZIP CRC 检查通过 | `verification/relocation_check.json` |
| 解压换目录运行 | 在新的含中文和空格的目录，用同一依赖环境重新执行完整 demo，通过训练、评估、4 张图及 12 帧视频推理 | `verification/relocation.log` |

低阈值候选框测试只用于覆盖代码分支，不作为检测准确率证据。两个退出码 1 的用例是有意输入错误的负向测试，其结果符合设计。

## 检测效果的实际边界

当前只用合成图执行 1 epoch 的流程测试。独立合成测试集的 precision、recall、mAP50、mAP50-95 **均为 0**。这份项目目前验证的是可运行的工程流程，不是已经训练好的真实转辙机故障检测器。

没有真实转辙机图片和标注，因此尚未验证真实识别能力；需要按 README 放入自己的数据，完成训练及独立测试。GPU、Windows、macOS 未在本次机器实测，不能承诺所有硬件和操作系统都零错误。

上游关于 FlashAttention 回退和默认 nano 规模的提示不属于执行错误。默认运行已关闭额外在线资源下载，字体使用 matplotlib 自带字体的本地缓存。

## 打包说明

ZIP 附带官方源码、配置、项目代码、许可证、演示产物和测试日志；不附虚拟环境或系统依赖 wheel。首次安装需网络。

为使解压后的默认演示命令能直接执行，原 `runs/demo/` 在 ZIP 中映射为 `example_output/`，`runs/` 不打包。JSON 和日志中的绝对路径保留原始执行记录。

开发目录中的演示产物仍位于 `runs/demo/`；如果直接在开发目录重跑演示，指定一个新输出目录，例如 `--output runs/my_demo`。
