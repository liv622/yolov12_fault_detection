# 上游来源

- 仓库：https://github.com/sunsmarterjie/yolov12
- 提交：64d5ffc6db35a1bde446689ab8450ecd43475e00
- 获取日期：2026-11-06（Asia/Shanghai）
- 本地目录：vendor/yolov12
- 模型配置：ultralytics/cfg/models/v12/yolov12.yaml
- 模型规模：n（YOLOv12-turbo nano）
- 上游许可：AGPL-3.0，保留原 LICENSE 及版权头

vendor 内的源代码保持原样。项目使用自己的固定依赖，避免上游 requirements.txt 中的特定 CUDA/Python 3.11 FlashAttention wheel。
运行时通过官方提供的 SDPA 回退支持 CPU；本项目没有伪造 YOLOv12 类或以其他网络替代。
打包时仅排除 Git 元数据、Python 缓存及安装生成的 egg-info；完整上游源文件保留。
