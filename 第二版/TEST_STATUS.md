# 测试状态

- Python `compileall`：已通过，覆盖 `src/`、`scripts/` 和 `tests/`。
- dependency-light group split smoke test：已通过，三个 fold 大小 `[4, 4, 4]`，无 group 交叉。
- 完整 pytest：未运行。当前发现的 Python runtime 缺少 PyYAML、OpenCV、PyTorch、Torchvision、Ultralytics、Matplotlib 和 pytest；pip 联网安装超时。
- GPU forward / loss / official Ultralytics decode 对照：未运行。
- 全量数据 audit/figure/split：未在本终端运行；已提供脚本和已知统计摘要。

正式训练和提交前必须在目标训练环境完成 `README.md` 与 `docs/06_复现与提交检查清单.md` 中的测试。
