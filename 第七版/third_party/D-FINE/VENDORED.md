# Vendored D-FINE

来源：官方仓库 <https://github.com/Peterande/D-FINE>（Apache-2.0，`master` 分支，经 jsDelivr 镜像
获取）。第七版只使用其中的**模型定义**（backbone / hybrid encoder / OQD decoder / FDR 头 /
criterion / matcher / postprocessor）；数据管线、训练循环、评估与提交全部使用 `aic_v7` 自有实现。

权重：`weights/dfine_l_coco.pth`（D-FINE-L COCO 预训练，31.4M 参数，31.2M 可训练）与
`weights/dfine_x_coco.pth`（X 档消融），下载自 HuggingFace `Peterande/D-FINE`（hf-mirror 镜像）。
官方检查点格式为 `{"model": <plain state_dict>}`，键名与 vendored 代码一致。

## 本地补丁（aic_v7 vendored trim，全部为最小改动）

| 文件 | 补丁 | 原因 |
|---|---|---|
| `src/__init__.py` | 不再 eager 导入 `data` 子包 | `src/data` 依赖 faster-coco-eval / pycocotools，只服务于官方 dataloader |
| `src/misc/__init__.py` | 只导入 `dist_utils` | logger/profiler/visualizer 依赖 loguru / calflops / PIL |
| `src/core/_config.py` | `SummaryWriter` 改为守卫导入 | tensorboard 只服务于官方训练循环 |
| `src/zoo/dfine/dfine_criterion.py` | forward 中记录 `self.last_indices` | Relation-Aware Refiner 复用匹配结果做辅助监督，不重复匹配 |
| `src/zoo/dfine/dfine_decoder.py` | 解码器返回 `pred_features`（非 dn 内容查询特征，训练/推理均输出） | 外观关系注意力 W_app 的输入 |

所有补丁以 `[aic_v7 vendored trim]` 注释标记。官方训练脚本（`train.py` / `src/solver` /
`src/data`）保留但第七版不调用。
