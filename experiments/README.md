# 训练数据与模型分类（2026-10-21）

这是按实验建立的索引目录，不是复制出新的训练集。每组 manifest 记录原路径、训练/验证 episode、窗口哈希、训练设置和已保存 checkpoint。原始数据、模型、已确认标注及训练合同保持原位置，避免破坏配置中的绝对路径。历史 run.json 的内嵌 data_contract 优先于旧配置模板中的 dataset_windows。

| 分类 | 数据 | 模型/状态 |
| --- | --- | --- |
| [01_old_success_adapters](01_old_success_adapters/manifest.json) | 旧101条成功：早期适配器基线 | 见manifest最后保存步数 |
| [02_old_success_lora](02_old_success_lora/manifest.json) | 旧101条成功：TA-VLA对齐LoRA基线 | 见manifest最后保存步数 |
| [03_correction_v3_lora](03_correction_v3_lora/manifest.json) | 后采31条成功：包含纠偏过程，未用细阶段采样 | 见manifest最后保存步数 |
| [04_stage_a_measured](04_stage_a_measured/manifest.json) | 阶段A：已批准窗口，实测7维历史 | 完成3k增量训练 |
| [05_old_success_external](05_old_success_external/manifest.json) | 旧101条成功：实测7维＋外部7维历史 | 见manifest最后保存步数 |
| [06_stage_b_external](06_stage_b_external/manifest.json) | 阶段B：与A配对，14维历史，尚未训练 | 已准备，未训练 |
| [90_smoke_preflight](90_smoke_preflight/manifest.json) | 技术冒烟/预检：不作为任务效果模型 | 仅预检 |

## 数据版本

- 旧批：`data/franka_single_left_31d`，窗口 `outputs/force_vla_31d_windows/windows.parquet`；101条，91训练/10验证，101,883窗口，其中91,372训练。
- 后采中间版：`data/franka_single_left_31d_v2` → `outputs/force_vla_31d_windows_v2/windows.parquet`；28条、53,578窗口，是同一批后采数据的较早快照。
- 后采最终版：同一原始根目录 → `outputs/force_vla_31d_windows_v3/windows.parquet`；31条，28训练/3验证，共57,362窗口（51,718训练/5,644验证）。31条均为成功结果候选，内部可能包含错误接触、人工纠正和等待，不等于31条失败示范。
- A/B：从v3训练部分筛选45,101个批准窗口，并使用阶段权重及一次到位轨迹配额。A完成；B未训练。已确认标签与采样合同仍是复现这两组的必要资产。
- [datasets.json](datasets.json)含各窗口版本完整episode集合、哈希和时间轴；各组manifest含实际train/validation划分。两个原始根目录的相同episode编号不是同一条示范。

## 核对结果和使用边界

三个windows.parquet已重算SHA-256；分组使用每次run的内嵌合同，不依赖文件夹名称推断。v3旧run.settings.dataset_windows仍写着旧窗口模板路径，但run.data_contract指向v3；清单已显式标记该差异，不篡改历史记录。

模型加载需要本组完整checkpoint、`models/pi05_sft`、tokenizer和对应归一化。14维输入指7维实测＋7维外部**关节力矩**，不是末端六维力。所有当前主模型的未来力矩目标仍为7维实测值。

最新success_30k从基座使用旧101条训练，没有使用后采v3、阶段A或阶段B数据。不能把它的真机问题归因于阶段标签污染。

## 清理状态

用户确认后，已将239个标注UI、草稿工具、草稿/收件箱和旧网页输出文件归档并删除；[逐文件清理记录](cleanup_candidates.json)保留原路径、大小和哈希。归档另含6个本次修改文件的清理前版本，共245个文件，已逐文件校验。[恢复说明](../archives/stage_cleanup_20261021/README.md)。旧`prepare`命令现在明确提示网页已退役，不会创建残缺输出目录。

`training.py → stage_sampling.py → stage_annotations.py`仍是直接导入链；`external_experiments.py`也依赖阶段采样。完全移除阶段功能需要单独解除依赖并更新训练入口及回归测试。已确认的28条标签、A/B合同/索引、模型、原始数据和推理日志不归类为垃圾。

此次仅退役标注工具，不取消阶段训练功能，没有添加模型程序、修改环境或启动训练/机器人。原始采集视频、模型、已确认的`annotations/stages_v3`及A/B训练合同保持原位置。
