# 独立环境与训练

全部工作位于 `/data0/wx/force-VLA`。沿用 Evo 所用 PI0.5 的源码副本、视觉/语言骨干、action expert、流匹配目标及 LeRobot AdamW/余弦学习率预设。历史与未来力矩窗口通过独立离线SFT入口接入；不调用原Evo的RLT或部署流程。

## 环境

独立Conda前缀 `/data0/wx/force-VLA/env` 已由此前独立复制准备，本次复用并核验。必须通过 `scripts/run.sh` 运行；无需预先`conda activate`，启动器自动选择该解释器并清除ROS/Python继承变量。原Evo、原Conda、本地数据和模型只读挂载，缓存、HOME及实验输出位于本目录。默认断网并隐藏机器人设备。

核心包版本见 `configs/training-tools.txt`。实际完整清单及GPU/依赖检查结果由下列命令写到 `outputs/environment/`：

```bash
cd /data0/wx/force-VLA
bash scripts/run.sh --gpu python scripts/check_training_environment.py
bash scripts/run.sh python -m unittest discover -s tests -v
```

`configs/force_vla_environment.yml` 仅描述Conda基础解释器，不是完整环境锁文件。当前独立环境保留原项目使用的LeRobot/PI-Gemma实现；不能仅凭同版本网上wheel保证字节级复现。无需升级或重新安装现有已通过检查的包。

## 数据

`outputs/force_vla_31d_windows/windows.parquet` 来源为本地31D副本中的101条成功候选。按episode分成91条训练/10条验证，对应91372/10511个窗口，种子42，同轨迹不会跨集合。原始数据不修改。

归一化只使用训练集合中最多10000个均匀选取的窗口。state/action以q01/q99映射，连续数值不截断；历史和未来力矩共用逐关节mean/std，避免抹去窗口的绝对幅值。文本离散state才裁剪至[-1,1]。统计、轨迹列表及窗口SHA256保存在 `outputs/pi05_force_prepared/data_contract.json`。

输入包括两路RGB、真实任务文本、10D state、10×7历史tau_J，监督包括50×10动作和50×7未来tau_J。动作对应t..t+49，未来力矩对应t+1..t+50；外部力矩和关节速度本版不接入。空任务文本与占位符`111`会报错。

## 运行

数据划分已存在时直接复用。需要重新准备时指定新目录，避免覆盖：

```bash
bash scripts/run.sh python -m force_vla.pi05.train_pi05_force --prepare --output outputs/pi05_force_prepared
bash scripts/train_force_pi05.sh --smoke --steps 2 --output outputs/pi05_force_smoke_new
bash scripts/train_force_pi05.sh --train --steps 3000 --output outputs/pi05_force_sft_v1
```

配置为 `configs/pi05_force_sft.json`。默认batch=1，仅训练torque adapter、state投影/融合和新增的动作/力矩输入输出投影。视觉语言骨干与expert参数冻结，但梯度通过expert传到条件token。`training.trainable="expert"`可解冻expert；该模式显存需另测。

加载 `models/pi05_sft` 时严格检查所有骨干权重，加载后才迁移原动作投影；不允许缺失骨干时静默随机初始化。语言沿用PI0.5的 `Task/State/Action` prompt，本地tokenizer为`models/tokenizer`。

每步记录动作/力矩损失与梯度范数，定期在验证轨迹上评估并保存。验证固定噪声种子并恢复训练随机状态。短程检查使用真实PI0.5、图像文本与力矩，仅证明链路可运行，不证明训练收敛或任务成功率；不会自动开始3000步正式训练。

2026-09-20已通过19项本地测试和完整本地权重的两步GPU验证，所有新增模块有非零梯度，16个可训练张量更新，输出为1×50×10，验证联合损失0.20209。PyTorch显存分配峰值9,711,707,648字节（约9.0 GiB）。详细记录位于 `outputs/pi05_force_smoke_20260920/`；这是链路验证，非收敛结果。

新进程重新加载基础模型与适配器后，也已用真实图像/文本/历史力矩验证推理，动作输出有限且为1×50×10，结果见同目录 `restore_result.json`。优化器、调度器、随机状态与样本游标的恢复由单元测试覆盖。

视频按锚点读取，无训练视频缓存。当前解码从视频起点顺序读取，长轨迹可能限制吞吐。

## 保存与恢复

checkpoint保存可训练参数 `adapter.safetensors`、优化器/调度器/随机状态 `training.pt`、配置、数据统计与基础权重SHA256。冻结骨干不重复保存，恢复必须保留 `models/pi05_sft`。`complete.json`标记完整保存。恢复使用新输出目录及相同配置/计划总步数，例如：

```bash
bash scripts/train_force_pi05.sh --train --steps 3000 --resume outputs/pi05_force_sft_v1/checkpoint-000500 --output outputs/pi05_force_sft_resume
```

离线推理使用 `force_vla.pi05.training.restore_for_inference(checkpoint, device)`，返回policy和normalizer。输入通过同一normalizer及`attach_visual_language_inputs`；动作输出通过`normalizer.transform(actions, "action", inverse=True)`恢复单位。预测力矩不作为机器人命令。本任务不启动硬件部署。

模型结构和论文差异见 [模型说明](PI05_FORCE_MODEL.md)，复制文件的原路径与用途见 `provenance/force_model_sources.json`。独立训练入口使用本目录env中的依赖，不修改site-packages或原Evo。
