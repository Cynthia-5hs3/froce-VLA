# 复核标注后的实测/外部力矩对照实验

## 当前模型究竟使用什么

当前已训练的v3 checkpoint配置为 `use_external_torque=false`。输入是两路图像、任务文本、10D状态和过去2秒10点的7维实测关节力矩 `tau_J`；输出是未来50步10D动作及7维实测关节力矩。不是只使用外部力矩。

31D采集数据已保存 `tau_ext_hat_filtered`：第0–9维为原10D状态，第10–16维为tau_J，第17–23维为外部关节力矩，第24–30维为关节速度。窗口文件也已有 `external_joint_torque_history` 和 `future_external_joint_torque`，之前没有送入模型。

外部关节力矩是机器人估计并滤波的关节空间外部作用量，单位Nm；不是末端六维力，也不能直接当作物料接触力N。它能提供接触和阻碍线索，但抓取、姿态、负载、运动动力学和估计误差同样会影响读数；不能仅靠峰值判断是否放置成功。

## 第14条和一次放置成功组

第14条尾部 `[1890,2082)` 已按本次上传的confirmed/keep导入，旧版本自动归档。28条训练轨迹现在全部覆盖，6段仍为review，不自动批准。新索引：`outputs/stage_index_v3_20261016_round2/`，候选窗口45,101个，比上一版增加191个。

按已确认阶段顺序，轨迹15、25、28满足“接近→对准下压→可选就位保持→释放撤离”，没有标出的错误接触或纠偏阶段，也没有待复核/漏标段。列为一次成功候选组，依据是用户标注，并非本轮重新逐帧审核视频。允许取料和接触前的对准调整；不要求机器人全程直线运动。

第17条同样没有标出的失败重试，但撤离 `[1264,1515)` 仍为review，暂不加入该组；它已批准的窗口仍在其他组中正常使用。第14条包含碰撞和重新抓取，属于纠偏示范。3、4、9、17、19、38的待复核段继续不作为模仿目标。

## 两组实验及训练规则

生成目录：`outputs/stage_external_ab_20261016/`。

| 项目 | A：measured | B：measured_external |
| --- | --- | --- |
| 历史输入 | 10×7 tau_J | 10×14，每个时间点先tau_J后外部关节力矩 |
| 输入时间 | 同一原始索引，过去2秒10点 | 相同 |
| 未来动作 | 50×10 | 相同 |
| 辅助未来力矩目标 | 50×7 tau_J | 相同，首轮不增加未来外部力矩目标 |
| 训练窗口和采样计划 | 相同45,101个已批准窗口 | 相同 |
| 初始化 | v3 checkpoint-030000 | 同一checkpoint，新增输入列置零 |
| 预算 | 初步3,000更新，batch=4，lr=5e-6 | 相同 |

新增外部力矩是论文路线之外的对照扩展，不能称为论文原样复现。仅扩大力矩适配器的输入投影：70→140；仍压缩成一个力矩token，保持当前separate力矩/state条件布局和未来动作解码结构。旧权重按“时间点×7维”映射到每个14维时间点的前7列，外部列初始化为0，因此加载时不会因随机新增输入而立刻改变原适配器输出；随后外部列可获得梯度。

只恢复模型参数，不恢复旧优化器/旧训练游标；这是独立新运行。tau_J、状态和动作归一化沿用来源checkpoint，外部力矩独立使用已批准训练历史拟合均值/标准差，验证集不参与拟合。

采样规则已接入训练入口，并非仅生成候选文件：

1. 未来50个动作必须全部批准（keep/downsample）。review、未标和exclude不作为动作目标。错误接触可以留在过去观测中，让模型模仿随后正确恢复，而不是模仿错误顶压。
2. 一次成功候选组15、25、28合计期望占25%，其他已批准示范占75%。每组内各轨迹等概率，避免长纠偏轨迹只因时长更长而支配训练。
3. 单轨迹内，接近/释放权重1，纠偏1.5，下压3，必要保持1.5，idle为0.1；再乘原标注downsample的0.1。不是修改Transformer注意力，也没有奖励/critic。
4. 不再叠加旧RB关键阶段25%重采样，避免把两种规则重复加权。
5. 固定验证轨迹7、18、39。采样文件和原窗口SHA256写入训练合同，resume必须匹配。原28条以外的旧91条未混入本轮，避免同时改变数据来源；本轮的一次成功组来自这28条。

这些比例是待验证的首轮设置。按生成的概率，下压锚点期望占18.78%，纠偏27.36%，接近38.71%，释放14.43%，就位保持0.715%，idle约0.0029%。阶段按锚点统计，50步目标可能跨多个已批准阶段。保持片段稀少，不能用一味增大重复次数替代更多示范。

目标是减少错误动作和操作者等待的模仿，保留“直接成功”和“错误状态下正确恢复”两类路径，再判断外部力矩是否改善接触后的动作选择。不能保证外部力矩一加入就能识别放置成功，需在固定观测下比较，并最终验证闭环行为。

## 运行指令

以下仅供手动执行。本轮没有启动完整PI0.5训练、GPU推理或机器人。

先检查两组输入（CPU）：

```bash
cd /data0/wx/force-VLA
bash scripts/run.sh python -m force_vla.pi05.training --check-data \
  --config outputs/stage_external_ab_20261016/config_measured.json
bash scripts/run.sh python -m force_vla.pi05.training --check-data \
  --config outputs/stage_external_ab_20261016/config_measured_external.json
```

完整基础模型的两步连通性测试（另存目录，不是正式实验）：

```bash
bash scripts/run.sh --gpu python -m force_vla.pi05.training --smoke \
  --config outputs/stage_external_ab_20261016/config_measured_external.json \
  --output outputs/pi05_stage_external_smoke --device cuda:0
```

随后按顺序运行两组独立训练，不同时占用同一张卡：

```bash
bash scripts/run.sh --gpu python -m force_vla.pi05.training --train \
  --config outputs/stage_external_ab_20261016/config_measured.json \
  --output outputs/pi05_stage_measured_3k --device cuda:0

bash scripts/run.sh --gpu python -m force_vla.pi05.training --train \
  --config outputs/stage_external_ab_20261016/config_measured_external.json \
  --output outputs/pi05_stage_external_3k --device cuda:0
```

配置内已指定新的合同和原始窗口路径。输出目录必须不存在；同配置中断恢复时使用新输出目录并加 `--resume <本次实验已保存checkpoint>`，不能拿旧30k checkpoint当作新实验的resume。初始化旧30k参数已由 `initialize_checkpoint` 完成。

训练完成后的固定观测A/B：

```bash
bash scripts/run.sh --gpu python -m force_vla.pi05.hover_ab_evaluation \
  --checkpoint-a outputs/pi05_stage_measured_3k/checkpoint-003000 \
  --checkpoint-b outputs/pi05_stage_external_3k/checkpoint-003000 \
  --windows outputs/force_vla_31d_windows_v3/windows.parquet \
  --output outputs/stage_external_ab_eval_3k --seeds 32 --device cuda:0
```

该工具保留历史文件别名old_aligned_30k/v3_30k；实际模型路径以manifest为准。两组读取同一验证观测，分别按各自配置读取7维/14维历史。关注下压方向和幅度，也检查保持、回升，不用总loss或Z误差单独认定部署成功。当前工具的治具Z筛选仍可能包含取料观测，最终报告要复核manifest（上一轮episode39/frame185需排除），不能直接把原始汇总当作纯放置结果。

2026-10-19更新：实时入口已按checkpoint配置支持7维/14维历史；同一次31D快照的tau_J和外部关节力矩合并后，共用时间戳抽取过去2秒10点，并分别使用训练统计归一化。缺失或无效外部力矩会报错，不补零。已完成CPU时间对齐和归一化一致性测试，尚未运行外部力矩模型的完整训练或真机验证。新的独立B输出及启动指令见 [两套外部力矩实验](EXTERNAL_TORQUE_VARIANTS.md)，旧7维模型仍按原维度运行。

## 文件位置和验证范围

- `src/force_vla/pi05/stage_sampling.py`：规则、来源检查、独立配置/合同生成、旧适配器权重迁移。
- `src/force_vla/pi05/training.py`：消费采样计划和独立初始化配置。
- `src/force_vla/pi05/window_dataset.py`、`preparation.py`：外部力矩读取和独立归一化。
- `configuration_pi05_force.py`、`modeling_pi05_force.py`：可选14维力矩历史，不改变未来7维目标。
- `outputs/stage_external_ab_20261016/summary.json`：候选组、规则、预期分布和数据来源。

已在隔离CPU环境检查真实数据：两组均45,101个可选窗口，batch历史形状分别4×10×7和4×10×14；所有原窗口力矩值有限；种子42的一轮抽样一次成功组占24.97%，没有验证集泄漏。测试覆盖归一化、权重迁移、外部列反向梯度、缩小模型的50步推理、确定性采样及旧路径回归。没有把缩小模型测试描述为实际PI0.5训练完成。

以后更新标注后重新导入、build-index，再调用下列准备程序，用新的输出目录保留每版实验：

```bash
bash scripts/run.sh python -m force_vla.pi05.stage_sampling \
  --index outputs/stage_index_v3_20261016_round2 \
  --one-pass-fraction 0.25 \
  --output outputs/stage_external_ab_next
```
