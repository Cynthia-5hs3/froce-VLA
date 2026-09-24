# TA-VLA 对齐、关键阶段优化与评估

## 已完成和未执行

当前已有的30k模型是 **adapter/projection 微调**：约548万参数更新，Transformer 主体冻结；不是全量微调，也没有 LoRA。新代码提供独立 torque/state token、编码器与解码器 LoRA 的训练配置，但尚未运行这组训练，也未验证其真机效果。旧模型权重、原始数据、Evo-RLT和Conda环境保持不变。

完整架构的meta张量预检确认：旧checkpoint的16个可训练张量、5,483,559个参数名与形状完全匹配；新配置520个可训练张量、37,907,495个参数，其中两侧Transformer各252个LoRA张量。记录见 `outputs/ta_vla_alignment_preflight_20260921.json`；该预检没有加载大模型权重执行前向，也没有使用GPU。

本次推理时间对齐可直接用于旧模型：独立30Hz观测、过去2秒10点力矩缓存、最多8Hz异步推理请求、按30Hz动作时间轴消费50步输出。依据观测时间补偿已发生的延迟，不反复执行第0步。默认3步块间位姿过渡，夹爪采用新块当前目标而非平均。预测力矩不作为电机力矩命令。

历史偏移与已有训练窗口相同：`[-60,-53,-47,-40,-33,-27,-20,-13,-7,0]/30` 秒；这是30Hz离散网格上近似均匀的10点，包括现在。观测时间戳最近邻选择、最大约33.23ms匹配误差，超过100ms缺口的历史不使用。没有足够2秒历史就等待，不复制填充。真机入口会核对checkpoint数据窗口元信息，拒绝不一致的时间合同。

## 论文的“拼接”

论文2509.07962 Sec.4.1比较 Enc / DePre / DePost。DePre把力矩拼入状态；最终Sec.6使用DePost-1 Token。其含义是沿**序列维**拼接，而非沿特征维合成一个state token：

```text
[视觉/文本 prefix] → [历史力矩 token, state token, 50个动作/力矩 flow token]
```

官方 `third_party/ta_vla_149d4be/src/openpi/models/pi0.py` 的 `embed_suffix` 先追加 effort token，再追加state token。新配置与这一顺序和注意力分块对齐；旧 fused 配置保留用于加载现有模型。

训练动作`t..t+49`、力矩`t+1..t+50`与官方 `training/data_loader.py` 相同。保持β=0.1、测量关节力矩输入。仍保留PI0.5的状态文本、AdaRMS时间条件、单臂TCP动作、双相机和冻结SigLIP等适配差异，不宣称完全复现论文的π0/JAX全部参数和本体。

## 少量人工关键阶段标记

`anchor_phase=1` 表示人工标记的关键阶段；`0` 表示未标记，不能解释成无接触或失败。标记仅用于训练抽样和离线评估，不作为部署时必须提供的输入，不增加接触分类损失。

新配置 `configs/pi05_force_tavla_aligned.json`：

- `conditioning_layout="separate"`。
- `trainable="lora"`：PaliGemma语言Transformer rank/alpha=16/16，action expert=32/32，对应官方Gemma变体；适配器、state/flow投影和时间MLP也训练。基础Transformer权重与SigLIP冻结，不是全量微调。
- `critical_sample_fraction=0.25`：每个训练采样周期约25%来自有标记窗口，其余来自未标记窗口。均有放回，固定seed与游标可恢复，严格只从91条训练轨迹抽样。原自然比例约7.75%。25%是本地增强方案，不是论文规定；没有同时叠加关键阶段loss加权。
- 验证逐episode、逐标记组均匀取窗口，单独报告动作/力矩loss，避免少量标记被总体均值掩盖。
- 稀疏RB标记尚不能区分下降、闭爪、抬升、放置；进一步细分需要额外阶段标注。按锚点分组的指标衡量整段50步预测，不等同于仅对接触帧计算的损失。

先做新结构的小规模链路检查，再另行启动正式训练。以下命令**未在本次执行**，不会隐式调用机器人：

```bash
cd /data0/wx/force-VLA
bash scripts/train_force_pi05.sh --smoke --steps 2 \
  --config configs/pi05_force_tavla_aligned.json \
  --output outputs/pi05_force_tavla_aligned_smoke

bash scripts/train_force_pi05.sh --train \
  --config configs/pi05_force_tavla_aligned.json \
  --output outputs/pi05_force_tavla_aligned_v1
```

新配置batch=4、30k步；新LoRA/布局的真实显存占用尚未实测。应先完成小规模检查，不将旧适配器实验的显存数字当作新配置保证。新结构从本地 `models/pi05_sft` 初始化，不能直接把旧fused adapter当作完整新模型续训。原训练/验证划分与归一化统计复用。

## 独立阶段评估

已有评估可以按episode/frame与phase关联，**无需GPU或再次模型推理**：

```bash
bash scripts/run.sh python -m force_vla.pi05.phase_evaluation \
  --checkpoint outputs/pi05_force_sft_30k/checkpoint-030000 \
  --existing outputs/pi05_force_sft_30k/validation-030000.json \
  --output outputs/phase_loss_regrouped.json
```

本次已生成 `outputs/phase_audit_20260921_loss.json` 和 `outputs/phase_audit_20260921_samples.json`。原320个loss验证窗口中只有20个关键标记窗口、覆盖7条轨迹；其宏平均torque_loss约2.052，未标记组约1.409。样本数量与轨迹组成不同，不可视为控制变量对比或“标记导致更差”；但说明总体均值不能代表关键阶段。

重新做保证关键阶段覆盖的独立评估时（以下命令尚未运行）：

```bash
bash scripts/run.sh --gpu python -m force_vla.pi05.phase_evaluation \
  --checkpoint outputs/pi05_force_sft_30k/checkpoint-030000 \
  --per-phase-per-episode 16 --output outputs/phase_loss_balanced.json

bash scripts/run.sh --gpu python -m force_vla.pi05.phase_evaluation \
  --checkpoint outputs/pi05_force_sft_30k/checkpoint-030000 \
  --samples --per-phase-per-episode 16 --output outputs/phase_samples_balanced.json
```

第二条报告位置mm、姿态、夹爪及力矩Nm误差。每组包含窗口数/轨迹数，空组不报告虚构分数。新分组采样总指标不可与旧均匀取样总体均值直接比较；比较checkpoint时固定同一采样规则与seed。

## 验证范围

CPU小模型测试覆盖旧fused布局、新双token注意力、完整前向与缓存推理一致性、双Transformer LoRA梯度和增量权重恢复；时间轴测试覆盖历史缺口、因果采样、延迟跳步、块过期和遥操在途预测作废。没有启动真机、没有修改速度/力矩限制，没有重训现有30k模型。动作连续性与任务成功率仍需后续人工控制的真机验证。

本轮完整本地测试结果：`46 passed`。还验证了首次历史预热不发送动作、观测模式不发送动作、键盘停止先于推理线程等待，以及遥操释放后保持夹爪目标的回归路径。
