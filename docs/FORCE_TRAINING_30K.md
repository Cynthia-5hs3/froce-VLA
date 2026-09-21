# 力矩 VLA 累计三万步训练

全部新增代码、配置、日志和模型位于 `/data0/wx/force-VLA`；使用 `scripts/run.sh --gpu` 和本目录 `env/`。数据、基础模型、Evo-RLT 与原 conda 环境不修改。训练进程无法访问机器人或网络，不执行部署。

## 预算与初始化

配置：`configs/pi05_force_continue_30k.json`。入口：`src/force_vla/pi05/extended_training.py`，原来的三千步训练入口和配置保持可用。

- 继承 `outputs/pi05_force_sft_v1/checkpoint-003000` 的适配器、AdamW 动量、随机状态和样本游标；累计终点为30000，再执行27000次优化更新。
- micro batch=1，梯度累积4次，有效batch=4。原3000个窗口加本阶段108000次窗口访问约为1.215次训练集遍历。保持原有91条训练/10条验证划分和归一化统计。
- 原三千步余弦周期已结束。本阶段显式开启新的学习率周期：2.5e-6开始，500次更新预热至1e-5，然后余弦下降到1e-6。不把修改预算伪装成原训练的精确恢复。
- 保持约548万适配器参数可训练，视觉语言骨干和action expert冻结；外部力矩、速度不增加输入，力矩损失权重仍为0.1。先控制变量判断增加训练量的收益。该阶段不保证冻结expert足以达到任务精度。

## 验证与采样推理

开始训练前，先对第3000步进行相同新评估，建立可比基线。

- 每1000步，每条验证轨迹沿时间轴均匀选择32个窗口，共320个；按轨迹等权汇总动作/力矩流匹配损失，保存每条轨迹与每个窗口的指标。
- 每5000步，对每条轨迹的3个时间位置执行10步完整流采样；每个位置使用2个固定随机种子。
- 对每次采样使用相同噪声比较真实力矩历史、物理值为0 Nm的历史、其他验证轨迹的历史。其他轨迹来自验证集，不能解释成训练好的无力矩基线。
- 最终第30000步扩大为每条轨迹16个位置，共160个窗口、2个种子、3种历史条件。
- 记录50步位置误差mm、前10步位置误差、姿态角误差、夹爪误差/0.5阈值分类错误率、力矩RMSE Nm、动作相邻点最大位移、退化rot6d比例和夹爪越界比例。离线夹爪阈值不是实际控制器的滞回规则。
- 采样延迟包含首次warmup，GPU与现有服务共享时只反映当前负载。离线视频解码耗时单独记录，不能当作真机相机延迟。
- `best_validation.json` 只根据轨迹平均动作流损失选择候选，保留所有checkpoint；不会自动把最低损失模型标为可部署。

时间均匀采样不等价于人工标注的接触阶段评估。接触峰值、压合成功率、实机闭环表现仍须单独验证。没有通用的loss或训练步数阈值能证明可以部署；完整采样误差与无力基线对照后，才进入实时观测推理与受限真机试验。

## 启动与查看

已占用GPU的Evo-RLT策略服务不会被停止。只有在机器人不依赖其及时推理时才共用GPU进行训练。

```bash
cd /data0/wx/force-VLA
bash scripts/train_force_pi05_30k.sh --output outputs/pi05_force_sft_30k
```

输出目录必须不存在，避免覆盖模型。后台任务的终端输出由启动方保存为 `outputs/pi05_force_sft_30k.console.log`。

本次长程任务运行于独立tmux服务的 `force-30k` 会话，socket为 `runtime/tmux-30k.sock`；不会使用或改变原Evo的tmux会话。查看是否存活：

```bash
tmux -S /data0/wx/force-VLA/runtime/tmux-30k.sock list-sessions
```

```bash
tail -f outputs/pi05_force_sft_30k.console.log
cat outputs/pi05_force_sft_30k/status.json
cat outputs/pi05_force_sft_30k/best_validation.json
```

主要产物：`run.json`（来源/代码SHA256）、`metrics.jsonl`（每次优化更新）、`validation-*.json`、`inference-*.json`、`checkpoint-*/`。checkpoint仍为基础模型加适配器增量，必须保留 `models/pi05_sft`。

SIGINT/SIGTERM会请求在当前更新/评估后保存退出。不能同时重复启动多个训练进程。异常退出以最后存在 `complete.json` 的checkpoint为准，进程状态需与日志一起检查，不能仅凭残留 `status.json` 判断仍在运行。

精确恢复同一扩展周期时使用相同配置、代码与新的输出目录：

```bash
bash scripts/train_force_pi05_30k.sh \
  --resume outputs/pi05_force_sft_30k/checkpoint-010000 \
  --output outputs/pi05_force_sft_30k_resume
```

`--max-updates 2 --skip-initial-evaluation` 用于小规模验证，不改变30000步学习率计划。它会保存暂停checkpoint，可以按同样方式恢复。
