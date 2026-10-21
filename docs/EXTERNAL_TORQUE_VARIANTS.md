# 两套外部关节力矩训练与测试

2026-10-20更新：`success_30k`已完成30,000步，退出码0，模型在`outputs/pi05_external_success_30k/checkpoint-030000`。B组尚未启动。本次没有连接真机。所有改动限于force-VLA；不修改Evo-RLT、采集数据、历史checkpoint和环境。

## 本轮结果与部署参数

完整训练分析及曲线在`outputs/pi05_external_success_30k/external_success_analysis_20261020/`。同一批272个验证观测、10条验证轨迹上，旧实测力矩30k模型action_loss为0.028992，新外部力矩18k为0.027090（降低6.56%），新30k为0.028620（降低1.28%）。新30k总loss/力矩loss在本次训练中最低，但相对旧实测30k略高；不能据此宣称外部力矩已经明显改善真机成功率。18k和30k都保留作部署候选，不按总loss单独判断动作表现。

三模型GPU离线采样已完成：相同30个验证观测、4个噪声种子、10次解码。旧7维30k/新14维18k/新14维30k的前10点位置RMSE分别5.333/5.369/5.307mm，全50点分别13.249/13.317/13.555mm；新30k夹爪开度MAE较旧版降低约20%，但不代表真实释放成功率。保持脚本默认30k为首轮测试候选，18k作对照。动作改善有限，仍有保持组偏向下压、上升幅度不足的现象，不能宣称已解决徘徊或就位偏差。完整指标与限制见上述目录的`REPORT.md`。

部署切换的是**整个checkpoint目录**，不是替换一个`adapter.safetensors`。需要该目录内的`adapter.safetensors`、`force_config.json`、`run.json`、`complete.json`，以及本工作区的`models/pi05_sft`基座、`models/tokenizer`和时间合同引用的窗口元信息。`training.pt`用于恢复训练，不是实时推理的必需输入。不要把14维适配器放进旧7维模型目录或沿用旧run.json。

本项目更新后的入口会读取checkpoint的`use_external_torque`和归一化统计，自动选择10×7或10×14历史，无需额外指定force维度。TCP/rot6d/夹爪状态仍为10D；31D是采集格式。外部关节力矩通过现有libfranka快照读取，不需要新ROS话题。

| 参数 | 当前值/位置 | 含义与修改方式 |
| --- | --- | --- |
| 模型 | CLI `--checkpoint` | 默认新30k；可显式改为`outputs/pi05_external_success_30k/checkpoint-018000`作对照 |
| 机器人参数 | `configs/deployment_force_vla_smooth_test.yaml` | 相机、串口、工作区、速度、平滑和夹爪阈值均在此；`model-check`与动作测试已统一使用它 |
| 平移速度 | YAML `robot.max_linear_velocity: 0.05` | 50 mm/s速度限制，不是每步移动50 mm；初次对比先保持相同速度 |
| 旋转速度 | YAML `robot.max_angular_velocity: 0.12` | 约6.9°/s，和单次旋转跳变门控不同 |
| 跟踪时间常数 | YAML `robot.velocity_tracking_tau_s: 0.25` | 将目标误差转换为速度；当前速度×时间常数对应约12.5 mm的位置误差限幅，再经过平滑 |
| 平滑 | YAML `target_smoothing_alpha: 0.45`、`velocity_smoothing: 0.45` | 位姿目标和速度的两层平滑，不要在换模型时同时改动以免混淆效果 |
| 单次跳变门控 | CLI `--max-position-jump 0.08 --max-rotation-jump 0.35` | 预测目标相对当前姿态超过80 mm/约20°则拒绝；增大门控不会自动提高运动速度 |
| 工作区 | YAML `workspace.min/max` | 基坐标米制边界；保留现有边界，模型切换不要求扩展 |
| 推理请求频率 | CLI `--rate-hz 8` | GPU处理若较慢，实际频率低于8Hz；采样/控制仍30Hz |
| 解码迭代 | CLI `--inference-steps 10` | 流匹配采样迭代数，不是执行10个动作；每次输出仍50步 |
| 连续动作 | CLI `--continuous` | 否则默认只执行1步；旧的“一下就退出”要首先检查此参数 |
| 初始开爪 | CLI `--startup-open` | ARM确认前主动开爪，只有execute模式允许 |
| 夹爪阈值 | YAML `gripper.close_trigger: 0.75`、`open_trigger: 0.875`、`post_close_lock_s: 2.0` | 本次未改变防反转逻辑；新模型并不保证解决短开爪/误闭合 |

先检查完整模型加载（不连接机器人），再由操作者启动观测与动作：

```bash
cd /data0/wx/force-VLA
bash scripts/force_external.sh success_30k model-check
FORCE_VLA_HARDWARE=1 bash scripts/force_external.sh success_30k observe --max-steps 300
```

观测模式不发送机械臂/夹爪动作，仍会连接并读取硬件。确认后执行：

```bash
FORCE_VLA_HARDWARE=1 bash scripts/force_external.sh success_30k execute \
  --startup-open --continuous --rate-hz 8 --inference-steps 10 \
  --log outputs/pi05_external_success_30k/realtime_external_success_30k_test_20261020.jsonl
```

若对照18k，把以下两项追加到上面任一命令末尾（必须位于同一个shell命令中）：

```text
--checkpoint outputs/pi05_external_success_30k/checkpoint-018000
--log outputs/pi05_external_success_30k/realtime_external_success_18k_test_20261020.jsonl
```

完整18k动作命令：

```bash
FORCE_VLA_HARDWARE=1 bash scripts/force_external.sh success_30k execute \
  --checkpoint outputs/pi05_external_success_30k/checkpoint-018000 \
  --startup-open --continuous --rate-hz 8 --inference-steps 10 \
  --log outputs/pi05_external_success_30k/realtime_external_success_18k_test_20261020.jsonl
```

每次运行的JSONL开头新增`deployment_session`事件，记录权重哈希、模型步数、机器人YAML及其哈希、力矩维度、频率和运动门控。新记录`telemetry_schema_version=2`中，`action/requested_*`是请求；`sent_xyz`为控制器计算速度时采用的平滑位置参考（本周期未提交运动则null），机器人实际接收的是`command_debug.commanded_linear_velocity/commanded_angular_velocity`对应的速度命令。`sent_gripper`为控制器保持的二值目标，是否本周期更新由`gripper_command_sent`表示，传感器反馈仍看`measured_*`。这些目标都不等于物理到位。旧日志没有自动重写，不能按新语义解释旧`sent_*`。

## 真机测试中用手柄纠正（不训练）

在同一个推理进程启用手柄，不要同时启动另一套遥操或采集程序抢占机器人和手柄。当前设备路径为`/dev/input/by-id/usb-Microsoft_Controller_3039373130303635393336353232-event-joystick`；换手柄后可通过`--teleop-device`指定路径。

```bash
cd /data0/wx/force-VLA
FORCE_VLA_HARDWARE=1 bash scripts/force_external.sh success_30k execute \
  --checkpoint outputs/pi05_external_success_30k/checkpoint-030000 \
  --startup-open --continuous --rate-hz 8 --inference-steps 10 \
  --teleop-assist --teleop-latched \
  --teleop-linear-speed 0.02 --teleop-angular-speed 0.10 \
  --log "outputs/pi05_external_success_30k/realtime_external_success_30k_assist_$(date +%Y%m%d_%H%M%S).jsonl"
```

启动先开爪，输入ARM后模型开始工作。按LB（左肩键，非LT扳机）或操作运动/夹爪控件可接管；松开控件仍保持人工模式，不自动恢复模型。松开全部运动、扳机和LB后，单击RB（右肩键）交回模型；正在操作时按RB会忽略该请求，须松开后重新按。交回会清空旧动作和力矩缓存，等待2秒新历史及新预测，之后继续任务。键盘`r`不会退出人工模式。手柄`X`是旋转，键盘`x`才是退出。

| 控件 | 人工模式动作（机器人基坐标） |
| --- | --- |
| 左摇杆 | XY平移；左摇杆Y映射+X，X映射−Y |
| Y / A | Z升高 / 降低 |
| 右摇杆 | 绕X/Y轴旋转 |
| X / B | 绕Z轴正向 / 反向旋转 |
| LT / RT | 请求打开 / 闭合夹爪 |
| LB / RB | 接管并保持人工模式 / 中立状态下交回模型 |
| 键盘q、x、Esc、Ctrl+C | 停止并退出 |

手动平移每轴最大20mm/s；可把`--teleop-linear-speed`降到0.01做高度精调。仍沿用速度、工作区、平滑和夹爪锁定逻辑，因此手动夹爪请求也可能受原2秒闭合锁定影响。松手是停止手动运动输入，并非硬件急停；手柄断连或事件丢失会使循环报错退出并走现有停止流程，不会继续复用最后的手柄输入。

日志增加`teleop_takeover/teleop_release`事件，逐帧保留人工接管标记、请求速度/夹爪、观测状态和力矩。它只用于复核，没有录制完整训练视频或自动生成示范数据，没有更新任何模型参数。在线纠正不会使下一次测试自动学会该位置。旧`--teleop-assist`不加`--teleop-latched`仍保留松手自动交回的模式。

## 实时位置、角度和力矩曲线

先在第一个终端启动上述推理命令；使用更新代码重新启动才能记录7个关节角。第二个终端启动只读监视程序：

```bash
cd /data0/wx/force-VLA
LOG="$(ls -t outputs/pi05_external_success_30k/realtime_external_success_30k*.jsonl | head -n 1)"
printf '本次监视日志：%s\n' "$LOG"
bash scripts/run.sh python -m force_vla.pi05.telemetry_dashboard \
  --log "$LOG" --output outputs/realtime_dashboard --points 1800
```

本机浏览器打开`file:///data0/wx/force-VLA/outputs/realtime_dashboard/index.html`。该命令启动时选择最近修改的success_30k日志，此后固定跟踪这一文件；切换推理日志后需重新运行监视命令。也可将`--log`显式指定为推理终端使用的文件，避免选错会话。推理尚未启动时不要用此命令误将旧测试当作实时数据。

页面包含TCP位置XYZ（mm）、姿态Roll/Pitch/Yaw（°）、7个关节角（°）、原始10D动作、7维实测/预测关节力矩、7维外部关节力矩（Nm）。位置/姿态区分模型预测、请求目标和实测反馈；请求目标不是实际到达位置。淡橙色背景表示人工接管，状态栏区分模型控制、手柄接管和等待新历史。人工接管时不补造模型预测；传感器与人工请求曲线继续更新。

姿态采用`R = Rz(yaw) Ry(pitch) Rx(roll)`，从rot6d正交化转换，只用于显示，不修改模型和机器人控制。±180°跨界时断线，pitch接近±90°存在欧拉角奇异性；需要时可切换原始10D页面看rot6d。关节角来自机器人q，不是从TCP姿态反算。旧日志缺失关节角时留空。

默认每0.25秒刷新页面数据，缓存1800点可显示约60秒的30Hz记录。预测力矩对应未来目标，和当前实测不是同一物理时刻，不能将两线差直接当预测误差。外部力矩为关节空间估计量，不是末端六维力。监视程序只读日志，不连接硬件、不占用模型GPU；关闭页面或按Ctrl+C关闭监视程序都不会停止机器人。

## 两个版本

| 项目 | success_30k：旧成功数据版 | stage_b：对应A组的B方案 |
| --- | --- | --- |
| 数据 | `data/franka_single_left_31d` | `data/franka_single_left_31d_v2` |
| 窗口 | `outputs/force_vla_31d_windows/windows.parquet` | `outputs/force_vla_31d_windows_v3/windows.parquet` |
| 训练/验证轨迹 | 91/10，共101条旧成功示范 | 28/3，共31条成功结果的示范，部分包含纠偏过程 |
| 训练窗口 | 91,372 | 45,101个已批准窗口 |
| 数据规则 | 原RB采样比例0.25；不加入后来采集的纠偏批次 | 原人工批准规则、阶段加权、一次成功候选组25% |
| 训练起点 | `models/pi05_sft`，与旧论文对齐LoRA版相同基座，重新训练30k | 与A组相同的v3 30k checkpoint，新增外部输入列置零 |
| 预算 | 30,000步，batch=4，lr=1e-5 | 3,000步，batch=4，lr=5e-6 |
| 对照模型 | `outputs/pi05_force_tavla_aligned_30k_resume/checkpoint-030000` | `outputs/pi05_stage_measured_3k/checkpoint-003000` |
| 新训练输出 | `outputs/pi05_external_success_30k` | `outputs/pi05_external_stage_b_3k` |

“旧30k”本次选用后来加入纠偏数据之前、已经对齐2秒历史的LoRA版，而不是更早的`pi05_force_sft_30k`适配器版。旧成功数据版从基座重新训练，不是从v3或A组继续训练，也没有混入新纠偏数据；“成功”仅指原分类，不承诺每段都是一次无纠偏完成。

B保持此前约定的实验，不从A组3k终点再续训，而是从A的相同起点`outputs/pi05_force_v3_resume_from_014000/checkpoint-030000`初始化。继承模型参数，不恢复旧优化器；70列力矩历史投影扩为140列，按每个时间点映射原实测列、外部列置零。B的来源模型已经接触过v3数据，不能把B称作完全未看过纠偏的模型。两版本的数据、初始化和预算不同，彼此比较不构成单变量消融；分别与各自的7维对照比较。

## 数据和模型接口

两版均使用两路图像、任务文本`Place the white lid firmly onto the jig.`、10D状态，以及过去2秒10点14D力矩历史。每点顺序固定为`[tau_J的7维, tau_ext_hat_filtered的7维]`；offset为`[-60,-53,-47,-40,-33,-27,-20,-13,-7,0]`帧，30Hz。

未来动作仍为50×10，未来辅助目标仍为50×7实测tau_J；没有加入未来外部力矩损失，也没有把预测力矩直接作为机器人力控命令。外部力矩输入扩展沿用现有单力矩token的适配器，状态和动作格式不变。

外部力矩独立归一化，只从训练历史拟合。旧成功版使用原训练划分；B保留原B合同里的外部统计与采样计划，验证轨迹不参与拟合。状态、动作、实测力矩统计沿用各自原合同。过去错误接触可以作为历史条件，B模仿目标仍须全部批准。

配方在`configs/force_external_variants.json`，准备程序为`src/force_vla/pi05/external_experiments.py`。生成文件分别在：

- `outputs/pi05_external_success_30k/external_success_30k_prepared/{config.json,data_contract.json,preparation.json}`。
- `outputs/external_stage_b_prepared/{config.json,data_contract.json,preparation.json}`。

`preparation.json`记录来源checkpoint、合同和窗口哈希、划分、初始化、预算。准备目录已生成；`prepare`拒绝覆盖。日后更改实验应使用新的准备/训练目录，并同步脚本的版本映射。不要覆盖旧合同或把新实验用旧checkpoint的`--resume`启动。

## 现在可以执行：仅CPU检查

```bash
cd /data0/wx/force-VLA
bash scripts/force_external.sh success_30k check
bash scripts/force_external.sh stage_b check
```

检查窗口哈希、轨迹划分、采样计划、数值有限性、任务文本、时间约定、视频文件存在和归一化后14维形状，不加载完整模型、不连接机器人。检查文件存在并不等于逐视频解码审查。本轮已确认两组分别引用202/62个视频文件，旧数据历史上的清理不等于本次检查时所有旧成功视频不存在。

首次在新的工作副本准备时，可将上述`check`改为`prepare`；本地已经准备完成，无需重复。

## 之后手动训练：本次未执行

两条命令分别启动独立实验，不要同时运行。旧成功版已经完成30k；下面的success_30k命令仅供确实需要重训时使用，输出到该模型根目录下一个尚不存在的子目录：

```bash
cd /data0/wx/force-VLA
bash scripts/force_external.sh success_30k train --output outputs/pi05_external_success_30k/retrain_01
```

```bash
cd /data0/wx/force-VLA
bash scripts/force_external.sh stage_b train
```

脚本使用原独立`env`和GPU隔离启动器，并占用现有训练锁。可在末尾指定`--device cuda:1`选择可见设备；这里不创建CUDA/Conda环境。训练输出目录必须不存在。success_30k的准备文件已位于模型根目录内，即使首次准备也会创建该根目录，因此训练应显式指定新的子目录。上述命令预期最终模型分别为：

- `outputs/pi05_external_success_30k/retrain_01/checkpoint-030000`。
- `outputs/pi05_external_stage_b_3k/checkpoint-003000`。

旧成功版30k当前已经存在，默认测试仍加载`outputs/pi05_external_success_30k/checkpoint-030000`；B版尚未生成。测试上述重训模型时需显式覆盖`--checkpoint`；离线配对工具对应`--checkpoint-b`。中断恢复须使用同一次实验的checkpoint，并指定新的输出目录。归类后旧checkpoint的历史配置保持原样；严格恢复请使用`train --config <本实验checkpoint>/force_config.json --contract <当前合同路径> --resume <本实验checkpoint> --output <新目录>`，避免仅路径变化触发settings一致性检查失败。

## 训练后测试：先离线，再由操作者启动真机

离线配对采样，不连接机器人：

```bash
bash scripts/force_external.sh success_30k offline --seeds 32
bash scripts/force_external.sh stage_b offline --seeds 32
```

分别与各自7维基线使用相同验证观测和噪声种子，输出为`outputs/pi05_external_success_30k/eval_external_success_30k`、`outputs/eval_external_stage_b`。使用现有hover评估程序，历史文件别名仍是old_aligned_30k/v3_30k，真实checkpoint以manifest为准。它主要比较Z运动方向、位移误差和种子波动；几何筛选可能包含取料观测，不能当作纯放置成功率，也不能证明夹爪释放问题已经解决。

只加载完整模型检查输入配置，不连接硬件（会占用GPU）：

```bash
bash scripts/force_external.sh success_30k model-check
bash scripts/force_external.sh stage_b model-check
```

实时程序已支持从同一31D快照读取实测和外部力矩，共用缓存和时间索引，按checkpoint分别归一化。缺失、错误维度或非有限外部值会报错；不会补零，也不会继续拿旧7维输入给14维模型。`r`重启时同时清空两种力矩历史，重新等待2秒。旧7维checkpoint仍走原输入路径。

由操作者在机器人主机执行观测测试，连接相机和机器人但不发动作、不打开夹爪：

```bash
FORCE_VLA_HARDWARE=1 bash scripts/force_external.sh success_30k observe --max-steps 300
```

换B版时将`success_30k`替换为`stage_b`。确认观测与离线评估后，操作者再运行连续动作测试：

```bash
FORCE_VLA_HARDWARE=1 bash scripts/force_external.sh success_30k execute \
  --startup-open --continuous --rate-hz 8 --inference-steps 10
```

`--startup-open`会在ARM确认前打开夹爪；输入ARM才启用机械臂动作。`q`、`x`、Esc或Ctrl+C退出，`r`清空历史和旧动作后重新推理。这里8Hz是请求频率，不保证GPU达到8Hz。沿用smooth配置和现有运动/停止/夹爪逻辑；本次没有实现上次诊断建议的夹爪防反转与就位判断，添加外部力矩不会自动解决这两项问题。

两版日志分别为`outputs/pi05_external_success_30k/realtime_external_success_30k.jsonl`和`outputs/realtime_external_stage_b.jsonl`。曲线监视命令：

```bash
bash scripts/run.sh python -m force_vla.pi05.telemetry_dashboard \
  --log outputs/pi05_external_success_30k/realtime_external_success_30k.jsonl \
  --output outputs/dashboard_external_success_30k
```

打开`outputs/dashboard_external_success_30k/index.html`。B版相应替换日志和输出目录。若使用本页新的20261020测试日志，请同步替换`--log`。界面区分模型预测与传感器反馈，夹爪二值目标看新日志`sent_gripper`或`command_debug.commanded_gripper`；2026-10-20前的旧日志仍保留原请求值口径。

## 验证边界

2026-10-19准备阶段进行了CPU回归、同一时间索引的14维因果采样、与离线训练归一化的数值一致性、缺失外部输入拒绝、旧7维兼容、外部输入投影迁移及缩小模型接口测试。当时尚未执行正式训练或完整模型GPU测试。2026-10-20旧成功版已经训练完成，新的完整模型离线评估结果见本页开头引用的分析目录；真机效果仍未验证，不能保证纠偏成功。

本轮29项CPU测试通过；另分别读取旧数据episode1/frame59、新数据episode0/frame59的两路真实视频帧和任务文本，确认离线训练与实时预处理产生的图像、文本token、10D状态、10×14历史数值一致。这是两个实际窗口的接口检查，不是全视频质量评估。
