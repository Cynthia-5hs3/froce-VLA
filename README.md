# froce-VLA：融入力反馈的VLA训练实验

工作目录：`/data0/wx/force-VLA`。目标是在PI0.5中加入力历史输入和未来力预测辅助损失。

本地训练结果按五版模型归入[outputs中各自的训练根目录](outputs/README.md)；[分类清单](outputs/model_classification.json)记录训练、日志和评估产物的新旧路径。五版最终checkpoint路径保持不变。

减少真机试错：[真实观测回放、松爪诊断与MuJoCo仿真方案](docs/MUJOCO_OFFLINE_EVALUATION_PLAN.md)。已确认旧成功LoRA30k为优先基线，阶段A3k为放置/释放对照；仿真尚未实施。

2026-10-21整理入口：[各组训练产物分类](outputs/README.md)、[抓放判断与ForceVLA/ForceVLA2复现评估](docs/FORCE_METHODS_REVIEW_20261021.md)。各模型的早期运行、预检、日志和报告已归入对应现有训练根目录，共享窗口和跨模型评估保持原位；标注网页、草稿工具和临时审阅产物已归档后删除，训练依赖与已确认标注保留。[归档与恢复说明](archives/stage_cleanup_20261021/README.md)。

外部关节力矩实验`success_30k`已于2026-10-20完成30k训练，模型在`outputs/pi05_external_success_30k/checkpoint-030000`；18k为本轮验证动作loss最低的候选。它沿用旧101条成功示范，从PI0.5基座重训；`stage_b`保留A组的已批准纠偏窗口和初始化，计划3k，尚未启动。两者输入过去2秒10点的`tau_J + tau_ext_hat_filtered`（10×14），未来输出仍为50×10动作和50×7实测力矩。完整模型检查命令为`bash scripts/force_external.sh success_30k model-check`（占用GPU，不连接机器人）。部署时选择整个checkpoint目录，入口自动加载14维历史与归一化，不单独替换权重文件；参数说明、18k/30k测试指令见 [两套外部力矩实验](docs/EXTERNAL_TORQUE_VARIANTS.md)。仅检查数据可运行`bash scripts/force_external.sh success_30k check`或`bash scripts/force_external.sh stage_b check`。

累计30000步扩展训练使用独立入口 `scripts/train_force_pi05_30k.sh`，继承第3000步并增加轨迹均衡验证、完整采样推理及力矩历史对照。预算、启动和指标解释见 [三万步训练说明](docs/FORCE_TRAINING_30K.md)。训练结束不自动等同于真机部署就绪。

真机测试使用独立的观测/动作入口，步骤和安全门控见 [实时部署说明](docs/FORCE_REALTIME_DEPLOYMENT.md)。原 Evo-RLT 部署器不直接加载 force adapter checkpoint。

第一轮3000步训练已完成，模型位于 `outputs/pi05_force_sft_30k/pi05_force_sft_v1/checkpoint-003000`，使用独立Conda前缀 `env/`。旧训练入口 `scripts/train_force_pi05.sh` 保留；新运行必须指定未使用的输出目录。数据划分、归一化、环境检查及保存/恢复见 [训练说明](docs/FORCE_TRAINING.md)。旧 `src/lab.py smoke` 仅为早期六维MLP测试。

## 新样本采集、训练和推理

下面的流程只使用 `/data0/wx/force-VLA` 的代码和输出目录。采集命令会连接真实机器人，因此必须先确认急停、工作区和相机状态；它不会写入 `/data0/sht/Evo-RLT`。采集使用原 Evo-RLT 的硬件依赖，但输出固定写入新的 `data/franka_single_left_31d_v2`，避免与已清洗的旧批次混用。

### 1. 启动31D手柄遥操采集

在机器人主机的交互终端执行，不要使用 `scripts/run.sh`（该启动器会隐藏硬件设备）：

```bash
cd /data0/wx/force-VLA
env -u PYTHONPATH \
  PYTHONPATH=/data0/wx/force-VLA/src:/data0/wx/force-VLA/reference/evo-rlt:/data0/wx/force-VLA/reference/evo-rlt/src \
  CUDA_VISIBLE_DEVICES='' \
  /data0/wx/force-VLA/env/bin/python -m collection_31d.gamepad_record_31d \
  --config /data0/wx/force-VLA/configs/collection_31d/single_left_gamepad_31d.yaml \
  --execute \
  --dataset-root /data0/wx/force-VLA/data/franka_single_left_31d_v2 \
  --task 'Place the white lid firmly onto the jig.'
```

启动后先按 `s` 启用遥操，再按 `n` 开始一条示范。`RB` 切换关键阶段标记；完成后按 `y` 保存成功，按 `x` 丢弃当前示范，异常或失败示范按脚本提示保留失败标记。结束采集按 `c` 或 `Esc`；未提交的轨迹不会作为完整样本保存。若发生录制故障，先退出程序，再对同一目录使用 `--resume` 恢复；正常新增批次不要使用 `--resume`。

采集完成后，将成功、失败和控制中断分别记录到 `annotations/31d_classification_v2/`。成功修正示范可用于SFT；只有失败结果、没有正确修正动作的轨迹不要直接作为成功SFT目标，但可作为失败/恢复评估数据保留。

### 2. 重建窗口和数据合同

新旧数据混训前必须重新构建窗口、划分episode和计算训练集归一化统计。默认只把新批次的成功样本写入新窗口目录：

```bash
cd /data0/wx/force-VLA
bash scripts/run.sh python src/build_force_vla_windows.py \
  --dataset-root data/franka_single_left_31d_v2 \
  --classification annotations/31d_classification_v2/sft_candidate.json \
  --output-dir outputs/force_vla_31d_windows_v2 \
  --future-steps 50 --fps 30
bash scripts/run.sh python -m force_vla.pi05.train_pi05_force \
  --prepare \
  --config configs/pi05_force_tavla_aligned.json \
  --windows outputs/force_vla_31d_windows_v2/windows.parquet \
  --output outputs/pi05_force_prepared_v2
```

若要混合旧的101条成功样本和新样本，应先生成合并后的窗口文件，再把该文件同时传给 `--windows`；不能只替换原始数据目录，也不能用旧 `data_contract.json` 恢复新增数据后的训练。

### 3. 训练力反馈VLA

下面从本地 PI0.5 SFT 基座初始化，使用LoRA训练力矩适配器、状态融合和相关输出层；不会修改 `models/pi05_sft`。将 `--windows` 和 `--contract` 指向同一批次对应的文件，并使用新的输出目录：

```bash
cd /data0/wx/force-VLA
bash scripts/train_force_pi05.sh \
  --train \
  --config configs/pi05_force_tavla_aligned.json \
  --windows outputs/force_vla_31d_windows_v2/windows.parquet \
  --contract outputs/pi05_force_prepared_v2/data_contract.json \
  --output outputs/pi05_force_new_run
```

训练使用视觉、任务文本、10D状态、过去2秒10个7D `tau_J` 力矩历史，预测未来50步10D动作并计算未来力矩辅助损失。训练前可把 `--steps 30000` 加入命令覆盖配置中的步数；新数据合同改变后不要直接使用旧checkpoint的普通 `--resume`，应从 PI0.5基座开始新的混合数据训练，或单独实现经过验证的增量恢复。

### 4. 真机推理测试

需要手柄纠正高度或对准时，在下方execute命令追加`--teleop-assist --teleop-latched --teleop-linear-speed 0.02 --teleop-angular-speed 0.10`。按LB或操作手柄接管，松手仍保持人工模式；控件回中后按RB交回模型，再等待2秒新力矩历史。Y/A升高/降低，左摇杆调整XY，LT/RT请求开/闭爪。仅记录运行日志，不微调模型；完整指令和按键见[外部力矩部署步骤](docs/EXTERNAL_TORQUE_VARIANTS.md)。

实时曲线界面：另开终端运行 `bash scripts/run.sh python -m force_vla.pi05.telemetry_dashboard --log outputs/pi05_external_success_30k/realtime_external_success_30k_test_20261020.jsonl --points 1800`，在本机浏览器打开 `outputs/realtime_dashboard/index.html`。请把`--log`改成当前推理实际使用的文件。页面可查看毫米级TCP位置、Roll/Pitch/Yaw角度、7个关节角、10D动作、7维实测/预测力矩和外部关节力矩，人工接管区间以淡橙色背景标识；推理终端不逐帧刷数字。关节角需更新后重新启动推理才能记录，旧日志缺失项留空。完整操作及最近日志选择命令见[外部力矩部署步骤](docs/EXTERNAL_TORQUE_VARIANTS.md)。

训练完成后先使用 `--mode observe` 验证相机、状态和模型加载，再执行动作。下面的命令会先打开夹爪，输入 `ARM` 后持续推理；按 `q`、`x` 或 `Esc` 停止。`FORCE_VLA_HARDWARE=1` 是硬件访问开关，部署命令不要在机器人未准备好时执行：

```bash
cd /data0/wx/force-VLA
bash scripts/force_external.sh success_30k model-check
FORCE_VLA_HARDWARE=1 bash scripts/force_external.sh success_30k observe --max-steps 300
FORCE_VLA_HARDWARE=1 bash scripts/force_external.sh success_30k execute \
  --startup-open --continuous --rate-hz 8 --inference-steps 10 \
  --log outputs/pi05_external_success_30k/realtime_external_success_30k_test_20261020.jsonl
```

上述命令明确选择外部力矩success_30k模型：两路图像、任务文本、10D状态、过去2秒10点14维实测＋外部关节力矩，归一化从该checkpoint自动加载。直接调用旧`run_force_realtime.sh`且不传checkpoint时仍默认A组7维模型，不能与本命令混淆。模型预测的未来力矩只用于辅助输出和日志，不直接发送为力控命令；实时观测异常会暂停重试，超过安全超时才退出。`--startup-open` 会在输入ARM前打开夹爪。持续模式由操作者按 `q`、`x`、`Esc` 或 `Ctrl+C` 结束；按 `r` 清空旧动作和历史并重新等待2秒。短时测试可把 `--continuous` 换成 `--max-steps 30 --hold-after-steps`。18k候选通过`--checkpoint outputs/pi05_external_success_30k/checkpoint-018000`选择，并使用独立日志。

## 当前31D采集与仓库范围（2026-09-15）

本项目 GitHub：`https://github.com/Cynthia-5hs3/froce-VLA`（远端名称为 `froce-VLA`）。
仓库保存代码、配置、文档、来源清单及下载的论文参考源码；数据、模型权重、Conda环境、缓存、实验输出和旧工程完整副本仅保留本地，不纳入Git。
`third_party/lerobot_6674e36/tests/data/` 的上游测试样本仅保留本地，不纳入Git；其中LFS文件仅有来源归档内的指针文本，没有对应实体。测试源码仍保留，运行依赖这些样本的上游测试前需另行从上游获取实体；来源清单描述的是下载归档，不代表测试样本已上传至本仓库。
克隆仓库不会自动获得上述本地资产；原文件位置见下表、`configs/collection_31d/PROVENANCE.md` 和 `provenance/`。运行 `scripts/run.sh` 前需独立准备本地环境和所需数据、模型及参考目录。

旧31D数据位于 `data/franka_single_left_31d`，来源为 `/data0/sht/Evo-RLT/datacollection/franka_robotiq_single_left/rollout_dataset_franky_31d`。历史上按重新采集要求做过清理；2026-10-19重新检查确认旧成功窗口及其引用的202个视频文件存在，可用于上述`success_30k`实验，不能据此认定原184条轨迹全部仍完整。新采样应写入上文的 `data/franka_single_left_31d_v2`。
历史元信息记录184条、177384帧、30Hz，两路RGB视频、31D状态与10D动作，包含成功、失败及控制中断记录；分类清单保留原始追溯信息，训练前仍需完成视频、时间戳和力矩同步检查。
31D状态：TCP位置3 + rot6d姿态6 + 夹爪宽度1（米）+ measured joint torque 7（Nm）+ external joint torque 7（Nm）+ joint velocity 7（rad/s）。动作的夹爪通道为开度比例。
对齐论文力矩路线时，按时间戳构建过去约2秒的10个力矩样本以及未来50步动作/力矩目标，不跨episode边界；采集时间同步与可用窗口仍需验证。

`src/collection_31d/` 和 `configs/collection_31d/` 是独立采集代码及格式来源副本，保留Evo-RLT依赖，不能视为克隆后即可独立运行的采集器。
数据分类清单位于 `annotations/31d_classification/`：当前包括101条成功SFT候选、30条人工失败候选、51条控制器异常和2条录制时序异常；新增批次为episode 152-183，其中episode 170因 `cartesian_reflex` 明确排除。
`configs/pi05_force_sft.json`、`src/force_vla/pi05/` 和 `src/force_vla/pi05/train_pi05_force.py` 已接通七关节力矩SFT：整轨迹划分、训练集归一化、视觉文本输入、优化与保存恢复。真实PI0.5两步验证已通过；训练收敛和任务效果仍需正式实验。
以下29D数据、六维wrench方案及相关检查描述保留为早期实验记录；成功/失败标签本身也不代表数据已满足RLT训练条件。

## 需要的数据

每个训练样本需要两路RGB图像、有效任务文本、本体状态、过去2秒的六维力/力矩历史、未来50步动作及对应力信号。
现有原始数据为30Hz LeRobot v3：29D状态、10D绝对动作，两路224×224 RGB视频。
29D状态由TCP位置3、rot6d姿态6、夹爪1、力/力矩6、线/角速度6、关节位置7组成；动作是目标位置3、姿态6和夹爪1。
状态`[10:16]`是Franka `O_F_ext_hat_K`外力估计，`[22:29]`是七关节位置；没有论文使用的逐关节电机力矩。
训练时保留episode边界，屏蔽无效动作和时间缺口，按episode划分验证集，归一化统计只用训练集。

| 本目录副本                                                                        | 原文件路径                                                                                 | 数据内容与用途                                                                          |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- |
| `data/franka_single_left_31d_v2`                                                  | 当前目录新采集目标                                                                  | 新31D成功/失败示范；成功修正轨迹用于SFT，失败轨迹用于恢复评估和后续筛选                 |
| `outputs/force_vla_31d_windows_v2`                                                | 由 `data/franka_single_left_31d_v2` 构建                                              | 过去10帧力矩到未来50步动作/力矩的训练窗口                                           |
| `reference/rlt/replay.sqlite3`                                                    | `/data0/sht/Evo-RLT/outputs/ac/gamepad_task3_20260909_online_c20_full/replay.sqlite3` | 26,427条transition；压缩状态不含原始力，供RLT流程参考                                   |

此前复制到 `datasets/` 的旧SFT/RLT数据已删除，不应按旧表格路径运行；原始来源路径只保留在历史 provenance 和文档中，当前训练以 `data/franka_single_left_31d_v2` 为准。

RLT及旧SFT的任务文本是`Place the white lid firmly onto the jig.`。失败轨迹不能全部作为成功示范做行为克隆。
新标注写入`annotations/`，不改原始副本。

## 模型框架与来源

参考arXiv:2509.07962v1 TA-VLA：将力历史压缩为一个token注入动作解码器，同时预测未来动作与力信号。
本实验采用 PI0.5 + `10×7` measured joint torque 历史适配器 + 状态融合 token + action expert 注入 + `50×7` 未来关节力矩辅助损失；机器人执行动作仍保持10D。
PI0.5把状态离散化为文本，简单改成22D/29D不等价于论文方案。机器人执行动作保持10D，预测的力不作为力控命令。

| 本目录副本                  | 原文件路径                                                                                                                    | 内容与用途                                                             |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `models/pi05_sft`           | `/data0/sht/Evo-RLT/outputs/sft/franka_single_left_gamepad_pi05_sft_task3_20260908_clean/checkpoints/073026/pretrained_model` | PI0.5完整SFT权重及预处理统计；力反馈适配的初始化骨干                   |
| `models/tokenizer`          | `/data0/sht/Evo-RLT/models/paligemma` → `/data0/sjh/lerobot_v44/models/paligemma`                                             | 仅复制tokenizer及必要配置，离线处理文本；不复制重复的PaliGemma模型权重 |
| `models/rl_token_reference` | `/data0/sht/Evo-RLT/outputs/rltoken/gamepad_task3_20260908_rl_token/checkpoints/010000/pretrained_model`                      | 已有RL Token权重和配置，供后续RLT适配参考                              |
| `models/ac_reference`       | `/data0/sht/Evo-RLT/outputs/ac/gamepad_task3_20260909_online_c20_full/checkpoints/checkpoint-1789021365979913376`             | Actor-Critic最后快照；不是完整VLA                                      |
| `reference/evo-rlt`         | `/data0/sht/Evo-RLT/{src,tests,scripts,datacollection,README.md,AGENTS.md,pyproject.toml,uv.lock}`                            | 源码、测试、配置及文档快照，供离线实现参考；无原Git数据库              |

历史checkpoint配置保持原样，内部可能有失效旧路径；运行时必须显式使用本目录的checkpoint和tokenizer路径。
逐文件原路径、目标路径、SHA256和字节数见`provenance/assets.json`，未列出的基础模型和历史checkpoint没有复制。

## 环境与配置

- 原环境：`/data0/miniconda3/envs/evo-rlt`；新环境：`/data0/wx/force-VLA/env`，离线`--clone --copy`独立复制。
- 依赖基线：Python 3.12、PyTorch 2.10.0 + CUDA 12.8、LeRobot 0.5.1。当前GPU为RTX 4090；冻结骨干、batch=1的完整模型短程验证，PyTorch显存分配峰值约9.0 GiB，不能外推到全量微调。
- 实际软件包版本清单：`provenance/python-packages.json`。
- 配置：`configs/pi05_force_sft.json`。历史2秒/10帧、measured joint torque 7D、动作10D、未来50步；辅助损失权重0.1是待验证起点。
- 新代码放`src/`，结果放`outputs/`；HOME、Conda/pip/HF/PyTorch缓存均在`runtime/`。
- 必须经`scripts/run.sh`运行：原工作区和原环境强制只读，清除ROS/Python环境继承，默认断网并隐藏机器人设备。副本数据、模型和参考源码在实验时也只读。

## 检查与训练连通性测试

复核28条阶段标注后的采样规则，以及“实测关节力矩 / 实测+外部关节力矩”独立训练配置与指令，见[阶段标注与外部力矩A/B实验](docs/STAGE_EXTERNAL_TORQUE_AB_20261016.md)。第14条已补全；新实验尚未启动。

```bash
cd /data0/wx/force-VLA
bash scripts/run.sh python src/lab.py isolation
bash scripts/run.sh --gpu python src/lab.py environment
bash scripts/run.sh python src/lab.py inspect
bash scripts/run.sh python -m unittest discover -s tests -v
bash scripts/run.sh python src/lab.py smoke --steps 20
```

`inspect`生成`outputs/data_readiness.json`。`smoke`用真实力历史训练一个小型未来力预测MLP，验证读取、窗口构建和反向传播；不加载PI0.5、不使用图像语言，不代表已完成VLA训练。
GPU实验在启动器后加`--gpu`，例如`bash scripts/run.sh --gpu python src/lab.py smoke --device cuda`。
后续实验：开展无力/单帧力/力历史/联合预测消融；旧80条RLT选择仍需确认，与本次31D成功示范SFT无关。

## 论文代码与新增采集项

已下载官方TA-VLA固定版本至`third_party/ta_vla_149d4be/`，并下载其固定LeRobot依赖源码至`third_party/lerobot_6674e36/`；仅保存源码，没有安装依赖或启动训练。
官方实现为JAX/Flax π0，不能直接加载现有PyTorch π0.5权重。
若对齐论文关节力矩路线，应新增Franka七维`tau_J`，建议同时保存`tau_ext_hat_filtered`及同步时间；现有29D保留，TCP速度6D可暂不作为基础模型输入。
29维取舍、字段来源、官方代码入口与采样/归一化差异见[采集与复现说明](docs/TA_VLA_COLLECTION_AND_REPRODUCTION.md)；采集配置草案见`configs/collection_force_proposal.json`，尚未应用到采集器。
