# froce-VLA：融入力反馈的VLA训练实验

工作目录：`/data0/wx/force-VLA`。目标是在PI0.5中加入力历史输入和未来力预测辅助损失。

累计30000步扩展训练使用独立入口 `scripts/train_force_pi05_30k.sh`，继承第3000步并增加轨迹均衡验证、完整采样推理及力矩历史对照。预算、启动和指标解释见 [三万步训练说明](docs/FORCE_TRAINING_30K.md)。训练结束不自动等同于真机部署就绪。

真机测试使用独立的观测/动作入口，步骤和安全门控见 [实时部署说明](docs/FORCE_REALTIME_DEPLOYMENT.md)。原 Evo-RLT 部署器不直接加载 force adapter checkpoint。

第一轮3000步训练已完成，模型位于 `outputs/pi05_force_sft_v1/checkpoint-003000`，使用独立Conda前缀 `env/`。旧训练入口 `scripts/train_force_pi05.sh` 保留；新运行必须指定未使用的输出目录。数据划分、归一化、环境检查及保存/恢复见 [训练说明](docs/FORCE_TRAINING.md)。旧 `src/lab.py smoke` 仅为早期六维MLP测试。

## 当前31D采集与仓库范围（2026-09-15）

本项目 GitHub：`https://github.com/Cynthia-5hs3/froce-VLA`（远端名称为 `froce-VLA`）。
仓库保存代码、配置、文档、来源清单及下载的论文参考源码；数据、模型权重、Conda环境、缓存、实验输出和旧工程完整副本仅保留本地，不纳入Git。
`third_party/lerobot_6674e36/tests/data/` 的上游测试样本仅保留本地，不纳入Git；其中LFS文件仅有来源归档内的指针文本，没有对应实体。测试源码仍保留，运行依赖这些样本的上游测试前需另行从上游获取实体；来源清单描述的是下载归档，不代表测试样本已上传至本仓库。
克隆仓库不会自动获得上述本地资产；原文件位置见下表、`configs/collection_31d/PROVENANCE.md` 和 `provenance/`。运行 `scripts/run.sh` 前需独立准备本地环境和所需数据、模型及参考目录。

最新数据为 `data/franka_single_left_31d`，复制自 `/data0/sht/Evo-RLT/datacollection/franka_robotiq_single_left/rollout_dataset_franky_31d`。
元信息记录184条、177384帧、30Hz，两路RGB视频、31D状态与10D动作，包含成功、失败及控制中断记录；分类清单保留原始数据不变，训练前仍需完成视频、时间戳和力矩同步检查。
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
| `datasets/raw/franka_single_left_rgb_block`                                       | `/data0/sht/Evo-RLT/datasets/raw/franka_single_left_rgb_block`                             | 最新100条人工示范、69,481帧；用于带力SFT。文本为`111`，需人工确认标注                   |
| `datasets/raw/gamepad_task3_20260909_online_c20_full`                             | `/data0/sht/Evo-RLT/datasets/raw/gamepad_task3_20260909_online_c20_full`                   | 113条RLT、55,672帧，81成功/32失败，含人工介入；用于接触动态学习及后续RL。未擅自选取80条 |
| `datasets/raw/franka_single_left_gamepad_task3__raw`                              | `/data0/sht/Evo-RLT/datasets/raw/franka_single_left_gamepad_task3__raw`                    | 此前102条29D原始示范、81,052帧；保留已有SFT的数据来源及力通道                           |
| `datasets/sft_reference/franka_single_left_gamepad_task3_20260908_split_seed1000` | `/data0/sht/Evo-RLT/datasets/sft/franka_single_left_gamepad_task3_20260908_split_seed1000` | 旧SFT的92/10训练验证划分；10D状态无力，仅作复现基线参考                                 |
| `reference/rlt/replay.sqlite3`                                                    | `/data0/sht/Evo-RLT/outputs/ac/gamepad_task3_20260909_online_c20_full/replay.sqlite3`      | 26,427条transition；压缩状态不含原始力，供RLT流程参考                                   |

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
