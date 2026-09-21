# 31D 数据分类

数据副本：`/data0/wx/force-VLA/data/franka_single_left_31d`

来源：`/data0/sht/Evo-RLT/datacollection/franka_robotiq_single_left/rollout_dataset_franky_31d`

分类依据为 `meta/episodes` 中的 `episode_success`、`failure_reason` 和帧数。所有分类文件都是只读选择清单，不会删除或改写原始数据。

| 文件 | 数量 | 用途 |
| --- | ---: | --- |
| `sft_candidate.json` | 101 | 成功且长度满足至少一个历史窗口的SFT候选；训练前仍需做视频、时间戳和力矩质量检查 |
| `failure_candidate.json` | 30 | 操作者明确标记失败的轨迹；不作为成功行为克隆目标 |
| `excluded_controller_fault.json` | 51 | Franka控制成功率异常或 `cartesian_reflex`，排除训练候选 |
| `excluded_recording_fault.json` | 2 | 相机/录制时序异常，排除训练候选 |

新增批次为 episode `152-183`：31条成功候选，episode `170` 因 Franky `cartesian_reflex` 排除。原始副本中保留 episode `170`，便于追溯；训练数据读取 `sft_candidate.json` 时不会选中它。

`failure_candidate` 只表示操作标签，不等同于可直接用于RLT的奖励或偏好标签。`failure_reason` 中成功条目的 `discard requested` 是采集器历史文本问题，成功判断以 `episode_success=true` 为准。
