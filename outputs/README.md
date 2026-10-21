# outputs：五版模型产物原位分类

所有训练目录、checkpoint、日志和报告均保留原来的文件名与路径。分类通过本页及[model_classification.json](model_classification.json)表达，不新增分类文件夹，不移动文件，不保留兼容链接。原训练与推理命令直接使用原路径。

| 模型类别 | 最终checkpoint（相对outputs） |
| --- | --- |
| 早期适配器30k（历史基线） | `pi05_force_sft_30k/checkpoint-030000` |
| 旧成功LoRA 30k（用户反馈最佳基线） | `pi05_force_tavla_aligned_30k_resume/checkpoint-030000` |
| 纠偏v3 LoRA 30k | `pi05_force_v3_resume_from_014000/checkpoint-030000` |
| 阶段A 3k（放置/释放对照） | `pi05_stage_measured_3k/checkpoint-003000` |
| 旧成功＋外部力矩LoRA 30k | `pi05_external_success_30k/checkpoint-030000` |

顺序用于分类，不代表真机测试时的版本编号。旧成功LoRA30k是用户反馈较好的基线；阶段A3k是放置/释放对照；两者都不同于早期仅适配器版。

## 早期适配器30k（历史基线）

- `pi05_force_sft_v1`
- `pi05_force_sft_30k`
- `pi05_force_sft_30k.console.log`
- `pi05_force_30k_preflight`
- `pi05_force_smoke_20260920`
- `phase_audit_20260921_loss.json`
- `phase_audit_20260921_samples.json`

该版本数据来源、划分和训练设置以对应训练目录的`run.json`及checkpoint中的配置为准。

## 旧成功LoRA 30k（用户反馈最佳基线）

- `pi05_force_tavla_aligned_30k`
- `pi05_force_tavla_aligned_30k_resume`
- `pi05_force_tavla_aligned_30k_resume.log`
- `pi05_force_tavla_aligned_smoke`
- `ta_vla_alignment_audit_20260921.md`
- `ta_vla_alignment_preflight_20260921.json`

该版本数据来源、划分和训练设置以对应训练目录的`run.json`及checkpoint中的配置为准。

## 纠偏v3 LoRA 30k

- `pi05_force_v3_run`
- `pi05_force_v3_resume_from_014000`
- `force_vla_v3_train.log`
- `force_vla_v3_resume_train.log`
- `force_vla_v3_resume.pid`
- `realtime_execute_v3_continuous.jsonl`
- `realtime_execute_v3_step1_retry.jsonl`

该版本数据来源、划分和训练设置以对应训练目录的`run.json`及checkpoint中的配置为准。

## 阶段A 3k（放置/释放对照）

- `pi05_stage_measured_3k`
- `pi05_stage_measured_3k.log`
- `stage_A_analysis_20261019`
- `realtime_A_diagnosis_20261019`
- `realtime_check_stage_A_3000_20261019.log`
- `realtime_stage_A_3000.jsonl`

该版本数据来源、划分和训练设置以对应训练目录的`run.json`及checkpoint中的配置为准。

## 旧成功＋外部力矩LoRA 30k

- `pi05_external_success_30k`
- `pi05_external_success_30k.log`
- `pi05_external_success_30k.exit`
- `external_success_30k_prepared`
- `external_success_analysis_20261020`
- `grasp_review_20261020`
- `realtime_external_success_30k_20261020_102833.jsonl`
- `realtime_external_success_30k_20261020_103246.jsonl`
- `realtime_external_success_30k_20261020_103525.jsonl`
- `realtime_external_success_30k_assist_20261020_104857.jsonl`
- `realtime_external_success_30k_assist_20261020_110449.jsonl`

该版本数据来源、划分和训练设置以对应训练目录的`run.json`及checkpoint中的配置为准。

## 共享产物和归属待确认的日志

### 共享数据窗口、训练合同和数据审查

- `classification_31d_update.txt`
- `data_readiness.json`
- `force_vla_31d_v2_torque_report`
- `force_vla_31d_windows`
- `force_vla_31d_windows_v2`
- `force_vla_31d_windows_v3`
- `force_vla_prepared_v2`
- `force_vla_prepared_v3`
- `label_audit.txt`
- `pi05_force_prepared`
- `stage_external_ab_20261016`
- `stage_index_v3_20261016`
- `stage_index_v3_20261016_round2`

### 阶段B准备文件（尚未训练，不计入五版）

- `external_stage_b_prepared`

### 旧成功LoRA与纠偏v3的跨模型评估

- `hover_ab_20261012`
- `hover_ab_20261012.log`
- `hover_ab_20261012_summary.log`

### 环境检查、历史备份、通用监控与冒烟检查

- `environment`
- `environment_check.json`
- `force_smoke_1789355844063981384.json`
- `git_pre_alignment_20260921`
- `realtime_dashboard`
- `realtime_dashboard_monitor.log`
- `torque_monitor_smoke.txt`
- `verification.json`

### 缺少明确checkpoint元数据的历史推理日志

不凭日期或简写把日志断定为某个模型；保留原文件待核对。

- `realtime_async_teleop_8hz.jsonl`
- `realtime_execute_030000_continuous.jsonl`
- `realtime_execute_030000_open_continuous.jsonl`
- `realtime_execute_030000_speed_test.jsonl`
- `realtime_execute_030000_step1_20260922.jsonl`
- `realtime_execute_030000_step1_20260922_retry.jsonl`
- `realtime_execute_030000_step1_manual.jsonl`
- `realtime_execute_async_8hz.jsonl`
- `realtime_execute_continuous.jsonl`
- `realtime_execute_gripper_open.jsonl`
- `realtime_execute_motion_test.jsonl`
- `realtime_execute_smooth_test.jsonl`
- `realtime_execute_speed_test.jsonl`
- `realtime_execute_step1.jsonl`
- `realtime_execute_step3.jsonl`
- `realtime_observe_027000.jsonl`
- `realtime_observe_manual.jsonl`

## 校验与使用

- 37项模型产物、904个文件已与整理前SHA-256树哈希逐项核对，内容一致；三个分析脚本已恢复原目录层级。
- 全部中间checkpoint保留；`smoke/preflight`为技术检查，历史PID不代表进程仍运行，B组准备文件不代表已有模型。
- 本页与分类JSON覆盖outputs根目录当前所有产物；共享数据不重复分配，缺少checkpoint元数据的日志不猜测归属。
- GitHub只同步本页和分类JSON，模型权重、原始日志/视频仍在本机。没有新增分类目录，也没有模型目录兼容链接。
