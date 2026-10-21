# outputs：按五版模型训练根目录归类

五版模型的专属产物已实际归入各自现有训练根目录，不新增外围分组目录。最终checkpoint路径保持不变；早期运行、预检、日志和报告保留原文件名，放到对应根目录内。

完整新旧路径映射及校验信息见 [model_classification.json](model_classification.json)。

| 模型 | 根目录（相对outputs） | 最终checkpoint |
| --- | --- | --- |
| 早期适配器30k（历史基线） | `pi05_force_sft_30k/` | `checkpoint-030000` |
| 旧成功LoRA 30k（用户反馈最佳基线） | `pi05_force_tavla_aligned_30k_resume/` | `checkpoint-030000` |
| 纠偏v3 LoRA 30k | `pi05_force_v3_resume_from_014000/` | `checkpoint-030000` |
| 阶段A 3k（放置/释放对照） | `pi05_stage_measured_3k/` | `checkpoint-003000` |
| 旧成功＋外部力矩LoRA 30k | `pi05_external_success_30k/` | `checkpoint-030000` |

## 各模型目录内容

每个根目录原有的checkpoint、run.json、metrics.jsonl和验证结果保持原位。以下列出归入该根目录的其他产物：

### 早期适配器30k（历史基线）

根目录：`outputs/pi05_force_sft_30k/`

- `pi05_force_sft_v1`
- `pi05_force_sft_30k.console.log`
- `pi05_force_30k_preflight`
- `pi05_force_smoke_20260920`
- `phase_audit_20260921_loss.json`
- `phase_audit_20260921_samples.json`

### 旧成功LoRA 30k（用户反馈最佳基线）

根目录：`outputs/pi05_force_tavla_aligned_30k_resume/`

- `pi05_force_tavla_aligned_30k`
- `pi05_force_tavla_aligned_30k_resume.log`
- `pi05_force_tavla_aligned_smoke`
- `ta_vla_alignment_audit_20260921.md`
- `ta_vla_alignment_preflight_20260921.json`

### 纠偏v3 LoRA 30k

根目录：`outputs/pi05_force_v3_resume_from_014000/`

- `pi05_force_v3_run`
- `force_vla_v3_train.log`
- `force_vla_v3_resume_train.log`
- `force_vla_v3_resume.pid`
- `realtime_execute_v3_continuous.jsonl`
- `realtime_execute_v3_step1_retry.jsonl`

### 阶段A 3k（放置/释放对照）

根目录：`outputs/pi05_stage_measured_3k/`

- `pi05_stage_measured_3k.log`
- `stage_A_analysis_20261019`
- `realtime_A_diagnosis_20261019`
- `realtime_check_stage_A_3000_20261019.log`
- `realtime_stage_A_3000.jsonl`

### 旧成功＋外部力矩LoRA 30k

根目录：`outputs/pi05_external_success_30k/`

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

## 共享与待确认产物

以下产物保持原路径，不重复归入某个模型。共享预处理窗口不是独立预训练权重；PI0.5基础模型仍在项目models目录。

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

## 路径与校验说明

- 32项产物、232个移动文件在移动前后逐文件SHA-256校验一致；之后仅调整3个分析脚本及1份准备配置中的路径。分类JSON记录调整后的当前哈希。
- 五版最终checkpoint路径、共享窗口及基础模型路径不变；模型权重、原始推理日志和历史训练元数据未修改。
- 启动脚本、相关配置、说明文档和仪表盘默认日志路径已更新。success_30k后续默认推理日志和离线评估也写入该模型根目录。
- 原始日志、run.json、checkpoint内force_config.json及数值报告中的历史路径保留作追溯，使用分类JSON的moves查找当前位置。
- 严格断点恢复会比较历史settings。需要恢复时使用对应checkpoint内force_config.json，以--contract显式指定当前合同路径，并使用新的--output目录；不要直接修改历史checkpoint元数据。
- 根目录哈希包含其所有嵌套产物，因此根目录与子产物的字节数不可重复相加。
- 未训练B组、技术冒烟/预检和跨模型评估不计为新模型；旧PID文件不代表进程仍在运行。
- GitHub保存分类说明、索引及代码配置；训练权重、原始日志、视频和本地分析产物仍留在本机。
