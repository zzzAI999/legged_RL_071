# 29维策略输入记录与实车离线对比

这份记录保存每次50Hz控制策略实际收到的29维float32输入。训练阶段保存全部并行环境，独立评估保存实际评分的512个环境；不抽行、不只留均值。记录点位于演员成功返回之后、该次env.step之前。每帧完成后再附加复位前的标准、奖励和成功标签，因此可以从完整轨迹筛选成功案例。

输入值来自当前传感延迟、噪声、缩放和裁剪之后的策略张量。它们是部署时应对照的网络输入，不是未加工的传感器真值。以前没有保存的逐帧输入不能从检查点恢复。

## 29维顺序与单位

以下索引从0开始，与NPZ的最后一维和CSV的obs_00到obs_28一致。精确的机器可读定义和canonical SHA256在每次运行的observation29/contract.json。

| 索引 | 内容 | 输入换算 |
| --- | --- | --- |
| 0–2 | BODY坐标角速度xyz | rad/s × 0.2 |
| 3–5 | BODY坐标单位重力方向xyz | 无单位，原值 |
| 6–8 | vx、yaw、保留命令 | 分别 × [1, 0.2, 1]；当前站立均为0 |
| 9–12 | 四个腿关节相对于模型零位的位置偏移 | rad × [1, 1, -1, -1] / 0.5 |
| 13–18 | 六个关节速度 | rad/s × [.05, .05, .05, .05, .03, .03] |
| 19–24 | 当前传感延迟环采样到的六路已完成ACK动作 | 归一化动作，保留原ACK相位；不是本次刚生成的新动作 |
| 25–28 | jump附加输入 | 当前站立均为0 |

六电机模型顺序为：left_front1_joint、left_rear1_joint、right_front1_joint、right_rear1_joint、right_rear4_joint、left_rear4_joint。轮速为关节／输出轴rad/s，不是电机转子转速。最终策略输入裁剪为[-100,100]，每帧记录到达裁剪边界的位图；裁剪后的值不能反推唯一原始值。

BODY适配、编码器零位、方向、轮减速比和ACK时间关系须与现有recovery合同核对，不能重复做两遍符号／比例转换。这份记录本身没有证明实车编码器零位或动力学已完成标定。

## 文件与关联信息

每次新运行的output/observation29目录包含：

- contract.json：29维定义、频率、坐标、单位、canonical SHA。
- coverage_start.json：覆盖起始检查点及累计更新数，明确过去输入没有重建。
- models/policy_*.pt：每个rollout实际使用的完整参考检查点，包含模型／Adam／RNG；chunk关联检查点和模型SHA。
- chunk_*.npz：按时间、环境排列的精确float32 observation，以及探索动作action_sampled、确定性网络输出action_mean、实际applied_action_after。
- chunks.jsonl：每块的SHA、字节数、连续帧区间、输入行数、模型身份和训练／评估模式。
- progress.json：已提交frames、rows和零丢行计数。frame指一次控制决策的所有已选环境，row指其中一个环境的29维输入。
- evaluation_labels_*.json：完整独立评估完成后，复制实际evaluation.json的512个case_passed标签。
- recording_receipt.json：正常或异常关闭时的覆盖范围和问题。

每帧还包含env_ids、episode_serial、episode_step_before/after、control_tick_before/after、主机UTC/monotonic时间、sensor_latency／history_index、复位前标准位图、终止／超时／安全状态和训练成功标签。control_tick每个策略决策增加20，对应1ms物理时钟；主机时间含并行仿真与IO耗时，不能直接当作物理时间。

训练原native success和含yaw的全标准成功分别记录。前者沿用原训练实现，不代表独立资格。正式成功轨迹应优先按evaluation_labels中的case_passed筛选。课程0的35.02s通过也不代表最终course4的120s实车部署资格。

## 离线提取一个成功评估案例

此命令只读已有数据，不启动Isaac Sim、不运行模型、不连接硬件。需要已完成的独立评估标签；尚在评估时会拒绝输出。

```bash
python -m dogleg_rl.observation29_compare_v1 \
  --recording /path/to/output/observation29 \
  --env-id 0 \
  --export-csv successful_case_0.csv
```

把env-id改为实际case_passed=true的0–511编号。工具逐块校验SHA和环境身份，要求35.02s共有1751个连续50Hz输入，不接受缺帧、重复或乱序。CSV保留episode_step_before和obs_00到obs_28；9位有效数字足以float32往返。输出文件必须尚不存在，以免覆盖旧比较结果。

完整NPZ会保留在服务器，文件较大。下载时需要同时保留contract.json、chunks.jsonl、evaluation_labels及所选case所属的全部评估chunk；只有几个采样点不能导出完整成功轨迹。模型PT是另一个独立的大文件集合。

## 未来实车输入的对照

实车侧应在实际推理接口采集同样的29个预处理后float32值，并保留模型SHA、合同SHA、原始传感时间、控制决策编号、回合编号及ACK采样时序。不要先把浮点输入变成显示用的低精度字符串。建议同时保存原始IMU／编码器／ACK值，便于定位适配错误；本工具不替代实车采集器，也不发送控制指令。

先确认模型、观测顺序、BODY坐标、单位、关节零位、输出轴轮速、缩放／裁剪和采样相位一致，再根据明确的物理时间／决策编号对齐。CSV至少有以下表头：

```csv
episode_step_before,obs_00,obs_01,obs_02,obs_03,obs_04,obs_05,obs_06,obs_07,obs_08,obs_09,obs_10,obs_11,obs_12,obs_13,obs_14,obs_15,obs_16,obs_17,obs_18,obs_19,obs_20,obs_21,obs_22,obs_23,obs_24,obs_25,obs_26,obs_27,obs_28
```

比较已对齐的实车CSV：

```bash
python -m dogleg_rl.observation29_compare_v1 \
  --recording /path/to/output/observation29 \
  --env-id 0 \
  --real-csv real_aligned_case_0.csv \
  --real-contract-sha256 <contract.json中的canonical_SHA256> \
  --report comparison_case_0.json
```

报告逐维给归一化bias、RMSE、绝对误差P95／最大值、有限值计数、排除裁剪值后的反缩放RMSE，以及float32是否完全一致。缺失对齐行和合同身份是否一致会显式报告。合同SHA不匹配时报告不能当作一致性证明。

不同初态和噪声下，两条闭环轨迹通常不会逐点相等；这些差值是描述统计。先用同一段输入检查预处理和推理一致性，再结合实车时间同步、传感噪声及任务要求制定每维容差。工具不会擅自给“可部署”的总通过结论，也不会自动部署或回放。

## 存储与训练开销

32768环境每更新记录32768×64=2097152行，29个float32输入本身为232MiB／更新，另外还有动作和时序／标签。43更新仅29维原值即约9.74GiB，完整未压缩档案更大，实际压缩比随数据变化。每8个决策帧组成一块，有界队列2；后台无损压缩，队列满时阻塞，不丢行。

每次优化前提交完该rollout的记录。磁盘余量低于8GiB加待写完整块时明确报错并按已有完整优化边界保存停止，不会悄悄禁用记录继续训练。记录会增加CPU、磁盘和部分GPU到CPU同步开销，实际吞吐以本次metrics为准。

当前2026-10-28接续来自原段已保存的第7更新，只保留余下43更新／51.79672047204028分钟；评估仍为累计1133／seed126204，接续local3。此记录改动不改变22项奖惩、物理、控制、PPO和通过阈值。
