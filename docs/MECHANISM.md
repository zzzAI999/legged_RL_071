# 当前奖励、控制和资格

当前版本为 `standing_v11_body_gyro_weight2_v1`，即 Guard17 使用的版本。base_gyro 的权重从0.5提高为2.0，保持原角速度观测、0.02秒决策间隔和裁剪边界：

```text
base_gyro = -0.02 * 2.0 * min(g^2, 25)
quiet_gyro = +0.02 * 2.0 * G * exp(-g^2/0.09)
G = upright * support * both_wheels * height_gate * clearance_gate
```

`g` 为 BODY 横滚／俯仰角速度范数。g=0.3且G=1时，这两项净值约0.011115；正的绝对净值并不能证明策略被激励去违规，因为该净值随g增大严格下降。是否最终达标还取决于动力学、其它代价、探索和优化，不能由两项盈亏平衡点保证。

当前22项的实现来源：

- `core/standing_math_v6.py`、`standing_math_v7.py`、`standing_math_v8.py` 定义姿态、支撑、速度、角速度、偏航、高度、间隙、动作和关节等项。
- `core/standing_math_guard.py` 对 unsafe 决策去掉正奖金，使用-5的guard终止代价；fall只在未被guard终止时扣-5，避免重复。
- `core/standing_margin_reward.py` 计算1ms腿速度峰值软边界代价。
- `core/standing_leg_velocity_reward_v1.py` 计算20个1ms样本的腿速平方积分代价，权重0.01。
- `core/standing_gyro_penalty_reward_v1.py` 仅替换base_gyro权重2.0，其余21项不变。

上述文件在打包时只调整包内导入和资产路径；数值实现仍使用当前实际依赖链。`docs/source_manifest.json` 记录原文件和打包文件的SHA256。带版本号的基础模块是当前实现仍然依赖的代码，不是额外可选的旧运行方案。

控制保持物理1ms、PD2ms、状态／ACK10ms、策略20ms；腿位置PD、轮直接力矩、原电流量化和超时／速度／行程锁存保持。PPO为每更新4096×64决策、4×4minibatch、gamma0.995、lambda0.95、entropy0.001、学习率限制1e-5至3e-4、tanh均值输出。

标准URDF不能直接完整表达这台机器的闭链和仿真构建。运行使用随仓库提供的实际5.5kg USD，不使用CAD源URDF的4.809kg参数替代。总solver effort、完整friction、closure impulses、warmstart、运行时逐刚体质量等未观测量不因本次打包变成已标定事实。

独立评估固定512，Soft／Protected各256；稳定要求包含倾角≤8度、水平速度≤0.15m/s、横滚俯仰gyro≤0.3rad/s、绝对yaw≤0.3rad/s、根高度≥0.105m、基座间隙≥0.015m、每轮支撑≥2N、总轮支撑≥80%重量、非轮接触≤1N、无安全停止。课程晋级需要连续两次两侧各≥244/256。

最终资格另要求课程4、5秒内进入稳定状态并连续保持至120秒。当前Guard17仅课程0/consecutive1，短评估251/256、256/256；不据此宣称通过最终资格或可实机部署。
