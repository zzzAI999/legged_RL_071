# Dogleg机器人模型：源URDF、网格与训练资产

训练项目现已附齐修正源URDF、其19个STL、机构配置，以及5.5kg模型来源场景和该场景的完整网格依赖。从仓库克隆后，保持本目录结构即可解析相对网格路径。

## 从哪里打开

- `urdf/dogleg_demo_finished_motor_belt_tree.urdf`：标准树结构URDF，建议先查看这个文件。包含19个link、18个joint、4个同步带mimic和4个电机传动定义。
- `urdf/dogleg_demo_finished_motor_belt_closed_isaac.urdf`：同一树结构加4个非标准球形`loop_joint`扩展。通用URDF查看器通常不会解这些闭链，文件名不代表已经验证任何Isaac导入器能直接支持它。
- `meshes/`：URDF引用的完整19个二进制STL。URDF使用`../meshes/...`，不要只下载一个URDF文件而丢掉旁边目录。
- `mechanism/motor_config.json`：同步带比、方向及4个闭链端点。文件中原Windows路径是来源记录，不是当前训练需要访问的路径。
- `mujoco/dogleg_ground.xml`：构建训练物理参数时使用的5.5kg来源场景。`mujoco/meshes/`保存其全部19个文件，包括当时采用的`base_link.obj`；`mujoco/motor_config.json`为相同机构配置。保留原字节和相对依赖，STL的重复副本可在Git对象中共享内容。
- `provenance/`：已有源文件哈希清单和5.5kg模型构建记录。

支持标准URDF及STL相对路径的查看器可打开主URDF。部分ROS查看器需要在ROS包中配置资源查找或先展开mimic；本次发布没有新增并验证ROS显示节点。直接浏览模型无需安装Isaac。MuJoCo场景属于可选的模型来源资料，Isaac训练不读取它，也无需为训练安装MuJoCo。

## 当前训练到底读哪个模型

正式入口读取仓库中的：

```text
src/dogleg_rl/assets/dogleg_lab.usda
src/dogleg_rl/assets/physical_payload.json
src/dogleg_rl/assets/reset_atlas.json
```

USD已内嵌网格，不依赖本机旧工程或Navila目录。URDF和来源XML让项目能查看、核对和维护模型；开始训练时不需要转换模型。每个文件的完整SHA见[资产清单](../../docs/robot_assets_manifest.json)。

## 源URDF与训练物理的差别

源URDF的总惯性质量约**4.808793095kg**，训练使用后续构建的**5.5kg**模型、已求解站姿、接触表示和约束。来源链为：

```text
CAD源 → 修正同步带／闭链的URDF与机构配置
      → 5.5kg dogleg_ground.xml
      → physical_payload.json
      → dogleg_lab.usda → 当前站立训练环境
```

原CAD导出中有14个有问题的mimic关系，本目录使用已修正、只保留4个预期同步带关系的版本。源URDF中的`effort=0.2`、`velocity=2.0`来自历史固定基座传动测试，不是实车电机额定值，也不是正式站立训练的控制上限。请勿直接把源URDF替换运行USD并假定动力学及检查点仍等价。

现有来源XML与runtime payload的`source_sha256`相同；运行USD、物理参数、复位表保持原训练固定哈希。此发布只补齐模型源文件，不改变当前动力学、奖励或控制配置。构建记录是历史记录，不能作为此次重新运行仿真或完成实车标定的证明。

## 克隆后的检查

在仓库根目录运行，无需第三方Python包、Isaac或GPU：

```bash
python scripts/check_robot_assets.py
```

检查内容包括45个新增模型来源文件和3个运行资产的SHA／字节数、URDF唯一连接树、正质量及正定惯量、所有网格路径、4个mimic／4个传动、STL三角形计数，以及来源XML与运行物理参数的对应关系。正常输出含`passed: true`。

软件环境仍按[项目README](../../README.md)和[完整安装说明](../../docs/INSTALL.md)安装。仓库保留项目源码、机器人模型与已审基线检查点；Isaac Sim、Isaac Lab、Python环境和CUDA依赖由安装流程准备。后续服务器大检查点与逐帧记录不在本源码仓库内。
