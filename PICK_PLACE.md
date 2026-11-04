# SO101：把方块放入盘子

入口 `pick_place.py`，配置 `pick_place_config.json`。对应课件：ArUco + solvePnP 定位 → FK/手眼变换 → TCP 偏移 + 数值 IK + 分段轨迹 → LeRobot 执行和人工结果检查。已经填入 robot ID `my_awesome_follower_arm`、相机 `0`、串口 `/dev/tty.usbmodem5B421342761`。

**当前标定不能直接用于执行**：现有 handeye.json 固定板位置 RMS=31.68 mm、旋转 RMS=5.80°；camera.yml 的像素 RMS=0.243 px。前者是厘米级位姿不一致。先检查关节零位、URDF 对应关系、板/底座是否移动、腕部相机是否松动，再重新采样。

主要原因：LeRobot `use_degrees=True` 的 0° 是电机标定时手动摆动范围的中点，不是 URDF 零位。`handeye_calibration.py` 现在会在求解时同时拟合 shoulder_lift / elbow_flex / wrist_flex 的零点偏移（首尾关节偏移不可观测，固定为 0），写入 handeye.json 的 `joint_offsets_deg`，`pick_place.py` 读取后自动加到 FK 输入上。对已有样本离线重算：`uv run python handeye_calibration.py --solve-only handeye_output/samples.json`，对比输出里的 `before_offset_fit` 和新的 RMS。脚本执行模式要求一致性 ≤5 mm、≤2°；这是此任务的筛查阈值，不是绝对抓取精度保证，还需独立姿态验证。定位和规划模式可用于排查。当前未实际连接机械臂或执行抓取。

## 实物布置和参数

假设方块顶面中心贴一个 marker，边缘与方块边缘平行；盘底中心贴另一个不同 ID 的 marker。两者均水平、marker 正面朝上，相机安装和焦距与标定时一致。盘子按矩形内外边界建模；圆盘可填写保守内接矩形作为内边界、外接矩形作为外边界。

方块/盘子上的单个 marker 边长必须重新测量，不能直接使用标定板的 18 mm。字典暂设 DICT_4X4_50，须与你实际标签一致。以下字段目前为 null，程序会停止并说明缺少哪项，不会猜测：

| 配置 | 如何填写 |
|---|---|
| cube.id / plate.id | 各自标签 ID，整数且不同 |
| cube.marker_mm / plate.marker_mm | 各自 marker 黑框外侧的边长，mm |
| cube.side_mm | 方块边长，mm |
| cube.grasp_depth_mm | 从顶面向下到夹持中心的距离，例如夹中心通常在边长一半处，但请按实物选取 |
| plate.rim_height_mm | 从 marker 所在盘底平面到最高盘沿的高度 |
| plate.inner_half_size_mm | 盘内无障碍区域的半宽、半长 `[x,y]`，轴与 marker 对齐 |
| plate.outer_half_size_mm | 盘外轮廓的半宽、半长 `[x,y]` |
| plate.drop_offset_mm | 放置点相对 marker 的偏移 `[x,y,z]`，默认零。边界模型以 marker 为盘中心，故应将 marker 贴在中心 |
| T_gripper_tcp | 抓取中心 TCP 到 gripper_link 的 4×4 变换，平移单位米 |
| table_z_m | 桌面在 base_link 中的高度，米，不能仅假设为零 |
| workspace_min_m / workspace_max_m | 实测可用 TCP 区域的最小/最大 `[x,y,z]`，米；不代替 IK 可达检查 |
| gripper_open / gripper_closed | 已测试的 LeRobot 夹爪开合值，0..100，非角度；closed < open，不应压坏方块 |
| tool_radius_mm | 以 TCP 为轴，包含打开的两指和工具的最大水平包络半径 |
| tcp_lowest_extent_mm | 向下抓取时 TCP 到工具最低点的竖直距离 |

`link_radius_m` 默认 0.035 m，是简化连杆包络半径，应按实际机器和相机支架增大。`clearance_m` 为盘沿上方额外抬升余量，默认 0.05 m。`release_gap_mm` 默认 3 mm，方块底部在盘底上方少许松开，避免硬压。工作区域必须清空其他障碍。

## 环境及识别 ID

用 uv 创建环境（依赖见 `pyproject.toml`，含 `lerobot[feetech]`）：

```bash
uv sync
uv run pick_place.py --scan-markers
```

扫描模式仅打开相机，在图像上显示字典内的 marker ID，不连接机械臂、不需要尺寸。Q/ESC 退出。没有识别时先核对字典、清晰度和打印图案。

## TCP 设置

`gripper_link` 的原点通常不是两指抓取中心。TCP 定义：+Z 沿向下插入/接近方块的方向，+X 沿两指闭合的方向，+Y 满足右手系。

可以依据 CAD/实测填写完整 `T_gripper_tcp`。也可通过示教辅助求取：

```bash
python pick_place.py --teach-tcp
```

工具不自动释放扭矩，不自动移动。使用已确认可手动示教的机械臂状态，先托住机械臂。将选定抓取中心持续对准同一个固定物理点，改变腕部的旋转方向；每次静止后 ENTER 记录，至少 6 个非退化姿态，建议 10–15 个，`s` 求解。输出 `tcp_translation.json` 给出 gripper_link 坐标系中的 TCP 平移。无法准确让抓取中心始终停在同一点时，应使用刚性指针/夹具或 CAD 测量，不应把随意移动的数据当作有效示教。

同一工具输入 `o` 可辅助设置旋转：将工具接近方向垂直朝向 **base_link 的 -Z**，夹爪闭合轴沿 **base_link 的 +X**，静止后确认，输出 `tcp_rotation.json`。这是基于人工轴对齐的示教，精度取决于对齐；桌面方向不一定等于 base_link 轴方向。

组合方式（R 为 tcp_rotation.json 中的旋转，t 为 tcp_translation.json 中的平移）：

```text
T_gripper_tcp = [[R00,R01,R02,tx],
                 [R10,R11,R12,ty],
                 [R20,R21,R22,tz],
                 [0,  0,  0,  1 ]]
```

将结果填写到配置，不自动替你覆盖。抓取中心的定义必须与 grasp_depth_mm 一致。

## 定位、规划、执行

1. 填好 marker 字典、ID 和实测边长，运行仅定位：

```bash
python pick_place.py --locate-only
```

手动把腕部相机放在能同时看到两个 marker 的观察位；静止后按空格。程序刷新缓存，检查多帧关节变化与目标稳定性，保存 observation.json/png。坐标换算为 `T_base_marker = FK(q) @ T_gripper_camera @ T_camera_marker`。单个平面 marker 有双解歧义，程序遇到近似等价但朝向差异大的解会拒绝，需改变视角。

2. 填完全部配置，运行默认规划模式：

```bash
python pick_place.py
```

观察位 TCP 必须已高于方块和盘沿加抬升余量，且工具 +Z 已竖直朝下（偏差 ≤5°），避免在接近桌面时大幅翻转工具。输出 `pick_place_output/plan.json`，不发送运动指令。阶段：align → approach → descend → grasp → lift → transfer → place → release → retract。SO101 是 5 个运动关节，不能随意达到任意 6D 位姿；程序尝试方块四个等价方向，用 URDF 限位求数值 IK，要求位置误差 ≤2 mm、姿态误差 ≤5°，失败就调整观察位/目标布局。程序采用局部连续 IK，并非全局完备路径规划器，规划失败不证明所有路径都不存在。

3. 标定通过筛查、实际路径检查完毕后运行执行模式：

```bash
python pick_place.py --execute
```

此模式重新观察并规划，打印阶段位置，请你在本机输入 `execute` 检查场景和路径。之后再按一次空格重新观察，检查目标是否移动；如果目标变了则退出，重新运行。控制器初始化可能临时释放扭矩，初始化前托住机械臂。不会运行之前保存的过期 plan.json。

采用低速平滑关节插值，检测反馈跟踪误差、超时、相机掉线；抬升后需要你观察并输入 `yes` 确认方块夹住，放置结束后再记录人工确认的成功/未确认。若抓空或场景改变，不要继续执行，重新观察规划。结果保存在 execution.json，不能把“命令执行完”当作抓取成功。

相机窗口 Q/ESC 或终端 Ctrl-C 停止发后续轨迹，并尝试以当前角度保持；软件停止不等于硬件急停。退出不会主动释放可能负载的机械臂。没有当前/力反馈闭环，夹爪闭合时允许物体阻挡达到命令位置，因此必须人工检查夹取结果。

## 碰撞模型范围与测试

程序检查 URDF 关节限位、TCP 工作空间、部分连杆简化包络与桌面的距离、TCP/被夹方块包络与盘沿及盘底的距离。所有路径点及执行插值均检查。这不是完整 mesh、自碰撞或环境碰撞检测；未建模底座附近结构、线缆、其他物体和支架，尤其需要检查抬升和转移路径。只适用于已人工确认无障碍的桌面布局。接近/放置方向固定竖直，盘和方块须近似水平。

运行离线验证（不会连接设备）：

```bash
python -m unittest -v test_calibration.py test_pick_place.py
```

API 依据：[OpenCV IPPE_SQUARE 的角点顺序和变换方向](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)、[LeRobot SO follower 电机接口](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py)。

## 只抓取并抬起（不使用盘子）

```bash
python pick_place.py --pick-only --locate-only
python pick_place.py --pick-only
python pick_place.py --pick-only --execute
```

`--pick-only` 完全跳过盘子参数、盘子检测及放置路径。cube.id 留空时，第一帧必须恰好检测到一个 marker，自动选择并锁定其 ID；若有多个 marker，请遮住其他标签或设置 cube.id，程序不会任意选择。采样仍按空格开始；坐标尺寸使用配置中的 cube.marker_mm。

序列为 align → approach → descend → grasp → lift。按用户要求配置中 gripper_open=100（标定范围最大张开）、gripper_closed=0（闭合终点）。闭合以小步递减；连续 8 次读数波动 ≤0.5、实际开口高于命令 ≥3、且距闭合终点 >5 时，仅判为可能接触，保持当前小闭合误差，不再继续挤压到零。空夹到底或超时不会抬起。交互模式在抬起前后要求目视确认；自动模式使用下文的接触保持和实测抬升判定。结束后保持扭矩和夹持，不松开或自动返回。

当前实物尺寸按用户最新说明保留：方块 250 mm，marker 180 mm；盘子另记录 reported_size_mm=[1000,1000]，未当作已知内外边界。请确认机械夹爪有足够张口；缩小配置尺寸不会改变实物是否可夹。

此模式仍需要 table_z_m、工作空间、工具包络、grasp_depth_mm 等实物参数，及通过检查的手眼标定；不会为缺失参数编造数值。模型名义 TCP 已由 gripper_frame_joint 写入配置，尚未验证实物偏差。

## 有限重试与故障恢复

`pick_place_config.json` 的 `retry` 控制重试；`--execute` 中启用动作恢复，定位/规划模式只重新采样，不转动电机。

| 情况 | 判定 | 恢复行为（默认配置） |
|---|---|---|
| cube / plate 未识别 | 所需目标缺失、PnP 无有效解，或观察批次不稳定 | 高位观察最多 3 批：原位、相对起始 wrist_roll +12°、-12°。每批先丢弃 5 帧，再要求 5 帧有效检测。扫描完整路径先过关节/桌面/工作空间检查，速度不超过 5°/s；姿态过低或路径不可行则停止。执行中 approach 搜索成功后恢复原腕角。 |
| 下抓前位置不稳定 | 5 帧位置偏离中位数 >5 mm、姿态相对末帧 >5°、关节变化 >0.3°；或 hover 观测相对 approach 偏移 >5 mm | 低位先原地最多采样 3 批，仍失败则竖直回撤到 approach 高度，重新观察 cube 并重新规划。默认允许一次重新规划，再失败回撤并停止，不盲目下降。 |
| 明确空抓 | 闭合时开度到达 `gripper_closed + 2`，此前未检测到稳定接触 | 保持闭合竖直回撤，高位张开，重新定位 cube、重算路径，再抓。总共最多 3 次抓取；最后失败回撤停止。20 秒无稳定接触、闭合时手臂偏移 >3°、人工否认抓住属于不确定/异常状态，直接停止，不自动松爪重抓。 |
| 抓起后移动失败 | 轨迹跟踪误差 >8°，或原有控制器 3 秒内未达到收敛条件；相机掉线/取消也终止 | 尝试保持当前关节位置及抓取时的夹持命令，不继续转移、不松爪、不自动重抓。记录 `holding_possible_object` 和 `hold_and_inspect_no_automatic_restart`。需要检查实物状态后再决定恢复。 |

原有运动控制器的收敛规则保持不变：优先误差 <1°；有限补偿后可接受 <4°。重试不会消除固定的标定/TCP 偏差。

完整 pick/place 和 pick-only 均使用接触判定，抬升/转移/下降持续保持检测接触时的夹爪命令；只在 release 阶段松爪。稳定接触仍不是“确实夹住 cube”的证明，自动模式通过抬升后反馈检查的只抓取结果记为 `lift_feedback_verified_holding`，并输出 `cube_picked_up=True`；这属于反馈推断，没有视觉持物验证。盘子须固定；重新抓取时保留原盘子坐标，cube 修正不会再平移放置点。

日志：各观察子目录的 `observation_retries.json` 记录失败原因；`execution.json` 记录 `wrist_search`、`pregrasp_retry`、`empty_grasp_retry`、`retry_retreat` 和 `motion_failure`。`retry_plan.json` 保存最近一次重新规划结果。Q/ESC、相机故障、关节通信故障不会触发继续扫描或重抓。

本次没有补造盘子尺寸或覆盖远端 TCP/偏移设置；完整放置仍需先填写当前配置中的盘子 ID、marker 尺寸及内外边界等必填值。

离线故障注入测试（模拟电机和相机）：

```bash
python -m unittest -v test_retries test_pick_place test_calibration
```

## 重启保护：先确认夹爪状态

所有 `--execute`、`--teach-tcp`、`--teach-grasp-view` 操作都在连接机器人前读取状态。**首次运行（无文件）及保存状态为 `empty` 时，默认空夹爪直接启动，不询问。** 普通抓放随后检测 cube 和 tray，`--pick-only` 只检测 cube。已有 `holding` 或 `unknown` 记录时，交互模式要求实物确认，`--auto` 停止；可用 `--startup-state` 显式说明检查后的实物状态。不要把 `--startup-state empty` 固定在无人检查的循环/启动脚本中。

- `empty`：允许正常抓取/示教；这是你刚目视检查后的声明，可以覆盖旧日志中的持物状态。
- `holding`：普通抓取/示教被阻止。仅允许下面的 `--recover-place` 恢复放置入口。
- `unknown` 或自动模式遇到未处理的持物/不确定记录：停止，不连接机器人、不发动作命令。

正常抓取（每次运行前必须确认确实空夹爪）：

```bash
python pick_place.py --execute --pick-only --auto
```

持物恢复放置（不加 `--pick-only`，需完整盘子几何配置）：

```bash
python pick_place.py --execute --recover-place --startup-state holding
# 已检查夹持和路径、需要非交互运行时，再添加 --auto。
```

恢复要求：本机持物记录存在、当前再次确认持物、夹持命令有效、开度与保存值相差不超过 5 个 LeRobot 开度单位且未接近闭合终点、所有关节/夹爪扭矩已开启，以及抓取几何/URDF/手眼标定未改变。恢复入口遇到缺失/损坏记录、闭合中断或释放中断时拒绝自动恢复，不能凭开度猜测一个夹持命令。记录不符合条件时不提供强制自动恢复；先由操作者确认并处理实物状态。

恢复会保持现有扭矩，不调用 `robot.configure()`，不先张爪，不转腕搜索。只重新观察盘子 tag，保存并检查 `recovery_plan.json`；复核盘子位姿后，从**当前实测关节位置重新规划**：竖直抬高 → 移到盘子上方 → 下降 → 松爪 → 回撤。盘子不可见、位姿变化 >5 mm / 5°、路径不可行时停止。该流程沿用现有简化碰撞模型，仍需要人工确认环境和所持 cube 状态。

持久状态在 `~/.local/state/hands-on-robo/<robot_id哈希>.json`，独立于仓库和 `--output`，包含空/持物/不确定、夹持命令、开度和标定指纹。更新使用原子替换及 fsync；同一 robot_id 加进程锁，避免两个本程序实例同时发动作。锁不约束其他控制软件，或以不同 robot_id 连接同一实体机器人的程序。

闭合前先记为不确定，检测到接触后记为持物，释放前先记为不确定。自动执行完松爪仍记为不确定；人工确认放置完成才记为空。首次运行及旧版本没有记录时按约定默认空；已有持物/不确定记录仍须处理。默认空是初始条件假设，并非传感器识别；删除状态文件会丢失保护依据。状态记录不能自动识别滑落、断电或物体被人取走。

测试：`python -m unittest -v test_startup_state test_retries test_pick_place test_calibration`。


## 抓起结果输出

`cube_picked_up` 是本次运行中“抓起检查已通过”的布尔标志：初始为 `False`，终端和 `execution.json` 都有输出。它是已完成步骤的记录，释放后不重置，不表示始终还持物，也不是 `task_complete`。

闭合接触时记录实际开度和实测关节 FK 对应的 TCP 高度。lift 到位后保持夹持，连续读取 8 次反馈（采样间隔至少 0.05 s）：开度波动 ≤0.5、平均开度距闭合终点 >5、平均开度比夹持命令大至少 1（初始接触要求 3，抬升后允许一定压缩）、相对接触时的开度变化 ≤2、关节波动 ≤0.5°，且所有样本对应的 TCP 均比接触时高至少 10 mm，才输出 `cube_picked_up=True`。开度数值均为 LeRobot 0–100 单位，不是毫米。

`pickup_verification.min_lift_mm` 默认 10，`max_opening_change` 默认 2。结果包含方法、抬升量、开度变化和失败原因；检查不通过则保持并停止，不继续转移。反馈只能推断“夹持着物体完成了抬升”，不能核实恰好为 25 mm cube，也无法排除夹爪机械卡住；尚无视觉身份验证或持续掉落检测。

新的持物记录保存接触基准，恢复时可重新检查。旧持物记录若缺少接触高度基准，则 `cube_picked_up` 保持 `False`（未验证），仍允许已有的人工确认持物后受控放置流程，不凭空补造证据。

新增离线测试：`python -m unittest -v test_pickup`。
