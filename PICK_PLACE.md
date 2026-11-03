# SO101：把方块放入盘子

入口 `pick_place.py`，配置 `pick_place_config.json`。对应课件：ArUco + solvePnP 定位 → FK/手眼变换 → TCP 偏移 + 数值 IK + 分段轨迹 → LeRobot 执行和人工结果检查。已经填入 robot ID `my_awesome_follower_arm`、相机 `0`、串口 `/dev/tty.usbmodem5B421342761`。

**当前标定不能直接用于执行**：现有 handeye.json 固定板位置 RMS=31.68 mm、旋转 RMS=5.80°；camera.yml 的像素 RMS=0.243 px。前者是厘米级位姿不一致。先检查关节零位、URDF 对应关系、板/底座是否移动、腕部相机是否松动，再重新采样。脚本执行模式要求一致性 ≤5 mm、≤2°；这是此任务的筛查阈值，不是绝对抓取精度保证，还需独立姿态验证。定位和规划模式可用于排查。当前未实际连接机械臂或执行抓取。

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

序列为 align → approach → descend → grasp → lift。按用户要求配置中 gripper_open=100（标定范围最大张开）、gripper_closed=0（闭合终点）。闭合以小步递减；连续 8 次读数波动 ≤0.5、实际开口高于命令 ≥3、且距闭合终点 >5 时，仅判为可能接触，保持当前小闭合误差，不再继续挤压到零。空夹到底或超时不会抬起。编码器稳定不能证明夹住，抬起前需目视确认输入 yes，抬起后再次确认并写日志。结束后保持扭矩和夹持，不松开或自动返回。

当前实物尺寸按用户最新说明保留：方块 250 mm，marker 180 mm；盘子另记录 reported_size_mm=[1000,1000]，未当作已知内外边界。请确认机械夹爪有足够张口；缩小配置尺寸不会改变实物是否可夹。

此模式仍需要 table_z_m、工作空间、工具包络、grasp_depth_mm 等实物参数，及通过检查的手眼标定；不会为缺失参数编造数值。模型名义 TCP 已由 gripper_frame_joint 写入配置，尚未验证实物偏差。
