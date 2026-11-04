## 安装

```bash
uv sync
```

## 1. 相机内参（一次）

```bash
uv run python camera_intrinsic.py --camera 0 --width 1280 --height 720
```

## 2. 手眼标定（舵机自动走姿态，不要用手托）

把标定板固定在桌上，手动把臂摆到板子在画面正中、距离约 30 cm，然后：

```bash
uv run python handeye_calibration.py --camera 0 --robot-id my_awesome_follower_arm --output handeye_output_v5 --auto-poses 25
```

启动后手拿开。采满自动求解，结果写入 `handeye_output_v5/handeye.json`（含关节零点偏移）。
当前这批：4.9 mm / 1.4°。配置 `pick_place_config.json` 的 `handeye_json` 指向它。

## 3. 指尖位置（一次，只用相机）

```bash
uv run python pick_place.py --teach-tcp-click
```

夹爪张开、两指尖在画面里，点两指中间，S 保存，Q 退出。

## 4. 抓取

```bash
uv run python pick_place.py --pick-only --execute --ignore-calibration-gate --auto
```

方块顶面贴 18 mm ArUco 码（DICT_4X4_50），画面里只有这一个码。全自动：定位 → 规划 → 对齐 → 接近（近距离再识别修正）→ 下降 → 闭合 → 抬起。
Q 或 Ctrl-C 停止，臂保持当前位置；要用手挪臂先托住再 `uv run python release_torque.py`。

`detector` 改成 `"color"` 则不用贴码：启动后在画面里点一下要抓的方块。

## 排查

- `joint_vs_camera_test.py --joint elbow_flex --delta 6`：舵机自己转一个关节，比较编码器和相机看到的角度，判断关节/相机/采样方式哪个有问题。
- 详细参数说明见 [PICK_PLACE.md](PICK_PLACE.md)。
