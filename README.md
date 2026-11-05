## 安装

```bash
uv sync
```

## 1. 相机内参

```bash
uv run python camera_intrinsic.py --camera 0
```

## 2. 手眼标定

标定板固定在桌上，把臂摆到板子在画面正中、距离约 30 cm，手拿开：

```bash
uv run python handeye_calibration.py --robot-id my_awesome_follower_arm --output handeye_output_v5 --auto-poses 25
```

结果写入 `handeye_output_v5/handeye.json`，由配置里的 `handeye_json` 引用。

## 3. 指尖位置

```bash
uv run python pick_place.py --teach-tcp-click
```

点两指中间，S 保存，Q 退出。

## 4. 抓取

```bash
uv run python pick_place.py --pick-only --execute --auto
```

`--scan-color` 预览检测，HSV 阈值在配置的 `cube`/`plate` 里。Q 或 Ctrl-C 停止，臂保持当前位置；托住臂再运行 `release_torque.py` 松开舵机。

## 排查

`joint_vs_camera_test.py --joint elbow_flex --delta 6`：比较单个关节的编码器角度和相机测得的角度。
