## 安装

```bash
uv sync
```

## 1. 相机内参

```bash
uv run python camera_intrinsic.py --camera 0
```

## 2. 手眼标定

```bash
uv run python handeye_calibration.py --robot-id my_awesome_follower_arm --output handeye_output_v5 --auto-poses 25
```

## 3. 指尖位置

```bash
uv run python pick_place.py --teach-tcp-click
```

## 4. 抓取

```bash
uv run python pick_place.py --pick-only --execute --auto
```

方块颜色在 `pick_place_config.json` 的 `cube.color` 里设置。

## Robot Learning

见 [robot_learning/README.md](robot_learning/README.md)。
