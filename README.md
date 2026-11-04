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
uv run python pick_place.py --pick-only --execute --auto
```

方块顶面贴 18 mm ArUco 码（DICT_4X4_50）。`cube.id` 留空时，画面里应只有一个候选标签；多个标签可见时指定 ID。

首次启动默认空夹爪，正常流程检测 cube 和 tray；`--pick-only` 只检测 cube。完整抓放需补齐盘子配置后去掉 `--pick-only`。

`detector` 改成 `"color"` 可点击选择方块颜色，目前仅支持只抓取模式。Q/ESC 或 Ctrl-C 停止，并尝试保持当前姿态和夹持。

## 当前实现的 retry 策略

| 情况 | 判定 | 处理方式 |
|---|---|---|
| 没识别到 cube / tray tag | 目标缺失或位姿无效 | 高位依次在原位、wrist +12°、−12°重新检测，最多 3 批；路径检查失败或仍识别不到就停。 |
| 下抓前定位不稳定 | 5 帧位置偏差 >5 mm、姿态变化 >5°、关节变化 >0.3°，或 cube 相比 approach 移动 >5 mm | 先原地重新采样，再竖直回撤、重新定位和规划；默认允许一次重新规划，再失败就停。 |
| 明确空抓 | 未检测到稳定接触，夹爪已接近闭合终点 | 闭合状态回撤 → 高位张开 → 重新识别和规划 → 再抓；总共最多 3 次抓取。 |
| 闭合状态不明 | 20 秒无稳定接触、手臂偏移 >3°，或人工否认抓住 | 停止，不自动松爪重抓。 |
| 抓起后移动失败 | 跟踪误差 >8°或运动未收敛 | 保持当前姿态和夹持命令，不松爪、不从头抓。 |
| 持物或异常后重启 | 保存状态为 `holding` / `unknown` | 阻止默认空夹爪流程；检查实物后才能恢复。有效持物记录可通过恢复入口重新观察 tray、规划并放下。 |

确认持物后恢复放置（需有效持物记录及完整盘子配置）：

```bash
uv run python pick_place.py --execute --recover-place --startup-state holding
```

重试参数在 `pick_place_config.json` 的 `retry` 中。接触判定不等于确认抓住，当前尚无自动持物/掉落/放置结果视觉验证；重抓时盘子应保持固定。

详细参数、日志和测试说明见 [PICK_PLACE.md](PICK_PLACE.md)。
