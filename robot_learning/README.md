# Robot Learning

用 LeRobot 做模仿学习：遥操作采集示教数据 → 训练 ACT 策略 → 在 SO-101 上评估。任务与第一部分相同：把方块放进托盘。

先设置 Hugging Face 用户名（`huggingface-cli login` 之后）：

```bash
export HF_USER=$(huggingface-cli whoami | head -n 1)
```

`--robot.port` / `--teleop.port` 换成自己的串口（`lerobot-find-port` 查看）。

## 1. 遥操作测试

```bash
uv run lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5B421342761 --robot.id=my_awesome_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader --teleop.port=<LEADER_PORT> --teleop.id=my_awesome_leader_arm \
  --display_data=true
```

## 2. 采集数据

```bash
uv run lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5B421342761 --robot.id=my_awesome_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader --teleop.port=<LEADER_PORT> --teleop.id=my_awesome_leader_arm \
  --dataset.repo_id=${HF_USER}/so101_pick_place \
  --dataset.num_episodes=50 \
  --dataset.single_task="Pick the cube and place it in the tray" \
  --display_data=true
```

→ 下一段，← 重录当前段，Esc 结束。

## 3. 训练

```bash
uv run lerobot-train \
  --dataset.repo_id=${HF_USER}/so101_pick_place \
  --policy.type=act \
  --policy.device=mps \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/act_so101 \
  --job_name=act_so101
```

Mac 用 `mps`，有 NVIDIA GPU 用 `cuda`。

## 4. 评估

```bash
uv run lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5B421342761 --robot.id=my_awesome_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=${HF_USER}/eval_so101_pick_place \
  --dataset.num_episodes=10 \
  --dataset.single_task="Pick the cube and place it in the tray" \
  --policy.path=outputs/train/act_so101/checkpoints/last/pretrained_model
```

可用 `../placement_success.py` 判定每次是否放进托盘。
