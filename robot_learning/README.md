# Robot Learning

用 LeRobot 采集示教数据、训练 ACT 策略并评估。

```bash
export HF_USER=$(huggingface-cli whoami | head -n 1)
```

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
