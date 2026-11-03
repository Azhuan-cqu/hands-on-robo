## 1
python camera_intrinsic.py --camera 0 --width 1280 --height 720

## 2
python handeye_calibration.py \  --camera 0 \                
  --robot-id my_awesome_follower_arm \
  --disable-torque
## 方块抓取与放置

新增入口 `pick_place.py`、配置 `pick_place_config.json`。安装、实测参数、TCP 示教、仅定位/规划/执行步骤见 [PICK_PLACE.md](PICK_PLACE.md)。当前手眼一致性误差偏大，执行前需重新验证标定。
