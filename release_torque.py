"""Disable torque on all motors so the arm can be moved by hand. Support the arm first!"""
import argparse, json
from handeye_calibration import robot_reader
c = json.load(open('pick_place_config.json'))
r = robot_reader(argparse.Namespace(port=c['port'], robot_id=c['robot_id'], calibration_dir=None, disable_torque=True))
r.bus.disconnect(disable_torque=False)
print('Torque OFF: arm is free.')
