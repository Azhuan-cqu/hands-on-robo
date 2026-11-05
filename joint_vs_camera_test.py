#!/usr/bin/env python3
"""Decisive check: torque ON, servo moves ONE joint by +/-delta, camera measures the board before/after.
Encoder delta == camera rotation  -> joint and camera fine (hand-moved captures were the problem).
Camera rotates consistently less  -> camera side (intrinsics / mount).
Out and back disagree             -> mechanical play in that joint.
"""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
from camera_intrinsic import make_board, detect, load_camera, open_camera
from handeye_calibration import robot_reader

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--joint', default='elbow_flex')
p.add_argument('--delta', type=float, default=6.0, help='degrees; sign chooses direction')
p.add_argument('--camera', default='0')
p.add_argument('--camera-yml', type=Path, default=Path('intrinsic_output/camera.yml'))
p.add_argument('--port', default='/dev/tty.usbmodem5B421342761')
p.add_argument('--robot-id', default='my_awesome_follower_arm')
p.add_argument('--calibration-dir', type=Path)
a = p.parse_args()

K, D, size, board_cfg = load_camera(a.camera_yml)
board = make_board(argparse.Namespace(**board_cfg))
params = cv2.aruco.CharucoParameters()
params.cameraMatrix, params.distCoeffs = K, D
detector = cv2.aruco.CharucoDetector(board, params)


def board_pose(cap):
    """Median over several frames; returns (R, t) of board in camera, or raises."""
    Rs, ts = [], []
    for _ in range(8):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError('Camera read failed')
        d = detect(frame, board, detector)
        if d is None:
            continue
        ok, r, t = cv2.solvePnP(d[0], d[1], K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        Rs.append(r.ravel()); ts.append(t.ravel())
    if len(Rs) < 4:
        raise RuntimeError('Board not reliably visible')
    r, t = np.median(Rs, axis=0), np.median(ts, axis=0)
    return cv2.Rodrigues(r)[0], t


def move(robot, name, start, end):
    steps = max(10, int(abs(end - start) / 5 * 50))  # ~5 deg/s at 50 Hz
    for i in range(1, steps + 1):
        u = i / steps
        robot.bus.write('Goal_Position', name, start + (end - start) * (3 * u * u - 2 * u ** 3))
        time.sleep(.02)
    time.sleep(.8)
    return float(robot.bus.read('Present_Position', name))


robot = robot_reader(argparse.Namespace(port=a.port, robot_id=a.robot_id, calibration_dir=a.calibration_dir, disable_torque=False))
cap = open_camera(a.camera, *size)
try:
    q0 = robot.bus.sync_read('Present_Position')
    if a.joint not in q0:
        raise ValueError(f'Unknown joint {a.joint}; choose from {list(q0)}')
    print('Holding current pose with torque ON. Keep hands clear; Ctrl-C aborts and leaves torque on.')
    robot.bus.sync_write('Goal_Position', q0)
    robot.bus.enable_torque()
    time.sleep(.5)
    for _ in range(10):
        cap.read()  # flush
    start = float(robot.bus.read('Present_Position', a.joint))
    print(f'{"leg":10s} {"encoder":>9s} {"camera":>9s} {"cam move":>9s}')
    rows = []
    for leg, target in (('out', start + a.delta), ('back', start)):
        before = float(robot.bus.read('Present_Position', a.joint))
        Rb, tb = board_pose(cap)
        after = move(robot, a.joint, before, target)
        Ra, ta = board_pose(cap)
        enc = after - before
        cam = np.rad2deg(np.linalg.norm(cv2.Rodrigues(Rb.T @ Ra)[0]))
        # camera displacement relative to the fixed board, mm
        mv = np.linalg.norm((-Rb.T @ tb) - (-Ra.T @ ta)) * 1000
        rows.append({'leg': leg, 'encoder_deg': enc, 'camera_deg': float(cam), 'camera_move_mm': float(mv)})
        print(f'{leg:10s} {enc:9.2f} {cam:9.2f} {mv:9.1f}')
    others = {n: abs(float(v) - float(q0[n])) for n, v in robot.bus.sync_read('Present_Position').items() if n != a.joint}
    print('other joints drifted (deg):', {n: round(v, 2) for n, v in others.items()})
    Path('joint_vs_camera_result.json').write_text(json.dumps({'joint': a.joint, 'delta': a.delta, 'legs': rows}, indent=2))
finally:
    cap.release()
    print('Torque left ON (arm holding). Support the arm before disabling torque.')
    robot.bus.disconnect(disable_torque=False)
