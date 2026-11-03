#!/usr/bin/env python3
"""SO101 eye-in-hand: T_base_gripper @ T_gripper_camera @ T_camera_board = constant."""
import argparse
import dataclasses
import hashlib
import itertools
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np
from camera_intrinsic import make_board, detect, open_camera, save_json, reprojection


def transform(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.asarray(t).reshape(3)
    return T


class URDFFK:
    def __init__(self, path, base='base_link', tip='gripper_link', offsets=None):
        """offsets: LeRobot-degrees -> URDF-degrees per joint (q_urdf = q_lerobot + offset)."""
        self.offsets = offsets or {}
        root = ET.parse(path).getroot()
        by_child = {j.find('child').get('link'): j for j in root.findall('joint')}
        chain = []
        while tip != base:
            if tip not in by_child or len(chain) > len(by_child):
                raise ValueError('No valid URDF chain')
            j = by_child[tip]
            chain.append(j)
            tip = j.find('parent').get('link')
        self.chain = chain[::-1]
        self.names = [j.get('name') for j in self.chain if j.get('type') != 'fixed']

    def __call__(self, degrees):
        return self.frames(degrees)[-1]

    def frames(self, degrees):
        """4x4 transform of every joint frame along the chain (last one is the tip)."""
        T = np.eye(4)
        out = []
        for j in self.chain:
            origin = j.find('origin')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            Rx, Ry, Rz = [cv2.Rodrigues(np.eye(3)[i] * rpy[i])[0] for i in range(3)]
            T = T @ transform(Rz @ Ry @ Rx, xyz)
            if j.get('type') != 'fixed':
                if j.get('type') not in ('revolute', 'continuous'):
                    raise ValueError('Only fixed/revolute/continuous joints supported')
                axis = np.fromstring(j.find('axis').get('xyz'), sep=' ')
                angle = np.deg2rad(degrees[j.get('name')] + self.offsets.get(j.get('name'), 0))
                T = T @ transform(cv2.Rodrigues(axis / np.linalg.norm(axis) * angle)[0], np.zeros(3))
            out.append(T)
        return out


def solve(samples):
    if len(samples) < 10:
        raise ValueError('Need at least 10 samples; recommended 15–25')
    G = [np.asarray(s['T_base_gripper'], dtype=float) for s in samples]
    C = [np.asarray(s['T_camera_board'], dtype=float) for s in samples]
    for T in G + C:
        if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]) or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5) or np.linalg.det(T[:3, :3]) < .99:
            raise ValueError('Invalid sample transform')
    rv = np.array([cv2.Rodrigues(a[:3, :3].T @ b[:3, :3])[0].ravel() for a, b in itertools.combinations(G, 2)])
    singular = np.linalg.svd(rv, compute_uv=False)
    if singular[1] < .15 or singular[1] / max(singular[0], 1e-12) < .05:
        raise ValueError('Insufficient rotation about different axes; collect more diverse poses')
    R, t = cv2.calibrateHandEye([x[:3, :3] for x in G], [x[:3, 3] for x in G],
                              [x[:3, :3] for x in C], [x[:3, 3] for x in C], method=cv2.CALIB_HAND_EYE_PARK)
    X = transform(R, t)
    if not np.isfinite(X).all() or not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or np.linalg.det(R) < .99:
        raise ValueError('Degenerate hand-eye solution')
    boards = np.array([g @ X @ c for g, c in zip(G, C)])
    center = np.mean(boards[:, :3, 3], axis=0)
    U, _, V = np.linalg.svd(np.sum(boards[:, :3, :3], axis=0))
    mean_R = U @ np.diag([1, 1, np.linalg.det(U @ V)]) @ V
    terr = np.linalg.norm(boards[:, :3, 3] - center, axis=1) * 1000
    rerr = [float(np.rad2deg(np.linalg.norm(cv2.Rodrigues(mean_R.T @ b[:3, :3])[0]))) for b in boards]
    return {'T_gripper_camera': X.tolist(), 'T_camera_gripper': np.linalg.inv(X).tolist(),
            'translation_unit': 'meter', 'convention': 'p_gripper = T_gripper_camera @ p_camera',
            'method': 'PARK', 'sample_count': len(samples), 'rotation_excitation_singular_values': singular.tolist(),
            'board_position_error_mm': terr.tolist(), 'board_rotation_error_deg': rerr,
            'board_position_rms_mm': float(np.sqrt(np.mean(terr ** 2))),
            'board_rotation_rms_deg': float(np.sqrt(np.mean(np.square(rerr))))}


def fit_joint_offsets(samples, fk):
    """Refit hand-eye X, the fixed board pose B and per-joint zero offsets so that B = FK(q+off) @ X @ C.
    LeRobot's 0 deg is the midpoint of the manually recorded range, not the URDF zero, so FK(q) is biased.
    First and last joint offsets are unobservable (absorbed into B / X) and stay 0.
    """
    from scipy.optimize import least_squares
    names = fk.names[1:-1]
    C = [np.asarray(s['T_camera_board'], dtype=float) for s in samples]
    Q = [s['joints_deg'] for s in samples]
    X0 = np.array(solve(samples)['T_gripper_camera'])
    B0 = np.array(samples[0]['T_base_gripper']) @ X0 @ C[0]

    def unpack(x):
        return (transform(cv2.Rodrigues(x[:3])[0], x[3:6]), transform(cv2.Rodrigues(x[6:9])[0], x[9:12]),
                {**fk.offsets, **dict(zip(names, x[12:]))})

    def residual(x):
        X, B, offsets = unpack(x)
        fk.offsets = offsets
        out = []
        for q, c in zip(Q, C):
            E = np.linalg.inv(B) @ fk(q) @ X @ c
            out.append(np.r_[E[:3, 3] * 100, cv2.Rodrigues(E[:3, :3])[0].ravel()])
        return np.concatenate(out)

    x0 = np.r_[cv2.Rodrigues(X0[:3, :3])[0].ravel(), X0[:3, 3], cv2.Rodrigues(B0[:3, :3])[0].ravel(), B0[:3, 3], np.zeros(len(names))]
    lo, hi = np.full(len(x0), -np.inf), np.full(len(x0), np.inf)
    lo[12:], hi[12:] = -45, 45  # ponytail: larger offsets would mean a sign flip or wrong URDF, not a zero error
    saved = fk.offsets
    try:
        fit = least_squares(residual, x0, bounds=(lo, hi), xtol=1e-10, ftol=1e-10)
    finally:
        fk.offsets = saved
    _, _, offsets = unpack(fit.x)
    fk.offsets = offsets
    corrected = [dict(s, T_base_gripper=fk(q).tolist()) for s, q in zip(samples, Q)]
    fk.offsets = saved
    result = solve(corrected)
    result['joint_offsets_deg'] = {n: float(offsets.get(n, 0)) for n in fk.names}
    result['joint_offsets_note'] = 'q_urdf = q_lerobot + offset; first/last joint fixed at 0 (unobservable)'
    return result


def robot_reader(a):
    try:
        from lerobot.robots.so_follower.so_follower import SO101Follower
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    except ModuleNotFoundError:
        from lerobot.robots.so101_follower.so101_follower import SO101Follower
        from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
    kwargs = dict(port=a.port, id=a.robot_id, use_degrees=True)
    if a.calibration_dir:
        kwargs['calibration_dir'] = a.calibration_dir
    robot = SO101Follower(SO101FollowerConfig(**kwargs))
    if not robot.calibration:
        raise ValueError('No existing motor calibration. Use the robot ID/calibration directory from LeRobot motor calibration.')
    # Connect only the bus: avoid robot.connect() configuring PID/operating modes.
    robot.bus.connect()
    try:
        if not robot.is_calibrated:
            raise ValueError('Motor calibration does not match hardware; run LeRobot calibration first')
        if a.disable_torque:
            robot.bus.disable_torque()
    except Exception:
        robot.bus.disconnect(disable_torque=False)
        raise
    return robot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--camera-yml', type=Path, default=Path('intrinsic_output/camera.yml'))
    p.add_argument('--camera', default='0')
    p.add_argument('--port', default='/dev/tty.usbmodem5B421342761')
    p.add_argument('--robot-id', help='Must match the ID used for LeRobot motor calibration')
    p.add_argument('--calibration-dir', type=Path)
    p.add_argument('--urdf', type=Path, default=Path(__file__).with_name('so101_new_calib.urdf'))
    p.add_argument('--output', type=Path, default=Path('handeye_output'))
    p.add_argument('--solve-only', type=Path, help='Previously saved samples.json; no hardware connection')
    p.add_argument('--disable-torque', action='store_true', help='Release motors for supported manual movement')
    p.add_argument('--max-motion-deg', type=float, default=.3)
    p.add_argument('--max-pnp-rms', type=float, default=1.0)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if a.solve_only:
        data = json.loads(a.solve_only.read_text())
    else:
        if not a.robot_id:
            p.error('--robot-id is required for capture')
        if (a.output / 'samples.json').exists():
            raise ValueError('Samples already exist: choose a new output or use --solve-only')
        fs = cv2.FileStorage(str(a.camera_yml), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise ValueError('Cannot open camera.yml; calibrate intrinsics first')
        K, D = fs.getNode('K').mat(), fs.getNode('D').mat()
        size = (int(fs.getNode('image_width').real()), int(fs.getNode('image_height').real()))
        config = json.loads(fs.getNode('board_json').string())
        fs.release()
        board = make_board(argparse.Namespace(**config))
        params = cv2.aruco.CharucoParameters()
        params.cameraMatrix, params.distCoeffs = K, D
        detector = cv2.aruco.CharucoDetector(board, params)
        fk = URDFFK(a.urdf)
        data = {'board': config, 'base_frame': 'base_link', 'gripper_frame': 'gripper_link',
                'robot_id': a.robot_id, 'urdf': a.urdf.read_text(), 'K': K.tolist(), 'D': D.tolist(), 'samples': []}
        robot, cap = None, None
        try:
            cap = open_camera(a.camera, *size)
            robot = robot_reader(a)
            data['motor_calibration'] = {name: dataclasses.asdict(value) for name, value in robot.calibration.items()}
            data['motor_calibration_path'] = str(robot.calibration_fpath)
            data['urdf_sha256'] = hashlib.sha256(a.urdf.read_bytes()).hexdigest()
            data['joint_mapping'] = 'LeRobot use_degrees=True; direct degrees-to-radians; no added offsets'
            data['image_size'] = list(size)
            save_json(a.output / 'session_metadata.json', {k: v for k, v in data.items() if k != 'samples'})
            def read():
                values = robot.bus.sync_read('Present_Position')
                return {n: float(values[n]) for n in fk.names}
            print('Keep board/base fixed. SPACE: capture stationary pose; C: solve; Q: quit. Support arm if torque disabled.')
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Camera read failed')
                if (frame.shape[1], frame.shape[0]) != size:
                    raise ValueError('Resolution differs from camera.yml')
                view = frame.copy()
                d = detect(frame, board, detector)
                if d:
                    cv2.aruco.drawDetectedCornersCharuco(view, d[1], d[2])
                cv2.putText(view, f'samples={len(data["samples"])} SPACE save | C solve | Q quit', (15, 30), 0, .65, (0, 255, 0), 2)
                cv2.imshow('Hand-eye', view)
                key = cv2.waitKey(1) & 255
                if key in (27, ord('q')):
                    return
                if key == ord('c'):
                    break
                if key != 32:
                    continue
                before = read()
                start = time.time()
                # Drain buffered images while the user holds the same pose.
                readings = [before]
                for _ in range(5):
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError('Camera read failed')
                    readings.append(read())
                delta = np.ptp([[q[n] for n in fk.names] for q in readings], axis=0)
                if delta.max() > a.max_motion_deg:
                    print('Rejected: robot moved during capture', delta)
                    continue
                d = detect(frame, board, detector)
                if not d:
                    print('Rejected: insufficient corners')
                    continue
                obj, img, ids = d
                success, r, t = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
                if not success:
                    continue
                error = reprojection(obj, img, r, t, K, D)
                R = cv2.Rodrigues(r)[0]
                if error > a.max_pnp_rms or np.min((obj.reshape(-1, 3) @ R.T + t.reshape(3))[:, 2]) <= 0:
                    print(f'Rejected: invalid pose / reprojection RMS={error:.3f}px')
                    continue
                q = {n: float(np.mean([v[n] for v in readings])) for n in fk.names}
                G = fk(q)
                if any(np.linalg.norm(G[:3, 3] - np.array(s['T_base_gripper'])[:3, 3]) < .005 and
                       np.linalg.norm(cv2.Rodrigues(G[:3, :3].T @ np.array(s['T_base_gripper'])[:3, :3])[0]) < np.deg2rad(5)
                       for s in data['samples']):
                    print('Rejected: pose too similar to an existing sample')
                    continue
                name = f'{len(data["samples"]):04d}.png'
                if not cv2.imwrite(str(a.output / name), frame):
                    raise IOError('Image save failed')
                data['samples'].append({'image': name, 'timestamp': start, 'capture_duration_s': time.time() - start,
                    'joints_deg': q, 'joint_readings_deg': readings, 'ids': ids.ravel().tolist(),
                    'corners': img.reshape(-1, 2).tolist(), 'pnp_rms_px': error,
                    'T_base_gripper': G.tolist(), 'T_camera_board': transform(R, t).tolist()})
                save_json(a.output / 'samples.json', data)
                print(f'Saved {len(data["samples"])}; PnP RMS={error:.3f}px; joints(deg)={q}', flush=True)
                coverage = {n: round(max(s['joints_deg'][n] for s in data['samples']) - min(s['joints_deg'][n] for s in data['samples']), 1) for n in fk.names}
                print('Joint coverage (deg):', coverage, flush=True)
        finally:
            if cap is not None:
                cap.release()
            if robot is not None:
                robot.bus.disconnect(disable_torque=False)
            cv2.destroyAllWindows()
    raw = solve(data['samples'])
    result = fit_joint_offsets(data['samples'], URDFFK(a.urdf))
    result['before_offset_fit'] = {k: raw[k] for k in ('board_position_rms_mm', 'board_rotation_rms_deg')}
    result.update(base_frame='base_link', gripper_frame='gripper_link')
    save_json(a.output / 'handeye.json', result)
    fs = cv2.FileStorage(str(a.output / 'handeye.yml'), cv2.FILE_STORAGE_WRITE)
    fs.write('T_gripper_camera', np.array(result['T_gripper_camera']))
    fs.write('T_camera_gripper', np.array(result['T_camera_gripper']))
    fs.write('translation_unit', 'meter')
    fs.release()
    print(json.dumps(result, indent=2))
    print('Residuals measure training-set consistency; validate on new poses before using the transform.')


if __name__ == '__main__':
    main()
