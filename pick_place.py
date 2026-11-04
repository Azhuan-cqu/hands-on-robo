#!/usr/bin/env python3
"""SO101 marker-based pick/place. Default: observe and plan, no motor commands."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from camera_intrinsic import open_camera, save_json, reprojection
from handeye_calibration import URDFFK, robot_reader, transform

from grasp_state import GraspState, startup_decision

ROOT = Path(__file__).resolve().parent


def select_marker(ids, selected=None):
    values = np.asarray(ids).reshape(-1).tolist() if ids is not None else []
    if selected is None:
        if len(values) != 1:
            raise ValueError(f'Automatic target selection needs exactly one visible marker; detected {values}. Hide other markers or set cube.id.')
        return int(values[0])
    if values.count(selected) != 1:
        raise ValueError(f'Locked target ID {selected} is not uniquely visible')
    return selected


def contact_candidate(readings, command, closed):
    """Stable position with closing error: possible contact, not proof of grasp."""
    if len(readings) < 8:
        return False
    tail = np.asarray(readings[-8:], dtype=float)
    return bool(np.isfinite(tail).all() and np.ptp(tail) <= .5 and
                np.mean(tail) > closed + 5 and np.mean(tail) - command >= 3)


def matrix(value):
    T = np.asarray(value, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]) or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5) or np.linalg.det(T[:3, :3]) < .99:
        raise ValueError('Expected a rigid 4x4 transform')
    return T


def roles(c):
    if c.get('recover_place'):
        return ('plate',)
    return ('cube',) if c.get('pick_only') else ('cube', 'plate')


def check_config(c, planning=True):
    retry = retry_settings(c)
    for key in ('observation_attempts', 'stable_frames', 'grasp_attempts', 'pregrasp_attempts'):
        if type(retry[key]) is not int or not 1 <= retry[key] <= 10:
            raise ValueError(f'Invalid retry.{key}: expected integer 1..10')
    if retry['stable_frames'] < 3:
        raise ValueError('Need at least 3 stable frames')
    for key, maximum in (('position_mm', 15), ('orientation_deg', 10), ('wrist_scan_deg', 20)):
        if not np.isfinite(retry[key]) or not 0 < retry[key] <= maximum:
            raise ValueError(f'Invalid retry.{key}')
    if c.get('detector') == 'color':
        if not c.get('pick_only'):
            raise ValueError('Color detector currently supports --pick-only (no plate marker)')
        if not c['cube'].get('hsv_ranges') or c['table_z_m'] is None or c['cube']['side_mm'] is None:
            raise ValueError('Color detector needs cube.hsv_ranges, cube.side_mm and table_z_m')
        c['cube'].setdefault('id', -1); c['cube'].setdefault('marker_mm', 1)
    for role in roles(c):
        for key in ('id', 'marker_mm'):
            if c[role][key] is None and not (c.get('pick_only') and key == 'id'):
                raise ValueError(f'Fill {role}.{key} with measured data in config')
        if c[role]['marker_mm'] <= 0:
            raise ValueError('Marker length must be positive')
    if not c.get('pick_only') and c['cube']['id'] == c['plate']['id']:
        raise ValueError('Cube and plate need different marker IDs')
    if planning:
        for key in ('T_gripper_tcp', 'table_z_m', 'workspace_min_m', 'workspace_max_m', 'gripper_open', 'gripper_closed', 'tool_radius_mm', 'tcp_lowest_extent_mm'):
            if c[key] is None:
                raise ValueError(f'Fill {key} in config first')
        matrix(c['T_gripper_tcp'])
        dimensions = [('cube', 'side_mm'), ('cube', 'grasp_depth_mm')]
        if not c.get('pick_only'):
            dimensions += [('plate', 'rim_height_mm'), ('plate', 'inner_half_size_mm'), ('plate', 'outer_half_size_mm')]
        for role, key in dimensions:
            if c[role][key] is None and not (c.get('pick_only') and key == 'id'):
                raise ValueError(f'Fill {role}.{key} in config first')
        if not 0 < c['cube']['grasp_depth_mm'] < c['cube']['side_mm']:
            raise ValueError('Grasp depth must be between cube top and bottom')
        if not 0 <= c['gripper_closed'] < c['gripper_open'] <= 100:
            raise ValueError('Gripper values use LeRobot 0..100; closed < open')
        if not c.get('pick_only') and min(c['plate']['inner_half_size_mm']) <= c['cube']['side_mm'] / 2 + 10:
            raise ValueError('Plate interior too small for cube plus 10 mm margin')
        if not c.get('pick_only') and not np.all(np.asarray(c['plate']['outer_half_size_mm']) > c['plate']['inner_half_size_mm']):
            raise ValueError('Plate outer dimensions must exceed inner dimensions')
        if c['speed_deg_s'] <= 0 or c['speed_deg_s'] > 15 or c['clearance_m'] < .03 or c['link_radius_m'] <= 0:
            raise ValueError('Invalid speed/clearance/radius')
        if c['tool_radius_mm'] <= 0 or c['tcp_lowest_extent_mm'] < 0 or (not c.get('pick_only') and c['plate']['rim_height_mm'] < 0):
            raise ValueError('Invalid measured tool/plate envelope')
        if not np.all(np.asarray(c['workspace_min_m']) < c['workspace_max_m']):
            raise ValueError('Invalid workspace bounds')


def calibration_gate(h):
    # Task policy, not a universal guarantee of accuracy.
    errors = np.array([h.get('board_position_rms_mm', float('inf')), h.get('board_rotation_rms_deg', float('inf'))], dtype=float)
    if not np.isfinite(errors).all() or np.any(errors < 0) or np.any(errors > [5, 2]):
        raise ValueError('Execution blocked: hand-eye consistency exceeds 5 mm / 2 deg. Recalibrate and validate on independent poses.')


def marker_pose(corners, side_m, K, D, max_rms=1.5, up_hint=None):
    """up_hint: base 'up' direction expressed in the camera frame; resolves the planar ambiguity."""
    s = side_m / 2
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
    img = np.asarray(corners, dtype=np.float64).reshape(4, 1, 2)
    ok, rs, ts, _ = cv2.solvePnPGeneric(obj, img, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    candidates = []
    if ok:
        for r, t in zip(rs, ts):
            R = cv2.Rodrigues(r)[0]
            if np.min((obj @ R.T + t.reshape(3))[:, 2]) > 0:
                candidates.append((reprojection(obj, img, r, t, K, D), transform(R, t)))
    candidates.sort(key=lambda x: x[0])
    if not candidates or candidates[0][0] > max_rms:
        raise ValueError('Marker PnP reprojection too large / no valid solution')
    if up_hint is not None and len(candidates) > 1 and candidates[1][0] - candidates[0][0] < .15:
        # Planar ambiguity: both solutions fit; the marker lies flat, so take the one whose normal points up.
        candidates.sort(key=lambda x: -float(up_hint @ x[1][:3, :3] @ [0, 0, 1]))
    return candidates[0]


class Scene:
    def __init__(self, c, fk):
        self.c, self.fk = c, fk
        self.selected_id = c['cube']['id']
        fs = cv2.FileStorage(str(ROOT / c['camera_yml']), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise ValueError('Cannot read camera calibration')
        self.K, self.D = fs.getNode('K').mat(), fs.getNode('D').mat()
        self.size = int(fs.getNode('image_width').real()), int(fs.getNode('image_height').real())
        fs.release()
        self.h = json.loads((ROOT / c['handeye_json']).read_text())
        if self.h.get('translation_unit') != 'meter' or self.h.get('gripper_frame') != 'gripper_link':
            raise ValueError('Hand-eye units/frame mismatch')
        self.X = matrix(self.h['T_gripper_camera'])
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, c['dictionary'])), params)

    def color_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(frame.shape[:2], np.uint8)
        for lo, hi in self.c['cube']['hsv_ranges']:
            mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    def pixel_to_plane(self, T_base_cam, px, z_plane):
        """Cast the ray through pixel px and intersect with the horizontal plane z=z_plane (base frame)."""
        und = cv2.undistortPoints(np.array([[px]], np.float64), self.K, self.D).reshape(2)
        ray = T_base_cam[:3, :3] @ np.array([und[0], und[1], 1.0])
        origin = T_base_cam[:3, 3]
        if abs(ray[2]) < 1e-6 or (z_plane - origin[2]) / ray[2] <= 0:
            raise ValueError('Ray does not hit the table plane in front of the camera')
        return origin + ray * (z_plane - origin[2]) / ray[2]

    def detect_color(self, frame, q):
        """Cube by color: top-face centroid cast onto the plane z = table + side. Needs FK + hand-eye only."""
        view = frame.copy()
        mask = self.color_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [k for k in contours if cv2.contourArea(k) > self.c['cube'].get('min_area_px', 400)]
        if not contours:
            return {}, cv2.addWeighted(view, .7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), .3, 0)
        blob = max(contours, key=cv2.contourArea)
        (cx, cy), (w, h), angle = cv2.minAreaRect(blob)
        cv2.drawContours(view, [cv2.boxPoints(((cx, cy), (w, h), angle)).astype(int)], 0, (0, 255, 0), 2)
        T_base_cam = self.fk(q) @ self.X
        top = self.c['table_z_m'] + self.c['cube']['side_mm'] / 1000
        centre = self.pixel_to_plane(T_base_cam, (cx, cy), top)
        # yaw: direction of one rectangle edge, cast onto the same plane
        a = np.deg2rad(angle)
        edge = self.pixel_to_plane(T_base_cam, (cx + 20 * np.cos(a), cy + 20 * np.sin(a)), top) - centre
        x = np.array([edge[0], edge[1], 0]); x /= np.linalg.norm(x) + 1e-12
        R = np.column_stack([x, np.cross([0, 0, 1], x), [0, 0, 1]])
        B = transform(R, centre)
        C = np.linalg.inv(T_base_cam) @ B
        cv2.drawFrameAxes(view, self.K, self.D, cv2.Rodrigues(C[:3, :3])[0], C[:3, 3], .025)
        return {'cube': {'T_base_marker': B.tolist(), 'T_camera_marker': C.tolist(), 'id': -1, 'rms_px': 0.0,
                         'pixel': [float(cx), float(cy)], 'area_px': float(cv2.contourArea(blob))}}, view

    def detect(self, frame, q):
        if (frame.shape[1], frame.shape[0]) != self.size:
            raise ValueError('Camera resolution differs from calibration')
        if self.c.get('detector') == 'color':
            return self.detect_color(frame, q)
        corners, ids, _ = self.detector.detectMarkers(frame)
        view = frame.copy()
        found = {}
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(view, corners, ids)
            for role in roles(self.c):
                target_id = self.c[role]['id']
                if self.c.get('pick_only'):
                    self.selected_id = select_marker(ids[ids != self.c.get('plate', {}).get('id')], self.selected_id)
                    target_id = self.selected_id
                indices = np.flatnonzero(ids.ravel() == target_id)
                if len(indices) != 1:
                    continue
                T_base_cam = self.fk(q) @ self.X
                error, C = marker_pose(corners[indices[0]], self.c[role]['marker_mm'] / 1000, self.K, self.D, up_hint=T_base_cam[:3, :3].T @ [0, 0, 1])
                B = T_base_cam @ C
                found[role] = {'T_base_marker': B.tolist(), 'T_camera_marker': C.tolist(), 'id': int(target_id), 'rms_px': error,
                               'pixel': np.asarray(corners[indices[0]]).reshape(4, 2).mean(axis=0).tolist()}
                r = cv2.Rodrigues(C[:3, :3])[0]
                cv2.drawFrameAxes(view, self.K, self.D, r, C[:3, 3], .025)
        return found, view


def hsv_ranges_around(h, sat, val):
    lo, hi = [max(0, int(h)-10), max(0, int(sat)-70), max(0, int(val)-80)], [min(180, int(h)+10), 255, 255]
    ranges = [[lo, hi]]
    if h < 10: ranges.append([[170 + int(h), lo[1], lo[2]], [180, 255, 255]])   # red wraps around
    if h > 170: ranges.append([[0, lo[1], lo[2]], [int(h) - 170, 255, 255]])
    return ranges


def pick_color_by_click(cap, c):
    """Show the camera; the operator clicks the cube to grasp. Its colour becomes cube.hsv_ranges for this run."""
    state = {'hsv': None, 'done': False}
    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN and state['hsv'] is not None:
            patch = state['hsv'][max(0, y-4):y+5, max(0, x-4):x+5].reshape(-1, 3)
            h, sat, val = np.median(patch, axis=0)
            c['cube']['hsv_ranges'] = hsv_ranges_around(h, sat, val); state['done'] = True
            print(f'Target colour HSV=({int(h)},{int(sat)},{int(val)})')
    cv2.namedWindow('Pick/place'); cv2.setMouseCallback('Pick/place', on_mouse)
    print('CLICK the cube to grasp in the camera window (Q/ESC cancels).')
    while not state['done']:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError('Camera read failed')
        state['hsv'] = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.putText(frame, 'CLICK the cube to grasp', (15, 30), 0, .8, (0, 255, 0), 2)
        cv2.imshow('Pick/place', frame)
        if cv2.waitKey(1) & 255 in (27, ord('q')):
            raise KeyboardInterrupt
    cv2.setMouseCallback('Pick/place', lambda *args: None)


def read_q(robot, fk):
    values = robot.bus.sync_read('Present_Position')
    return {n: float(values[n]) for n in fk.names}, float(values['gripper'])


class ObservationRejected(ValueError):
    pass


class EmptyGrasp(RuntimeError):
    pass


class MotionFailure(RuntimeError):
    pass


def retry_settings(c):
    defaults = dict(observation_attempts=3, stable_frames=5, position_mm=5,
                    orientation_deg=5, wrist_scan_deg=12, grasp_attempts=3,
                    pregrasp_attempts=2)
    return defaults | c.get('retry', {})


def stable_observation(cap, robot, scene, required=None, stop=None):
    """Flush buffered images; require a stationary arm and consistent base-frame poses."""
    required = roles(scene.c) if required is None else required
    cfg = retry_settings(scene.c)
    readings, detections = [], []
    for i in range(5 + cfg['stable_frames']):
        q0, _ = read_q(robot, scene.fk)
        if stop is None:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Camera read failed')
            cv2.imshow('Pick/place', frame)
            if cv2.waitKey(1) & 255 in (27, ord('q')):
                raise KeyboardInterrupt
        else:
            frame = stop()
        q1, grip = read_q(robot, scene.fk)
        if i < 5:
            continue
        readings.extend([list(q0.values()), list(q1.values())])
        q = {n: (q0[n] + q1[n]) / 2 for n in q0}
        try:
            found, _ = scene.detect(frame, q)
        except ValueError as error:
            # Only target selection / pose ambiguity is recoverable; bad camera config is not.
            if 'resolution' in str(error):
                raise
            raise ObservationRejected(str(error)) from error
        if any(role not in found for role in required):
            raise ObservationRejected('Required target missing: ' + ', '.join(required))
        detections.append(found)
    if np.max(np.ptp(readings, axis=0)) > .3:
        raise ObservationRejected('Robot moved during observation (>0.3 deg)')
    result = dict(detections[-1])
    for role in required:
        poses = np.array([d[role]['T_base_marker'] for d in detections])
        if not np.isfinite(poses).all():
            raise ObservationRejected('Non-finite target pose')
        centre = np.median(poses[:, :3, 3], axis=0)
        if np.max(np.linalg.norm(poses[:, :3, 3] - centre, axis=1))*1000 > cfg['position_mm']:
            raise ObservationRejected('Target position unstable')
        if any(np.linalg.norm(cv2.Rodrigues(poses[-1, :3, :3].T @ R)[0]) > np.deg2rad(cfg['orientation_deg']) for R in poses[:, :3, :3]):
            raise ObservationRejected('Target orientation unstable')
        T = poses[-1].copy(); T[:3, 3] = centre
        result[role] = dict(result[role], T_base_marker=T.tolist())
    return {'time': time.time(), 'joints_deg': q, 'gripper': grip, 'targets': result}, frame


def observe(cap, robot, scene, output, scan=None, required=None, stop=None):
    output.mkdir(parents=True, exist_ok=True)
    if not scene.c.get('auto'):
        print('SPACE capture, Q/ESC cancel.')
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Camera read failed')
            cv2.imshow('Pick/place', frame)
            key = cv2.waitKey(1) & 255
            if key in (27, ord('q')):
                raise KeyboardInterrupt
            if key == 32:
                break
    failures = []
    for attempt in range(retry_settings(scene.c)['observation_attempts']):
        try:
            record, frame = stable_observation(cap, robot, scene, required, stop)
            save_json(output / 'observation.json', record)
            if not cv2.imwrite(str(output / 'observation.png'), frame):
                raise IOError('Image save failed')
            return record
        except ObservationRejected as error:
            failures.append(str(error))
            save_json(output / 'observation_retries.json', {'failures': failures})
            print(f'Observation {attempt+1} rejected: {error}')
            if attempt + 1 < retry_settings(scene.c)['observation_attempts']:
                if scan is not None:
                    scan(attempt)
                time.sleep(.3)
    raise ObservationRejected('Observation retries exhausted: ' + '; '.join(failures))


def wrist_search(robot, planner, origin, attempt, poll):
    """Small absolute wrist offsets; preflight the whole joint path before issuing commands."""
    c = planner.c
    q, grip = read_q(robot, planner.fk)
    start = np.array([q[n] for n in planner.fk.names])
    target = np.array([origin[n] for n in planner.fk.names])
    target[planner.fk.names.index('wrist_roll')] += (1 if attempt % 2 == 0 else -1) * retry_settings(c)['wrist_scan_deg']
    rows = np.linspace(start, target, max(2, int(np.ceil(np.max(np.abs(target-start))/.5))+1))
    for row in rows:
        planner.clearance(row)
        if planner.tcp(row)[2, 3] < c['table_z_m'] + c['cube']['side_mm']/1000 + c['clearance_m']:
            raise ValueError('Wrist search needs a high observation pose')
    try:
        for row in rows[1:]:
            poll()
            robot.bus.sync_write('Goal_Position', dict(zip(planner.fk.names, row)) | {'gripper': grip})
            time.sleep(.1)  # <=5 deg/s
            measured, _ = read_q(robot, planner.fk)
            if max(abs(measured[n]-row[i]) for i, n in enumerate(planner.fk.names)) > 4:
                raise MotionFailure('Wrist search tracking error >4 deg')
        time.sleep(.3)
    except BaseException:
        robot.bus.sync_write('Goal_Position', robot.bus.sync_read('Present_Position'))
        raise


class Planner:
    def __init__(self, fk, c, calibration=None):
        self.fk, self.c = fk, c
        self.tool = matrix(c['T_gripper_tcp'])
        self.plate = None
        if calibration:
            # Measured hard stops from LeRobot motor calibration, already in LeRobot degrees (0 = range midpoint).
            half = np.array([(calibration[n].range_max - calibration[n].range_min) / 2 * 360 / 4095 for n in fk.names])
            self.lo, self.hi = -half, half
        else:
            # URDF limits expressed in LeRobot degrees (q_urdf = q_lerobot + offset).
            offset = np.array([fk.offsets.get(n, 0) for n in fk.names])
            self.lo = np.array([np.rad2deg(float(j.find('limit').get('lower'))) + 2 for j in fk.chain if j.get('type') != 'fixed']) - offset
            self.hi = np.array([np.rad2deg(float(j.find('limit').get('upper'))) - 2 for j in fk.chain if j.get('type') != 'fixed']) - offset

    def tcp(self, q):
        return self.fk(dict(zip(self.fk.names, q))) @ self.tool

    def ik(self, target, seed):
        def residual(q):
            actual = self.tcp(q)
            return np.r_[(actual[:3, 3] - target[:3, 3]) * 100,
                         cv2.Rodrigues(target[:3, :3].T @ actual[:3, :3])[0].ravel()]
        result = least_squares(residual, np.clip(seed, self.lo + 1e-6, self.hi - 1e-6), bounds=(self.lo, self.hi), max_nfev=180,
                               ftol=1e-9, xtol=1e-9, gtol=1e-9)
        actual = self.tcp(result.x)
        pe = np.linalg.norm(actual[:3, 3] - target[:3, 3])
        re = np.rad2deg(np.linalg.norm(cv2.Rodrigues(target[:3, :3].T @ actual[:3, :3])[0]))
        if pe > .003:  # position only; orientation is best-effort on a 5-DOF arm  # ponytail: this 5-DOF model cannot point the tool straight down anywhere; accept a tilted grasp
            raise ValueError(f'Unreachable TCP pose: {pe*1000:.1f} mm / {re:.1f} deg; reposition targets')
        if np.max(np.abs(result.x - seed)) > 60:
            raise ValueError('IK discontinuity: joint step >60 deg')
        return result.x

    def clearance(self, q):
        if np.any(q < self.lo) or np.any(q > self.hi) or not np.isfinite(q).all():
            raise ValueError('Invalid joint angles / joint limit exceeded')
        # Conservative joint-to-joint capsules; no mesh/self-collision model.
        points = [T[:3, 3] for T in self.fk.frames(dict(zip(self.fk.names, q)))]
        # Exempt fixed base/shoulder mounting region; check elbow onwards.
        for a, b in zip(points[2:-1], points[3:]):
            if min(a[2], b[2]) - self.c['link_radius_m'] < self.c['table_z_m']:
                raise ValueError('Conservative arm/table clearance failed')
        tcp = self.tcp(q)[:3, 3]
        lowest = tcp[2] - max(self.c['tcp_lowest_extent_mm'], self.c['cube']['side_mm']-self.c['cube']['grasp_depth_mm'])/1000
        if tcp[2] - self.c['tcp_lowest_extent_mm']/1000 < self.c['table_z_m'] + .001 or lowest < self.c['table_z_m'] - .001:
            raise ValueError('Tool/cube envelope intersects table')
        if self.plate is not None:
            local = (np.linalg.inv(self.plate) @ np.r_[tcp, 1])[:3]
            radius = max(self.c['tool_radius_mm'], self.c['cube']['side_mm']/np.sqrt(2))/1000
            outer = np.array(self.c['plate']['outer_half_size_mm'])/1000
            inner = np.array(self.c['plate']['inner_half_size_mm'])/1000
            near = np.all(np.abs(local[:2]) < outer + radius)
            below_rim = lowest < self.plate[2, 3] + self.c['plate']['rim_height_mm']/1000 + .003
            if near and below_rim and np.any(np.abs(local[:2]) + radius + .003 > inner):
                raise ValueError('Tool/cube envelope intersects plate rim')
            if near and lowest < self.plate[2, 3] + .001:
                raise ValueError('Tool/cube envelope intersects plate floor')
        if np.any(tcp < self.c['workspace_min_m']) or np.any(tcp > self.c['workspace_max_m']):
            raise ValueError('TCP outside measured workspace')

    def segment(self, prev, target, q):
        """Straight-line TCP motion prev->target as a list of joint waypoints, starting from joints q."""
        angle = cv2.Rodrigues(prev[:3, :3].T @ target[:3, :3])[0].ravel()
        count = max(1, int(np.ceil(np.linalg.norm(target[:3, 3]-prev[:3, 3])/.005)), int(np.ceil(np.linalg.norm(angle)/.05)))
        rows = []
        for u in np.linspace(0, 1, count+1)[1:]:
            goal = transform(prev[:3, :3] @ cv2.Rodrigues(angle*u)[0], (1-u)*prev[:3, 3]+u*target[:3, 3])
            q = self.ik(goal, q)
            self.clearance(q)
            rows.append(q.tolist())
        return rows, q

    def refine(self, plan, delta, after='approach'):
        """Correct cube stages only; preserve the independently observed tray destination."""
        idx = next(i for i, s in enumerate(plan) if s['stage'] == after)
        q = np.array(plan[idx]['joints_deg'][-1]); prev = np.array(plan[idx]['T_base_tcp'])
        out = plan[:idx + 1]
        for s in plan[idx + 1:]:
            target = np.array(s['T_base_tcp'])
            if s['stage'] in ('hover', 'descend', 'grasp', 'lift'):
                target[:3, 3] += delta
            rows, q = self.segment(prev, target, q)
            out.append(dict(s, joints_deg=rows, T_base_tcp=target.tolist()))
            prev = target
        return out

    def plan(self, obs):
        c = self.c
        cube = matrix(obs['targets']['cube']['T_base_marker'])
        pick_only = c.get('pick_only', False)
        pick = cube[:3, 3] - np.array([0, 0, c['cube']['grasp_depth_mm']/1000]) + np.array(c.get('pick_offset_mm', [0, 0, 0]))/1000  # ponytail: manual bias correction until hand-eye is fixed
        # ponytail: hand-eye error can put the marker below its true height; never plan the tool into the table.
        floor = c['table_z_m'] + max(c['tcp_lowest_extent_mm'], c['cube']['side_mm'] - c['cube']['grasp_depth_mm'])/1000 + .002
        if pick[2] < floor:
            print(f'Pick height clamped from {pick[2]*1000:.0f} mm to {floor*1000:.0f} mm (table limit)')
            pick[2] = floor
        self.plate = None
        if pick_only:
            safe_z = cube[2, 3] + c['clearance_m']
        else:
            plate = matrix(obs['targets']['plate']['T_base_marker'])
            self.plate = plate
            drop = (plate @ np.r_[np.array(c['plate']['drop_offset_mm'])/1000, 1])[:3]
            drop[2] += (c['cube']['side_mm'] - c['cube']['grasp_depth_mm'] + c['release_gap_mm'])/1000
            safe_z = max(cube[2, 3], plate[2, 3] + c['plate']['rim_height_mm']/1000) + c['clearance_m'] + c['cube']['side_mm']/1000
        q0 = np.array([obs['joints_deg'][n] for n in self.fk.names])
        # ponytail: no hard rejection; widen the bounds so the observed pose is always inside them.
        self.lo, self.hi = np.minimum(self.lo, q0 - 1), np.maximum(self.hi, q0 + 1)
        self.clearance(q0)
        start = self.tcp(q0)
        if start[2, 3] < safe_z:
            print(f'Observation pose is low ({start[2,3]*1000:.0f} mm); align stage lifts to {safe_z*1000:.0f} mm first')
        candidates = []
        failures = []
        for yaw_offset in (0, np.pi/2, np.pi, 3*np.pi/2):
            yaw = np.arctan2(cube[1, 0], cube[0, 0]) + yaw_offset
            x = np.array([np.cos(yaw), np.sin(yaw), 0])
            z = np.array([0., 0., -1.])
            R = np.column_stack([x, np.cross(z, x), z])
            def pose(p):
                return transform(R, p)
            above_pick = pick.copy()
            above_pick[2] = safe_z
            align = start.copy()
            align[:3, :3] = R
            align[2, 3] = max(start[2, 3], safe_z)
            hover = pick.copy(); hover[2] += c.get('grasp_view', {}).get('hover_mm', 15)/1000
            targets = [('align', align, 'open'), ('approach', pose(above_pick), 'open'), ('hover', pose(hover), 'open'),
                       ('descend', pose(pick), 'open'), ('grasp', pose(pick), 'closed'),
                       ('lift', pose(above_pick), 'closed')]
            if not pick_only:
                above_drop = drop.copy()
                above_drop[2] = safe_z
                targets += [('transfer', pose(above_drop), 'closed'), ('place', pose(drop), 'closed'),
                            ('release', pose(drop), 'open'), ('retract', pose(above_drop), 'open')]
            q, prev, rows = q0.copy(), start.copy(), []
            try:
                for name, target, grip in targets:
                    segment, q = self.segment(prev, target, q)
                    rows.append({'stage': name, 'joints_deg': segment, 'gripper': c['gripper_'+grip], 'T_base_tcp': target.tolist()})
                    prev = target
                candidates.append((sum(np.linalg.norm(np.diff(np.vstack([q0]+[q for s in rows for q in s['joints_deg']]), axis=0), axis=1)), rows))
            except ValueError as e:
                failures.append(str(e))
        if not candidates:
            raise ValueError('No reachable square-aligned grasp: ' + '; '.join(failures))
        return min(candidates, key=lambda v: v[0])[1]


def pivot_tcp(poses):
    if len(poses) < 6:
        raise ValueError('Need >=6 distinct orientations with TCP at same fixed physical point')
    A = np.vstack([np.c_[T[:3, :3], -np.eye(3)] for T in poses])
    b = np.concatenate([-T[:3, 3] for T in poses])
    x, _, rank, s = np.linalg.lstsq(A, b, rcond=None)
    rms = float(np.sqrt(np.mean(np.sum((A @ x-b).reshape(-1, 3)**2, axis=1))))
    if rank < 6 or s[-1]/s[0] < .02 or rms > .003:
        raise ValueError(f'Poor pivot data: rank={rank}, RMS={rms*1000:.2f} mm')
    return x[:3], rms


def teach(robot, fk, out):
    poses = []
    print('No automatic torque release. Support arm and use existing manual setup. Keep the intended grasp center at ONE fixed point; vary wrist orientation.')
    while True:
        command = input('ENTER pivot sample; s solve translation; o teach orientation; q quit: ').strip().lower()
        if command == 'o':
            input('Set grasp insertion direction vertically DOWN (-base Z), jaw closing axis along +base X; ENTER when stationary: ')
            q, _ = read_q(robot, fk)
            R = fk(q)[:3, :3].T @ np.diag([1., -1., -1.])
            save_json(out / 'tcp_rotation.json', {'R_gripper_tcp': R.tolist(), 'joints_deg': q})
            print('Saved orientation; accuracy depends on physical alignment to BASE axes.')
            continue
        if command == 'q':
            return
        if command == 's':
            t, rms = pivot_tcp(poses)
            save_json(out / 'tcp_translation.json', {'translation_gripper_m': t.tolist(), 'pivot_rms_mm': rms*1000,
                     'note': 'Translation only. Define TCP rotation separately: +Z insertion direction; +X jaw closing direction.'})
            print('TCP translation in gripper_link:', t, 'RMS mm:', rms*1000)
            return
        q, _ = read_q(robot, fk)
        poses.append(fk(q))
        save_json(out / 'tcp_pivot_samples.json', [T.tolist() for T in poses])
        print('Recorded', len(poses))


def recovery_context(c):
    # Tray dimensions may be filled before recovery; grasp geometry must stay unchanged.
    keys = ('robot_id', 'port', 'calibration_dir', 'T_gripper_tcp', 'table_z_m',
            'cube', 'gripper_open', 'gripper_closed', 'tcp_lowest_extent_mm')
    data = {key: c.get(key) for key in keys}
    for key in ('urdf', 'handeye_json'):
        data[key] = (ROOT / c[key]).read_text()
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def validate_recovery_feedback(robot, fk, c, recovery):
    q, opening = read_q(robot, fk)
    command = float(recovery['held_command'])
    saved_opening = float(recovery['opening'])
    if (not np.isfinite([command, opening, saved_opening, *q.values()]).all()
            or not c['gripper_closed'] <= command < c['gripper_open']
            or opening <= c['gripper_closed'] + 5 or abs(opening-saved_opening) > 5):
        raise ValueError('Gripper feedback differs from saved grasp; recovery blocked')
    torque = robot.bus.sync_read('Torque_Enable')
    if any(torque.get(n) != 1 for n in [*fk.names, 'gripper']):
        raise ValueError('Recovery requires torque already enabled; support and inspect the arm')


def recovery_plan(planner, obs):
    """Fresh measured start -> vertical lift -> tray -> descend -> release -> retract."""
    c = planner.c
    planner.plate = matrix(obs['targets']['plate']['T_base_marker'])
    q = np.array([obs['joints_deg'][n] for n in planner.fk.names])
    planner.clearance(q)
    start = planner.tcp(q)
    drop = start.copy()
    drop[:3, 3] = (planner.plate @ np.r_[np.array(c['plate']['drop_offset_mm'])/1000, 1])[:3]
    drop[2, 3] += (c['cube']['side_mm']-c['cube']['grasp_depth_mm']+c['release_gap_mm'])/1000
    high = start.copy()
    high[2, 3] = max(start[2, 3], drop[2, 3], planner.plate[2, 3] +
                    c['plate']['rim_height_mm']/1000 + c['clearance_m'] +
                    max(c['tcp_lowest_extent_mm'], c['cube']['side_mm'])/1000)
    above = drop.copy(); above[2, 3] = high[2, 3]
    plan, prev = [], start
    for name, target in [('lift', high), ('transfer', above), ('place', drop), ('release', drop), ('retract', above)]:
        rows, q = planner.segment(prev, target, q)
        plan.append(dict(stage=name, joints_deg=rows, T_base_tcp=target.tolist(),
                         gripper=c['gripper_open'] if name in ('release','retract') else c['gripper_closed']))
        prev = target
    return plan


def execute(robot, planner, plan, obs, cap, out, scene=None, state_store=None, recovery=None):
    c = planner.c
    log = {'status': 'running', 'events': []}
    last = np.array([obs['joints_deg'][n] for n in planner.fk.names])
    frame_last = None
    held_grip = recovery['held_command'] if recovery else None
    def journal(state, **details):
        if state_store is not None:
            state_store.write(state, context=recovery_context(c), **details)
    grasp_attempt = 0
    pregrasp_retries = 0
    def event(stage, **kw):
        log['events'].append({'time': time.time(), 'stage': stage, **kw})
        save_json(out / 'execution.json', log)
        if frame_last is not None:
            cv2.imwrite(str(out / f'exec_{stage}.png'), frame_last)
    def check_stop():
        nonlocal frame_last
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError('Camera lost during execution')
        frame_last = frame
        cv2.imshow('Pick/place', frame)
        if cv2.waitKey(1) & 255 in (27, ord('q')):
            raise KeyboardInterrupt
        return frame
    def move(path, grip, allow_gripper_contact=False):
        """One continuous smooth motion through all waypoints of a stage; settle only at the end."""
        nonlocal last
        q, oldgrip = read_q(robot, planner.fk)
        old = np.array([q[n] for n in planner.fk.names])
        pts = np.vstack([old] + [np.asarray(p, dtype=float) for p in path])
        target = pts[-1]
        cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        duration = max(.3, np.sum(np.max(np.abs(np.diff(pts, axis=0)), axis=1))/c['speed_deg_s'], abs(grip-oldgrip)/20)
        steps = max(2, int(np.ceil(duration*25)))
        for u in np.linspace(0, 1, steps+1)[1:]:
            s = 3*u*u-2*u*u*u
            cmd = target.copy() if cum[-1] == 0 else np.array([np.interp(s*cum[-1], cum, pts[:, i]) for i in range(pts.shape[1])])
            planner.clearance(cmd)
            robot.bus.sync_write('Goal_Position', dict(zip(planner.fk.names, cmd)) | {'gripper': grip if held_grip is not None and allow_gripper_contact else oldgrip+(grip-oldgrip)*s})
            check_stop()
            measured, _ = read_q(robot, planner.fk)
            if max(abs(measured[n]-cmd[i]) for i, n in enumerate(planner.fk.names)) > 8:
                raise MotionFailure('Joint tracking error >8 deg')
            time.sleep(duration/steps)
        deadline = time.monotonic()+3
        boosts, boosted = 0, target.copy()
        while True:
            measured, measured_grip = read_q(robot, planner.fk)
            err = np.array([measured[n]-target[i] for i, n in enumerate(planner.fk.names)])
            if np.max(np.abs(err)) < 1 and (allow_gripper_contact or abs(measured_grip-grip) < 10):
                break
            if np.max(np.abs(err)) < 4 and boosts < 3 and time.monotonic() < deadline:
                # Gravity / load droop compensation: push the goal past the target by the measured error.
                boosted = np.clip(boosted - err, planner.lo, planner.hi)
                robot.bus.sync_write('Goal_Position', dict(zip(planner.fk.names, boosted)) | {'gripper': grip})
                boosts += 1
                time.sleep(.4)
                continue
            if np.max(np.abs(err)) < 4 and (allow_gripper_contact or abs(measured_grip-grip) < 10):
                break
            if time.monotonic() > deadline:
                diff = {n: round(measured[n]-target[i], 1) for i, n in enumerate(planner.fk.names)} | {'gripper': round(measured_grip-grip, 1)}
                raise MotionFailure(f'Motion failed to settle; measured-target = {diff}')
            check_stop()
            time.sleep(.05)
        last = target.copy()
    def close_until_contact():
        nonlocal held_grip
        journal('unknown', reason='closing_gripper')
        hold, opening = read_q(robot, planner.fk)
        hold = np.array([hold[n] for n in planner.fk.names])  # drift is measured from where the arm actually settled
        history = []
        command = opening
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            command = max(c['gripper_closed'], command - .5)
            robot.bus.write('Goal_Position', 'gripper', command)
            check_stop()
            q, actual = read_q(robot, planner.fk)
            if max(abs(q[n] - hold[i]) for i, n in enumerate(planner.fk.names)) > 3:
                raise RuntimeError('Arm moved while closing gripper')
            history.append(actual)
            if contact_candidate(history, command, c['gripper_closed']):
                # Keep the small current position error; do not continue squeezing to zero.
                held_grip = command
                journal('holding', held_command=command, opening=actual)
                event('possible_contact', measured_opening=actual, held_command=command)
                if not c.get('auto') and input('Possible contact detected. Is the object visibly between both fingers? Type yes to lift: ').strip().lower() != 'yes':
                    raise RuntimeError('Grasp not visually confirmed; no lift')
                return command
            if actual <= c['gripper_closed'] + 2:
                journal('empty', reason='closed_endpoint_empty_grasp')
                event('empty_grasp', measured_opening=actual)
                raise EmptyGrasp('Gripper reached closed endpoint: possible empty grasp; no lift')
            time.sleep(.05)
        raise RuntimeError('No stable contact before timeout; no lift')

    def cube_observation(label, scan=False):
        origin, grip = read_q(robot, planner.fk)
        def search(attempt):
            event('wrist_search', attempt=attempt+1, context=label)
            wrist_search(robot, planner, origin, attempt, check_stop)
        result = observe(cap, robot, scene, out / label, scan=search if scan else None,
                         required=('cube',), stop=check_stop)
        if scan:
            # Subsequent stage paths assume the original orientation. Restore after search.
            q, _ = read_q(robot, planner.fk)
            if max(abs(q[n]-origin[n]) for n in q) > .3:
                rows = np.linspace([q[n] for n in planner.fk.names], [origin[n] for n in planner.fk.names], 50)
                for row in rows:
                    planner.clearance(row)
                move(rows.tolist(), grip)
        return result

    def retreat_and_replan(retry):
        nonlocal obs, last
        q, grip = read_q(robot, planner.fk)
        seed = np.array([q[n] for n in planner.fk.names])
        start = planner.tcp(seed)
        high = start.copy()
        high[2, 3] = max(start[2, 3], next(np.array(s['T_base_tcp'])[2, 3] for s in plan if s['stage'] == 'approach'))
        rows, _ = planner.segment(start, high, seed)
        move(rows, grip, True)
        event('retry_retreat')
        if not retry:
            raise RuntimeError('Retry limit reached; stopped at high pose')
        if scene is None:
            raise RuntimeError('Cannot retry grasp without fresh vision')
        q, _ = read_q(robot, planner.fk)
        move([[q[n] for n in planner.fk.names]], c['gripper_open'])
        fresh = cube_observation('retry_cube', scan=True)
        # Keep the original destination; only the cube is reacquired.
        q, grip = read_q(robot, planner.fk)
        obs = dict(fresh, joints_deg=q, gripper=grip,
                   targets=obs['targets'] | {'cube': fresh['targets']['cube']})
        plan[:] = planner.plan(obs)
        save_json(out / 'retry_plan.json', {'observation': obs, 'stages': plan})

    try:
        current, grip = read_q(robot, planner.fk)
        if max(abs(current[n]-obs['joints_deg'][n]) for n in current) > .5:
            raise ValueError('Robot changed after observation; observe again')
        # Existing calibrated LeRobot controller setup; caller must support arm during configure.
        if recovery is None:
            robot.configure()
            robot.bus.sync_write('Goal_Position', current | {'gripper': grip})
            robot.bus.enable_torque(num_retry=5)
        else:
            # Never reconfigure/disable torque or replace preload with measured opening.
            robot.bus.sync_write('Goal_Position', current | {'gripper': held_grip})
        i = 0
        while i < len(plan):
            stage = plan[i]; i += 1
            if stage['stage'] == 'grasp':
                grasp_attempt += 1
                try:
                    held_grip = close_until_contact()
                except EmptyGrasp:
                    event('empty_grasp_retry', attempt=grasp_attempt)
                    # Retreat vertically while still closed, then open at the high pose.
                    retreat_and_replan(grasp_attempt < retry_settings(c)['grasp_attempts'])
                    i = 0
                    continue
                event('grasp', measured=robot.bus.sync_read('Present_Position'))
                continue
            carrying = stage['stage'] in ('lift', 'transfer', 'place')
            if carrying and held_grip is None:
                raise RuntimeError('Carrying stage without confirmed contact')
            if stage['stage'] == 'release':
                journal('unknown', reason='release_in_progress')
            move(stage['joints_deg'], held_grip if carrying else stage['gripper'], carrying)
            if stage['stage'] == 'release':
                held_grip = None
                journal('unknown', reason='release_command_completed_requires_visual_check')
            event(stage['stage'], measured=robot.bus.sync_read('Present_Position'))
            if stage['stage'] == 'approach' and scene is not None:
                try:
                    fresh = cube_observation('approach', scan=True)
                    old_p = np.array(obs['targets']['cube']['T_base_marker'])[:3, 3]
                    new_p = np.array(fresh['targets']['cube']['T_base_marker'])[:3, 3]
                    delta = np.r_[(new_p-old_p)[:2], 0.0]
                    if np.linalg.norm(delta) > .06:
                        raise ObservationRejected('Approach correction exceeds 60 mm')
                    plan[:] = planner.refine(plan, delta)
                    obs['targets']['cube'] = fresh['targets']['cube']
                    event('refine', delta_mm=(delta*1000).tolist())
                except ObservationRejected:
                    # Already high: stop after bounded acquisition, never descend blind.
                    raise
            if stage['stage'] == 'hover' and scene is not None:
                try:
                    fresh = cube_observation('pregrasp')
                    old_p = np.array(obs['targets']['cube']['T_base_marker'])[:3, 3]
                    new_p = np.array(fresh['targets']['cube']['T_base_marker'])[:3, 3]
                    if np.linalg.norm(new_p-old_p)*1000 > retry_settings(c)['position_mm']:
                        raise ObservationRejected('Cube moved since approach')
                    if c.get('grasp_view') and 'pixel' in fresh['targets']['cube']:
                        q_now, _ = read_q(robot, planner.fk)
                        T_base_cam = planner.fk(q_now) @ scene.X
                        top = c['table_z_m'] + c['cube']['side_mm']/1000
                        delta = scene.pixel_to_plane(T_base_cam, fresh['targets']['cube']['pixel'], top) - scene.pixel_to_plane(T_base_cam, c['grasp_view']['pixel'], top)
                        delta[2] = 0
                        if np.linalg.norm(delta) > .003:
                            if np.linalg.norm(delta) > .06:
                                raise ObservationRejected('Visual servo correction exceeds 60 mm')
                            plan[:] = planner.refine(plan, delta)
                            move(plan[i-1]['joints_deg'], c['gripper_open'])
                            # Recheck after the correction, with bounded corrections.
                            pregrasp_retries += 1
                            if pregrasp_retries >= retry_settings(c)['pregrasp_attempts']:
                                raise ObservationRejected('Visual servo did not converge')
                            i -= 1
                            continue
                except ObservationRejected as error:
                    pregrasp_retries += 1
                    event('pregrasp_retry', attempt=pregrasp_retries, reason=str(error))
                    retreat_and_replan(pregrasp_retries < retry_settings(c)['pregrasp_attempts'])
                    i = 0
                    continue
            if stage['stage'] == 'lift' and not c.get('auto'):
                answer = input('Visually check: cube is securely lifted? Type yes to continue: ').strip().lower()
                if answer != 'yes':
                    raise RuntimeError('Pickup not confirmed')
        if c.get('pick_only'):
            log['status'] = 'lift_contact_only_unverified' if c.get('auto') else 'lift_operator_verified_holding'
            print('Lift finished. Motors remain enabled, holding the object; no release command sent.')
        else:
            answer = 'auto' if c.get('auto') else input('Visually check: cube is inside plate and gripper clear? Type yes if successful: ').strip().lower()
            log['status'] = 'success_operator_verified' if answer == 'yes' else 'placement_not_verified'
            if answer == 'yes':
                journal('empty', reason='operator_verified_release')
    except BaseException as e:
        log['status'] = 'aborted'
        log['reason'] = str(e) or type(e).__name__
        log['holding_possible_object'] = held_grip is not None
        log['recovery'] = 'hold_and_inspect_no_automatic_restart' if held_grip is not None else 'stopped'
        if isinstance(e, MotionFailure):
            log['events'].append({'time': time.time(), 'stage': 'motion_failure', 'recovery': log['recovery']})
        # Hold current arm position; do not release a potentially loaded arm.
        try:
            hold = robot.bus.sync_read('Present_Position')
            if held_grip is not None:
                hold['gripper'] = held_grip
            robot.bus.sync_write('Goal_Position', hold)
        except Exception as hold_error:
            log['hold_error'] = str(hold_error)
        raise
    finally:
        save_json(out / 'execution.json', log)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'pick_place_config.json')
    p.add_argument('--output', type=Path, default=ROOT/'pick_place_output')
    p.add_argument('--locate-only', action='store_true')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--startup-state', choices=('empty', 'holding', 'unknown'),
                   help='Override startup state after inspection; defaults to empty unless journal records holding/unknown')
    p.add_argument('--recover-place', action='store_true',
                   help='Place an already held cube using fresh tray observation; requires --execute')
    p.add_argument('--pick-only', action='store_true', help='Auto-select one marker, grasp and lift; no plate required')
    p.add_argument('--teach-tcp', action='store_true')
    p.add_argument('--scan-markers', action='store_true', help='Show detected IDs without connecting robot')
    p.add_argument('--scan-color', action='store_true', help='Preview the colour mask (tune cube.hsv_ranges); no robot')
    p.add_argument('--teach-grasp-view', action='store_true', help='Hold the gripper centred just above the cube; records where the cube appears in the image')
    p.add_argument('--teach-tcp-click', action='store_true', help='Camera only: click the point between the fingertips; sets T_gripper_tcp translation')
    p.add_argument('--ignore-calibration-gate', action='store_true', help='Execute even if hand-eye consistency is poor (expect misses)')
    p.add_argument('--auto', action='store_true', help='No keypresses or confirmations: capture when still, run the plan straight through')
    a = p.parse_args()
    if sum([a.locate_only, a.execute, a.teach_tcp, a.scan_markers, a.scan_color, a.teach_grasp_view, a.teach_tcp_click]) > 1:
        p.error('Choose only one mode')
    if a.recover_place and (not a.execute or a.pick_only):
        p.error('--recover-place requires --execute without --pick-only')
    c = json.loads(a.config.read_text())
    c['pick_only'] = a.pick_only
    c['auto'] = a.auto
    c['recover_place'] = a.recover_place
    if a.recover_place:
        c['detector'] = 'aruco'  # Destination tag only; held cube need not be visible.
    a.output.mkdir(parents=True, exist_ok=True)
    if a.scan_color:
        cap = open_camera(c['camera'], 1280, 720)
        state = {'hsv': None, 'click': None}
        def on_mouse(ev, x, y, flags, _):
            if ev == cv2.EVENT_LBUTTONDOWN and state['hsv'] is not None and x < state['hsv'].shape[1]:
                patch = state['hsv'][max(0, y-4):y+5, max(0, x-4):x+5].reshape(-1, 3)
                h, sat, val = np.median(patch, axis=0)
                ranges = hsv_ranges_around(h, sat, val)
                c['cube']['hsv_ranges'] = ranges; state['click'] = (x, y)
                print(f'Picked HSV=({int(h)},{int(sat)},{int(val)}) -> hsv_ranges={ranges}. Press S to save, Q to quit.')
        win = 'Colour mask: click the cube | S save | Q quit'
        cv2.namedWindow(win); cv2.setMouseCallback(win, on_mouse)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Camera read failed')
                state['hsv'] = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = np.zeros(frame.shape[:2], np.uint8)
                for lo, hi in c['cube']['hsv_ranges']:
                    mask |= cv2.inRange(state['hsv'], np.array(lo, np.uint8), np.array(hi, np.uint8))
                if state['click']:
                    cv2.circle(frame, state['click'], 8, (0, 255, 0), 2)
                cv2.imshow(win, np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))
                key = cv2.waitKey(1) & 255
                if key in (27, ord('q')):
                    break
                if key == ord('s'):
                    saved = json.loads(a.config.read_text()); saved['cube']['hsv_ranges'] = c['cube']['hsv_ranges']
                    save_json(a.config, saved); print('Saved hsv_ranges to', a.config)
        finally:
            cap.release(); cv2.destroyAllWindows()
        return
    if a.teach_tcp_click:
        scene = Scene(c, URDFFK(ROOT/c['urdf']))
        cap = open_camera(c['camera'], *scene.size)
        tcp = matrix(c['T_gripper_tcp']); Xinv = np.linalg.inv(scene.X)
        tcp_cam = Xinv @ tcp                      # model TCP in camera frame
        state = {'click': None}
        def on_mouse(ev, x, y, flags, _):
            if ev == cv2.EVENT_LBUTTONDOWN:
                state['click'] = (x, y)
        win = 'Click between the FINGERTIPS (grasp centre) | S save | Q quit'
        cv2.namedWindow(win); cv2.setMouseCallback(win, on_mouse)
        print('Red circle = where the model thinks the fingertips are. Click the real grasp centre, then S to save.')
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Camera read failed')
                proj, _ = cv2.projectPoints(tcp_cam[:3, 3].reshape(1, 1, 3), np.zeros(3), np.zeros(3), scene.K, scene.D)
                p0 = tuple(np.round(proj.reshape(2)).astype(int))
                cv2.circle(frame, p0, 14, (0, 0, 255), 3); cv2.putText(frame, 'model TCP', (p0[0]+16, p0[1]), 0, .8, (0, 0, 255), 2)
                if state['click']:
                    cv2.circle(frame, state['click'], 14, (0, 255, 0), 3); cv2.putText(frame, 'new TCP', (state['click'][0]+16, state['click'][1]), 0, .8, (0, 255, 0), 2)
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 255
                if key in (27, ord('q')):
                    break
                if key == ord('s') and state['click']:
                    # Ray through the clicked pixel, at the same camera depth as the model TCP (lateral correction only).
                    und = cv2.undistortPoints(np.array([[state['click']]], np.float64), scene.K, scene.D).reshape(2)
                    new_cam = np.array([und[0], und[1], 1.0]) * tcp_cam[2, 3]
                    new_grip = scene.X[:3, :3] @ new_cam + scene.X[:3, 3]
                    shift = new_grip - tcp[:3, 3]
                    saved = json.loads(a.config.read_text())
                    T = np.array(saved['T_gripper_tcp']); T[:3, 3] = new_grip
                    saved['T_gripper_tcp'] = T.tolist(); saved['tcp_source'] = 'teach-tcp-click'; saved['tcp_status'] = f'clicked in image; shift from URDF {np.round(shift*1000,1).tolist()} mm'
                    save_json(a.config, saved)
                    print(f'Saved T_gripper_tcp: moved {np.round(shift*1000,1).tolist()} mm in gripper frame; was pixel {p0}, now {state["click"]}')
        finally:
            cap.release(); cv2.destroyAllWindows()
        return
    if a.scan_markers:
        cap = open_camera(c['camera'], 1280, 720)
        detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, c['dictionary'])))
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Camera read failed')
                corners, ids, _ = detector.detectMarkers(frame)
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                cv2.imshow('Marker IDs (Q quits)', frame)
                if cv2.waitKey(1) & 255 in (27, ord('q')):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
        return
    handeye_path = ROOT/c['handeye_json']
    offsets = json.loads(handeye_path.read_text()).get('joint_offsets_deg') if handeye_path.exists() else None
    fk = URDFFK(ROOT/c['urdf'], offsets=offsets)
    if not a.teach_tcp:
        if a.execute and not a.ignore_calibration_gate:
            calibration_gate(json.loads(handeye_path.read_text()))
        elif a.execute:
            print('WARNING: calibration gate ignored; hand-eye error is cm-level, expect misses.')
        check_config(c, planning=not a.locate_only)
        scene = Scene(c, fk)
    robot, cap, state_store = None, None, None
    recovery = None
    try:
        if a.execute or a.teach_tcp or a.teach_grasp_view:
            state_store = GraspState(c)
            previous = state_store.read()
            print('Saved gripper state:', previous['state'], 'journal:', state_store.path)
            confirmed = a.startup_state
            if confirmed is None and not a.auto and (a.recover_place or previous['state'] != 'empty'):
                confirmed = input('Inspect gripper NOW: empty / holding / unknown: ').strip().lower()
            decision = startup_decision(previous, confirmed, a.recover_place)
            if decision == 'recover':
                if previous.get('context') != recovery_context(c):
                    raise ValueError('Grasp calibration/config changed; automatic recovery blocked')
                recovery = previous
            else:
                state_store.write('empty', reason='operator_confirmed_empty_at_startup' if confirmed == 'empty' else 'default_empty_at_startup')
        robot = robot_reader(argparse.Namespace(port=c['port'], robot_id=c['robot_id'],
                    calibration_dir=Path(c['calibration_dir']) if c['calibration_dir'] else None, disable_torque=a.teach_grasp_view))
        if a.teach_tcp:
            teach(robot, fk, a.output)
            return
        if a.teach_grasp_view:
            cap = open_camera(c['camera'], *scene.size)
            print('Torque is OFF. Put the OPEN gripper centred over the cube, fingertips ~5 mm above its top. Hold still...')
            pick_color_by_click(cap, c)
            pixels = []
            for _ in range(15):
                ok, frame = cap.read()
                q, _ = read_q(robot, scene.fk)
                found, view = scene.detect(frame, q)
                if 'cube' in found:
                    pixels.append(found['cube']['pixel'])
                cv2.imshow('Pick/place', view); cv2.waitKey(1)
            if len(pixels) < 8:
                raise ValueError('Cube not detected reliably')
            px = np.median(pixels, axis=0).tolist()
            saved = json.loads(a.config.read_text())
            saved['grasp_view'] = {'pixel': px, 'hover_mm': 15, 'joints_deg': q, 'hsv_ranges': c['cube']['hsv_ranges']}
            saved['cube']['hsv_ranges'] = c['cube']['hsv_ranges']
            save_json(a.config, saved)
            print(f'Saved grasp_view pixel {np.round(px)} to {a.config}')
            return
        cap = open_camera(c['camera'], *scene.size)
        if recovery is not None:
            validate_recovery_feedback(robot, fk, c, recovery)
            # No wrist search, reconfiguration, or motor commands during observation/planning.
            obs = observe(cap, robot, scene, a.output/'recovery_observation', required=('plate',))
            planner = Planner(fk, c, robot.calibration)
            plan = recovery_plan(planner, obs)
            save_json(a.output/'recovery_plan.json', {'observation': obs, 'stages': plan})
            if not a.auto and input('Review recovery path to tray; type place to proceed: ').strip() != 'place':
                return
            fresh = observe(cap, robot, scene, a.output/'recovery_recheck', required=('plate',))
            old = matrix(obs['targets']['plate']['T_base_marker'])
            new = matrix(fresh['targets']['plate']['T_base_marker'])
            if np.linalg.norm(old[:3,3]-new[:3,3]) > .005 or np.linalg.norm(cv2.Rodrigues(old[:3,:3].T @ new[:3,:3])[0]) > np.deg2rad(5):
                raise ValueError('Tray moved during recovery review; stopped')
            if abs(fresh['gripper']-recovery['opening']) > 5:
                raise ValueError('Gripper changed during recovery observation; stopped')
            # Replan from fresh measured joints, never replay a saved joint trajectory.
            plan = recovery_plan(planner, fresh)
            validate_recovery_feedback(robot, fk, c, recovery)
            execute(robot, planner, plan, fresh, cap, a.output, scene, state_store, recovery)
            return
        if a.execute:
            # Hold the observation pose with torque so the operator can let go (steadier camera, no hand tremor).
            robot.bus.sync_write('Goal_Position', robot.bus.sync_read('Present_Position'))
            robot.bus.enable_torque(num_retry=5)
            print('Torque ON: arm is holding itself. Let go, keep hands out of view, observing...')
            time.sleep(1)
        if c.get('detector') == 'color':
            pick_color_by_click(cap, c)
        planner = Planner(fk, c, robot.calibration) if a.execute else None
        origin, _ = read_q(robot, fk)
        def search_poll():
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Camera lost during search')
            cv2.imshow('Pick/place', frame)
            if cv2.waitKey(1) & 255 in (27, ord('q')):
                raise KeyboardInterrupt
        def initial_search(attempt):
            wrist_search(robot, planner, origin, attempt, search_poll)
        obs = observe(cap, robot, scene, a.output, scan=initial_search if a.execute else None)
        print(json.dumps(obs['targets'], indent=2))
        if a.locate_only:
            return
        planner = Planner(fk, c, robot.calibration)
        plan = planner.plan(obs)
        save_json(a.output/'plan.json', {'config': c, 'observation': obs, 'stages': plan,
                  'collision_model': 'Partial arm/table capsules, tool/cube plate envelope, TCP bounds; no mesh/self-collision guarantee'})
        for s in plan:
            print(s['stage'], 'TCP m:', np.round(np.array(s['T_base_tcp'])[:3, 3], 4), 'samples:', len(s['joints_deg']))
        if not a.execute:
            print('Plan saved. No motion commands sent. Inspect plan and physical clearance before --execute.')
            return
        print('Support arm during controller setup. Clear path/table/plate rim and check gripper geometry. Q/ESC in camera window or Ctrl-C stops commands and attempts hold.')
        if not a.auto and input('Scene unchanged and path reviewed? Type execute: ').strip() != 'execute':
            return
        # Reject stale target positions after human review; plan must describe this observation.
        fresh = observe(cap, robot, scene, a.output/'recheck')
        for role in roles(c):
            old = matrix(obs['targets'][role]['T_base_marker'])
            new = matrix(fresh['targets'][role]['T_base_marker'])
            if np.linalg.norm(old[:3, 3]-new[:3, 3]) > .02 or np.linalg.norm(cv2.Rodrigues(old[:3, :3].T @ new[:3, :3])[0]) > np.deg2rad(10):
                raise ValueError('Scene changed; rerun observation and planning')
        execute(robot, planner, plan, obs, cap, a.output, scene, state_store)
    finally:
        if cap is not None:
            cap.release()
        if robot is not None:
            robot.bus.disconnect(disable_torque=False)
        if state_store is not None:
            state_store.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
