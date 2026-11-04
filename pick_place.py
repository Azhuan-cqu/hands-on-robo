#!/usr/bin/env python3
"""SO101 orange-cube / blue-tray vision pick/place. Default: observe and plan, no motor commands."""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from camera_intrinsic import open_camera, save_json
from handeye_calibration import URDFFK, robot_reader, transform

from detect_cube import detect_cube_top, draw_cube_detection
from detect_tray import detect_tray_center, draw_tray_detection

ROOT = Path(__file__).resolve().parent


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
    return ('cube',) if c.get('pick_only') else ('cube', 'plate')


def check_config(c, planning=True):
    if c.get('table_z_m') is None or not np.isfinite(c['table_z_m']):
        raise ValueError('Fill table_z_m with the measured table height')
    if c['cube'].get('side_mm') is None or not np.isfinite(c['cube']['side_mm']) or c['cube']['side_mm'] <= 0:
        raise ValueError('Fill cube.side_mm with a positive measured size')
    for role in roles(c):
        cfg = c[role]
        lo, hi = cfg.get('hsv_low'), cfg.get('hsv_high')
        if lo is None or hi is None or len(lo) != 3 or len(hi) != 3 or any(
                not 0 <= l <= h <= limit for l, h, limit in zip(lo, hi, [179, 255, 255])):
            raise ValueError(f'Invalid {role}.hsv_low / hsv_high')
        if cfg.get('min_area_px', 400) <= 0 or not 0 <= cfg.get('canny_low', 15) < cfg.get('canny_high', 40):
            raise ValueError(f'Invalid {role} area / Canny thresholds')
    if not c.get('pick_only'):
        for key in ('surface_height_mm', 'floor_height_mm'):
            value = c['plate'].get(key)
            if value is None or not np.isfinite(value) or value < 0:
                raise ValueError(f'Fill plate.{key}: measured height above the table, mm')
    if planning:
        for key in ('T_gripper_tcp', 'table_z_m', 'workspace_min_m', 'workspace_max_m', 'gripper_open', 'gripper_closed', 'tool_radius_mm', 'tcp_lowest_extent_mm'):
            if c[key] is None:
                raise ValueError(f'Fill {key} in config first')
        matrix(c['T_gripper_tcp'])
        dimensions = [('cube', 'side_mm'), ('cube', 'grasp_depth_mm')]
        if not c.get('pick_only'):
            dimensions += [('plate', 'rim_height_mm'), ('plate', 'inner_half_size_mm'), ('plate', 'outer_half_size_mm')]
        for role, key in dimensions:
            if c[role][key] is None:
                raise ValueError(f'Fill {role}.{key} in config first')
        if not c.get('pick_only'):
            for key in ('inner_half_size_mm', 'outer_half_size_mm'):
                value = c['plate'][key]
                try:
                    size = np.asarray(value, dtype=float)
                except (TypeError, ValueError):
                    size = np.array([])
                if size.shape != (2,) or not np.isfinite(size).all() or np.any(size <= 0):
                    raise ValueError(
                        f'plate.{key} must be [long_side_mm / 2, short_side_mm / 2], '
                        f'two positive measured values; got {value!r}. '
                        'Use --pick-only to omit tray placement.')
                if size[0] < size[1]:
                    raise ValueError(f'plate.{key} must list the long-side half-size first')
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


def detect_objects(frame, c):
    """Shared 2D detection for robot observation and camera-only preview."""
    found, debug = {}, {}
    view = frame.copy()
    for role in roles(c):
        cfg = c[role]
        detector = detect_cube_top if role == 'cube' else detect_tray_center
        result, details = detector(frame, hsv_low=cfg['hsv_low'], hsv_high=cfg['hsv_high'],
                                   min_area=cfg.get('min_area_px', 400),
                                   canny_low=cfg.get('canny_low', 15), canny_high=cfg.get('canny_high', 40))
        debug[role] = details
        if result is not None:
            found[role] = result
            # Render only successful geometry; show combined status below.
            draw = draw_cube_detection if role == 'cube' else draw_tray_detection
            rendered = draw(view, result)
            view[35:] = rendered[35:]
    status = ' | '.join(f'{role}: {"OK" if role in found else "not found"}' for role in roles(c))
    cv2.putText(view, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 220, 255), 2)
    return found, view, debug


class Scene:
    def __init__(self, c, fk):
        self.c, self.fk = c, fk
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

    def pixel_to_plane(self, T_base_cam, px, z_plane):
        """Cast the ray through pixel px and intersect with the horizontal plane z=z_plane (base frame)."""
        und = cv2.undistortPoints(np.array([[px]], np.float64), self.K, self.D).reshape(2)
        ray = T_base_cam[:3, :3] @ np.array([und[0], und[1], 1.0])
        origin = T_base_cam[:3, 3]
        if abs(ray[2]) < 1e-6 or (z_plane - origin[2]) / ray[2] <= 0:
            raise ValueError('Ray does not hit the table plane in front of the camera')
        return origin + ray * (z_plane - origin[2]) / ray[2]

    def object_pose(self, result, role, T_base_cam):
        """Intersect visible corners with a measured horizontal plane.

        Cube origin: top center. Plate origin: floor center, with XY measured
        on surface_height_mm (the plane of the detected outer border).
        Plate +X follows the longer edge, +Y the shorter edge; choose +X
        toward positive base X (positive Y on the tie). A plain rectangle
        has no identifiable front/back, so offsets use this convention.
        """
        cfg = self.c[role]
        height = cfg['side_mm'] if role == 'cube' else cfg['surface_height_mm']
        z = self.c['table_z_m'] + height / 1000
        corners = result['top_corners_px' if role == 'cube' else 'corners_px']
        points = np.array([self.pixel_to_plane(T_base_cam, px, z) for px in corners])
        center = self.pixel_to_plane(T_base_cam, result['center_px'], z)
        # Opposite edges are averaged; cyclic order from detectors is arbitrary.
        axes = [points[1] - points[0] + points[2] - points[3],
                points[2] - points[1] + points[3] - points[0]]
        if role == 'plate':
            x = max(axes, key=np.linalg.norm)
        else:
            # Square symmetry: choose the axis closest to base +X.
            x = max(axes, key=lambda v: abs(v[0]) / max(np.linalg.norm(v), 1e-12))
        x[2] = 0
        if np.linalg.norm(x) < 1e-8:
            raise ValueError(f'Degenerate {role} boundary')
        x /= np.linalg.norm(x)
        if x[0] < -1e-8 or (abs(x[0]) <= 1e-8 and x[1] < 0):
            x = -x
        if role == 'plate':
            center[2] = self.c['table_z_m'] + cfg['floor_height_mm'] / 1000
        R = np.column_stack([x, np.cross([0, 0, 1], x), [0, 0, 1]])
        B = transform(R, center)
        C = np.linalg.inv(T_base_cam) @ B
        return {'T_base_object': B.tolist(), 'T_camera_object': C.tolist(),
                'pixel': np.asarray(result['center_px']).tolist(),
                'corners_px': np.asarray(corners).tolist(), 'edge_support': result['edge_support']}

    def detect(self, frame, q):
        if (frame.shape[1], frame.shape[0]) != self.size:
            raise ValueError('Camera resolution differs from calibration')
        detections, view, _ = detect_objects(frame, self.c)
        T_base_cam = self.fk(q) @ self.X
        found = {role: self.object_pose(result, role, T_base_cam)
                 for role, result in detections.items()}
        return found, view


def read_q(robot, fk):
    values = robot.bus.sync_read('Present_Position')
    return {n: float(values[n]) for n in fk.names}, float(values['gripper'])


def observe(cap, robot, scene, output):
    output.mkdir(parents=True, exist_ok=True)
    print('Hold still with target object(s) visible. SPACE capture, Q/ESC cancel.')
    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError('Camera read failed')
        cv2.imshow('Pick/place', frame)
        key = cv2.waitKey(1) & 255
        if key in (27, ord('q')):
            raise KeyboardInterrupt
        if key != 32 and not scene.c.get('auto'):
            continue
        readings = []
        accepted = None
        required = set(roles(scene.c))
        try:
            for _ in range(10):
                q0, _ = read_q(robot, scene.fk)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Camera read failed')
                q1, grip = read_q(robot, scene.fk)
                readings.extend([list(q0.values()), list(q1.values())])
                q = {n: (q0[n] + q1[n]) / 2 for n in q0}
                found, view = scene.detect(frame, q)
                cv2.imshow('Pick/place', view)
                if cv2.waitKey(1) & 255 in (27, ord('q')):
                    raise KeyboardInterrupt
                if not required.issubset(found):
                    continue
                # Keep the last complete detection together with its own
                # image and robot readings, even if later frames miss targets.
                accepted = ({'time': time.time(), 'joints_deg': q,
                             'gripper': grip, 'targets': found}, frame.copy())
            if accepted is None:
                raise ValueError('No frame detected all required targets in 10 frames: '
                                 + ', '.join(sorted(required)))
            if np.max(np.ptp(readings, axis=0)) > .3:
                raise ValueError('Robot moved during observation')
            record, accepted_frame = accepted
            save_json(output / 'observation.json', record)
            if not cv2.imwrite(str(output / 'observation.png'), accepted_frame):
                raise IOError('Image save failed')
            return record
        except ValueError as e:
            print('Capture rejected:', e)
            if scene.c.get('auto'):
                time.sleep(1)


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
        if self.plate is not None:
            local = (np.linalg.inv(self.plate) @ np.r_[tcp, 1])[:3]
            radius = max(self.c['tool_radius_mm'], self.c['cube']['side_mm']/np.sqrt(2))/1000
            outer = np.array(self.c['plate']['outer_half_size_mm'])/1000
            inner = np.array(self.c['plate']['inner_half_size_mm'])/1000
            near = np.all(np.abs(local[:2]) < outer + radius)
            below_rim = lowest < self.plate[2, 3] + self.c['plate']['rim_height_mm']/1000 + .003
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
        """Shift every stage after `after` by delta (base frame, m) and recompute its joint path."""
        idx = next(i for i, s in enumerate(plan) if s['stage'] == after)
        q = np.array(plan[idx]['joints_deg'][-1]); prev = np.array(plan[idx]['T_base_tcp'])
        out = plan[:idx + 1]
        for s in plan[idx + 1:]:
            target = np.array(s['T_base_tcp']); target[:3, 3] += delta
            rows, q = self.segment(prev, target, q)
            out.append(dict(s, joints_deg=rows, T_base_tcp=target.tolist()))
            prev = target
        return out

    def plan(self, obs):
        c = self.c
        cube = matrix(obs['targets']['cube']['T_base_object'])
        pick_only = c.get('pick_only', False)
        pick = cube[:3, 3] - np.array([0, 0, c['cube']['grasp_depth_mm']/1000]) + np.array(c.get('pick_offset_mm', [0, 0, 0]))/1000  # ponytail: manual bias correction until hand-eye is fixed
        # ponytail: hand-eye error can put the target below its true height; never plan the tool into the table.
        floor = c['table_z_m'] + max(c['tcp_lowest_extent_mm'], c['cube']['side_mm'] - c['cube']['grasp_depth_mm'])/1000 + .002
        if pick[2] < floor:
            print(f'Pick height clamped from {pick[2]*1000:.0f} mm to {floor*1000:.0f} mm (table limit)')
            pick[2] = floor
        self.plate = None
        if pick_only:
            safe_z = cube[2, 3] + c['clearance_m']
        else:
            plate = matrix(obs['targets']['plate']['T_base_object'])
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


def execute(robot, planner, plan, obs, cap, out, scene=None):
    c = planner.c
    log = {'status': 'running', 'events': []}
    last = np.array([obs['joints_deg'][n] for n in planner.fk.names])
    frame_last = None
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
    def move(path, grip, allow_gripper_contact=False, stage_name='motion'):
        """One continuous smooth motion through all waypoints of a stage; settle only at the end."""
        nonlocal last
        q, oldgrip = read_q(robot, planner.fk)
        old = np.array([q[n] for n in planner.fk.names])
        pts = np.vstack([old] + [np.asarray(p, dtype=float) for p in path])
        target = pts[-1]
        cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        duration = max(.3, np.sum(np.max(np.abs(np.diff(pts, axis=0)), axis=1))/c['speed_deg_s'], abs(grip-oldgrip)/20)
        steps = max(2, int(np.ceil(duration*25)))
        print(f'[{stage_name}] Moving: planned duration {duration:.1f}s, gripper target {grip:.1f}', flush=True)
        for u in np.linspace(0, 1, steps+1)[1:]:
            s = 3*u*u-2*u*u*u
            cmd = target.copy() if cum[-1] == 0 else np.array([np.interp(s*cum[-1], cum, pts[:, i]) for i in range(pts.shape[1])])
            planner.clearance(cmd)
            robot.bus.sync_write('Goal_Position', dict(zip(planner.fk.names, cmd)) | {'gripper': oldgrip+(grip-oldgrip)*s})
            check_stop()
            measured, _ = read_q(robot, planner.fk)
            if max(abs(measured[n]-cmd[i]) for i, n in enumerate(planner.fk.names)) > 8:
                raise RuntimeError('Joint tracking error >8 deg')
            time.sleep(duration/steps)
        deadline = time.monotonic()+3
        next_report = 0.0
        boosts, boosted = 0, target.copy()
        while True:
            measured, measured_grip = read_q(robot, planner.fk)
            err = np.array([measured[n]-target[i] for i, n in enumerate(planner.fk.names)])
            if np.max(np.abs(err)) < 2 and (allow_gripper_contact or abs(measured_grip-grip) < 10):
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
            now = time.monotonic()
            diff = {n: round(float(err[i]), 1) for i, n in enumerate(planner.fk.names)}
            diff['gripper'] = round(float(measured_grip - grip), 1)
            if now >= next_report:
                print(f'[{stage_name}] Waiting to settle; measured-target = {diff}; '
                      f'gripper check={not allow_gripper_contact}', flush=True)
                next_report = now + 1
            if now > deadline:
                raise RuntimeError(f'[{stage_name}] Motion failed to settle within 3s; '
                                   f'measured-target = {diff}')
            check_stop()
            time.sleep(.05)
        last = target.copy()
        print(f'[{stage_name}] Motion complete', flush=True)
    def close_until_contact():
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
                event('possible_contact', measured_opening=actual, held_command=command)
                if not c.get('auto') and input('Possible contact detected. Is the object visibly between both fingers? Type yes to lift: ').strip().lower() != 'yes':
                    raise RuntimeError('Grasp not visually confirmed; no lift')
                return command
            if actual <= c['gripper_closed'] + 2:
                event('empty_grasp', measured_opening=actual)
                raise RuntimeError('Gripper reached closed endpoint: possible empty grasp; no lift')
            time.sleep(.05)
        raise RuntimeError('No stable contact before timeout; no lift')

    try:
        current, grip = read_q(robot, planner.fk)
        if max(abs(current[n]-obs['joints_deg'][n]) for n in current) > .5:
            raise ValueError('Robot changed after observation; observe again')
        # Existing calibrated LeRobot controller setup; caller must support arm during configure.
        robot.configure()
        robot.bus.sync_write('Goal_Position', current | {'gripper': grip})
        robot.bus.enable_torque(num_retry=5)
        held_grip = None
        i = 0
        while i < len(plan):
            stage = plan[i]; i += 1
            print(f'Starting stage {i}/{len(plan)}: {stage["stage"]}', flush=True)
            if c.get('pick_only') and stage['stage'] == 'grasp':
                held_grip = close_until_contact()
                event('grasp', measured=robot.bus.sync_read('Present_Position'))
                continue
            grip_command = held_grip if c.get('pick_only') and stage['stage'] == 'lift' else stage['gripper']
            move(stage['joints_deg'], grip_command, stage['stage'] in ('grasp', 'lift', 'transfer', 'place'), stage['stage'])
            event(stage['stage'], measured=robot.bus.sync_read('Present_Position'))
            if stage['stage'] == 'approach' and scene is not None and c.get('refine_at_approach', True):
                # Closed-loop correction: the camera is now ~10 cm from the cube, so hand-eye rotation error matters far less.
                time.sleep(.5)
                for _ in range(5):
                    check_stop()
                q_now, _ = read_q(robot, planner.fk)
                found, _ = scene.detect(frame_last, q_now)
                if 'cube' in found:
                    old_p = np.array(obs['targets']['cube']['T_base_object'])[:3, 3]
                    new_p = np.array(found['cube']['T_base_object'])[:3, 3]
                    delta = np.r_[(new_p - old_p)[:2], 0.0]
                    print(f'Refine at approach: cube moved {np.round(delta*1000,1).tolist()} mm in base xy')
                    if np.linalg.norm(delta) < .06:
                        try:
                            plan[:] = planner.refine(plan, delta)
                            event('refine', delta_mm=(delta*1000).tolist())
                        except ValueError as e:
                            print('Refine failed, keeping original plan:', e)
                    else:
                        print('Refine delta too large; keeping original plan')
                else:
                    print('Refine: cube not seen from approach pose; keeping original plan')
            if stage['stage'] == 'hover' and scene is not None and c.get('grasp_view'):
                # Visual servo: bring the cube to the taught pixel (physically fixed w.r.t. the fingers).
                ref = tuple(c['grasp_view']['pixel'])
                for k in range(4):
                    time.sleep(.4)
                    for _ in range(5):
                        check_stop()
                    q_now, _ = read_q(robot, planner.fk)
                    found, _ = scene.detect(frame_last, q_now)
                    if 'cube' not in found:
                        print('Servo: cube not seen; continuing with current plan'); break
                    T_base_cam = planner.fk(q_now) @ scene.X
                    top = c['table_z_m'] + c['cube']['side_mm']/1000
                    delta = scene.pixel_to_plane(T_base_cam, found['cube']['pixel'], top) - scene.pixel_to_plane(T_base_cam, ref, top)
                    delta[2] = 0
                    print(f'Servo {k+1}: cube is {np.round(delta*1000,1).tolist()} mm from the taught grasp view')
                    event('servo', delta_mm=(delta*1000).tolist())
                    if np.linalg.norm(delta) < .003:
                        break
                    if np.linalg.norm(delta) > .06:
                        print('Servo delta too large; giving up servo'); break
                    plan[:] = planner.refine(plan, delta, after='approach')
                    hover_rows = plan[i - 1]['joints_deg']  # recomputed hover stage (same index as current)
                    move(hover_rows, stage['gripper'], False, 'hover-servo')
            if stage['stage'] == 'lift' and not c.get('auto'):
                answer = input('Visually check: cube is securely lifted? Type yes to continue: ').strip().lower()
                if answer != 'yes':
                    raise RuntimeError('Pickup not confirmed')
        if c.get('pick_only'):
            log['status'] = 'lift_operator_verified_holding'
            print('Lift finished. Motors remain enabled, holding the object; no release command sent.')
        else:
            answer = 'auto' if c.get('auto') else input('Visually check: cube is inside plate and gripper clear? Type yes if successful: ').strip().lower()
            log['status'] = 'success_operator_verified' if answer == 'yes' else 'placement_not_verified'
    except BaseException as e:
        log['status'] = 'aborted'
        log['reason'] = str(e) or type(e).__name__
        # Hold current arm position; do not release a potentially loaded arm.
        try:
            hold = robot.bus.sync_read('Present_Position')
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
    p.add_argument('--pick-only', action='store_true', help='Detect the orange cube, grasp and lift; no tray required')
    p.add_argument('--teach-tcp', action='store_true')
    p.add_argument('--scan-color', action='store_true', help='Preview cube/tray detection and masks; no robot or calibration needed')
    p.add_argument('--teach-grasp-view', action='store_true', help='Hold the gripper centred just above the cube; records where the cube appears in the image')
    p.add_argument('--teach-tcp-click', action='store_true', help='Camera only: click the point between the fingertips; sets T_gripper_tcp translation')
    p.add_argument('--ignore-calibration-gate', action='store_true', help='Execute even if hand-eye consistency is poor (expect misses)')
    p.add_argument('--auto', action='store_true', help='No keypresses or confirmations: capture when still, run the plan straight through')
    a = p.parse_args()
    if sum([a.locate_only, a.execute, a.teach_tcp, a.scan_color, a.teach_grasp_view, a.teach_tcp_click]) > 1:
        p.error('Choose only one mode')
    c = json.loads(a.config.read_text())
    c['pick_only'] = a.pick_only or a.teach_grasp_view
    c['auto'] = a.auto
    a.output.mkdir(parents=True, exist_ok=True)
    if a.scan_color:
        cap = open_camera(c['camera'], 1280, 720)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Camera read failed')
                _, view, debug = detect_objects(frame, c)
                cv2.imshow('Cube / tray detection', view)
                for role, details in debug.items():
                    cv2.imshow(f'{role} mask', details['mask'])
                if cv2.waitKey(1) & 255 in (27, ord('q')):
                    break
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
                    saved['T_gripper_tcp'] = T.tolist()
                    save_json(a.config, saved)
                    print(f'Saved T_gripper_tcp: moved {np.round(shift*1000,1).tolist()} mm in gripper frame; was pixel {p0}, now {state["click"]}')
        finally:
            cap.release(); cv2.destroyAllWindows()
        return
    handeye_path = ROOT/c['handeye_json']
    offsets = json.loads(handeye_path.read_text()).get('joint_offsets_deg') if handeye_path.exists() else None
    fk = URDFFK(ROOT/c['urdf'], offsets=offsets)
    if not a.teach_tcp:
        if a.execute and not a.ignore_calibration_gate:
            calibration_gate(json.loads(handeye_path.read_text()))
        elif a.execute:
            print('WARNING: calibration gate ignored; hand-eye error is cm-level, expect misses.')
        check_config(c, planning=not (a.locate_only or a.teach_grasp_view))
        scene = Scene(c, fk)
    robot, cap = None, None
    try:
        robot = robot_reader(argparse.Namespace(port=c['port'], robot_id=c['robot_id'],
                    calibration_dir=Path(c['calibration_dir']) if c['calibration_dir'] else None, disable_torque=a.teach_grasp_view))
        if a.teach_tcp:
            teach(robot, fk, a.output)
            return
        if a.teach_grasp_view:
            cap = open_camera(c['camera'], *scene.size)
            print('Torque is OFF. Put the OPEN gripper centred over the cube, fingertips ~5 mm above its top. Hold still...')
            pixels = []
            for _ in range(15):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Camera read failed')
                q, _ = read_q(robot, scene.fk)
                found, view = scene.detect(frame, q)
                if 'cube' in found:
                    pixels.append(found['cube']['pixel'])
                cv2.imshow('Pick/place', view); cv2.waitKey(1)
            if len(pixels) < 8:
                raise ValueError('Cube not detected reliably')
            px = np.median(pixels, axis=0).tolist()
            saved = json.loads(a.config.read_text())
            saved['grasp_view'] = {'pixel': px, 'hover_mm': 15, 'joints_deg': q}
            save_json(a.config, saved)
            print(f'Saved grasp_view pixel {np.round(px)} to {a.config}')
            return
        cap = open_camera(c['camera'], *scene.size)
        if a.execute:
            # Hold the observation pose with torque so the operator can let go (steadier camera, no hand tremor).
            robot.bus.sync_write('Goal_Position', robot.bus.sync_read('Present_Position'))
            robot.bus.enable_torque(num_retry=5)
            print('Torque ON: arm is holding itself. Let go, keep hands out of view, observing...')
            time.sleep(1)
        obs = observe(cap, robot, scene, a.output)
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
            old = matrix(obs['targets'][role]['T_base_object'])
            new = matrix(fresh['targets'][role]['T_base_object'])
            if np.linalg.norm(old[:3, 3]-new[:3, 3]) > .02 or np.linalg.norm(cv2.Rodrigues(old[:3, :3].T @ new[:3, :3])[0]) > np.deg2rad(10):
                raise ValueError('Scene changed; rerun observation and planning')
        execute(robot, planner, plan, obs, cap, a.output, scene)
    finally:
        if cap is not None:
            cap.release()
        if robot is not None:
            robot.bus.disconnect(disable_torque=False)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
