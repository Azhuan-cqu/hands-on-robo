#!/usr/bin/env python3
"""SO101 orange-cube / blue-tray vision pick/place. Default: observe and plan, no motor commands."""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from camera_intrinsic import load_camera, open_camera, save_json
from handeye_calibration import URDFFK, matrix, robot_reader, transform
from detect_cube import COLOR_HSV_RANGES, detect_cube_top, draw_cube_detection
from detect_tray import detect_tray_center, draw_tray_detection
from placement_success import cube_overlaps_tray

ROOT = Path(__file__).resolve().parent


def contact_candidate(readings, command, closed):
    """Stable position with closing error: possible contact, not proof of grasp."""
    if len(readings) < 8:
        return False
    tail = np.asarray(readings[-8:], dtype=float)
    return bool(np.isfinite(tail).all() and np.ptp(tail) <= .5 and
                np.mean(tail) > closed + 5 and np.mean(tail) - command >= 3)


def roles(c):
    return ('cube',) if c.get('pick_only') else ('cube', 'plate')


def check_config(c):
    for key in ('T_gripper_tcp', 'table_z_m', 'workspace_min_m', 'workspace_max_m', 'gripper_open', 'gripper_closed', 'tcp_lowest_extent_mm'):
        if c.get(key) is None:
            raise ValueError(f'Fill {key} in config first')
    matrix(c['T_gripper_tcp'])
    if not np.isfinite(c['table_z_m']):
        raise ValueError('Fill table_z_m with the measured table height')
    if not 0 < c['cube']['grasp_depth_mm'] < c['cube']['side_mm']:
        raise ValueError('Grasp depth must be between cube top and bottom')
    if c['cube']['color'] not in COLOR_HSV_RANGES:
        raise ValueError(f'cube.color must be one of {tuple(COLOR_HSV_RANGES)}')
    for role in roles(c):
        cfg = c[role]
        if cfg['min_area_px'] <= 0 or not 0 <= cfg['canny_low'] < cfg['canny_high']:
            raise ValueError(f'Invalid {role} area / Canny thresholds')
    if not c.get('pick_only'):
        for key in ('surface_height_mm', 'floor_height_mm'):
            value = c['plate'].get(key)
            if value is None or not np.isfinite(value) or value < 0:
                raise ValueError(f'Fill plate.{key}: measured height above the table, mm')
    if not 0 <= c['gripper_closed'] < c['gripper_open'] <= 100:
        raise ValueError('Gripper values use LeRobot 0..100; closed < open')
    if not 0 < c['grip_squeeze'] <= 20:
        raise ValueError('grip_squeeze: how far past contact to close, 0..20 gripper units')
    if c['speed_deg_s'] <= 0 or c['speed_deg_s'] > 15 or c['clearance_m'] < .03 or c['link_radius_m'] <= 0:
        raise ValueError('Invalid speed/clearance/radius')
    if c['tcp_lowest_extent_mm'] < 0 or (not c.get('pick_only') and c['plate']['rim_height_mm'] < 0):
        raise ValueError('Invalid tool/plate envelope')
    if not np.all(np.asarray(c['workspace_min_m']) < c['workspace_max_m']):
        raise ValueError('Invalid workspace bounds')


def calibration_gate(h):
    # Task policy, not a universal guarantee of accuracy.
    errors = np.array([h.get('board_position_rms_mm', float('inf')), h.get('board_rotation_rms_deg', float('inf'))], dtype=float)
    if not np.isfinite(errors).all() or np.any(errors < 0) or np.any(errors > [5, 2]):
        raise ValueError('Execution blocked: hand-eye consistency exceeds 5 mm / 2 deg. Recalibrate and validate on independent poses.')


def detect_objects(frame, c, pose_context=None):
    """Shared detection; cube uses 3D top-face check when pose_context is given, else 2D preview."""
    found, debug = {}, {}
    view = frame.copy()
    for role in roles(c):
        cfg = c[role]
        size = dict(min_area=cfg['min_area_px'], canny_low=cfg['canny_low'], canny_high=cfg['canny_high'])
        if role == 'cube':  # HSV ranges come from the color preset in detect_cube
            result, details = detect_cube_top(frame, color=cfg['color'], pose_context=pose_context, **size)
            draw = draw_cube_detection
        else:  # tray uses detect_tray's default blue range
            result, details = detect_tray_center(frame, **size)
            draw = draw_tray_detection
        debug[role] = details
        if result is not None:
            found[role] = result
            view[35:] = draw(view, result)[35:]  # keep the combined status line below
    status = ' | '.join(f'{role}: {"OK" if role in found else "not found"}' for role in roles(c))
    cv2.putText(view, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 220, 255), 2)
    return found, view, debug


class Scene:
    def __init__(self, c, fk):
        self.c, self.fk = c, fk
        self.K, self.D, self.size, _ = load_camera(ROOT / c['camera_yml'])
        h = json.loads((ROOT / c['handeye_json']).read_text())
        if h.get('translation_unit') != 'meter' or h.get('gripper_frame') != 'gripper_link':
            raise ValueError('Hand-eye units/frame mismatch')
        self.X = matrix(h['T_gripper_camera'])

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
        return {'T_base_object': transform(R, center).tolist(), 'pixel': np.asarray(result['center_px']).tolist()}

    def detect(self, frame, q):
        """q: joint array in fk.names order."""
        if (frame.shape[1], frame.shape[0]) != self.size:
            raise ValueError('Camera resolution differs from calibration')
        T_base_cam = self.fk(dict(zip(self.fk.names, q))) @ self.X
        cube = self.c['cube']
        side = cube['side_mm'] / 1000
        context = dict(K=self.K, D=self.D, T_base_camera=T_base_cam,
                       side_m=side, top_z_m=self.c['table_z_m'] + side)
        context.update({k: cube[k] for k in ('enforce_height', 'height_tolerance_m', 'max_tilt_deg', 'max_reprojection_px') if k in cube})
        detections, view, _ = detect_objects(frame, self.c, context)
        found = {role: self.object_pose(result, role, T_base_cam)
                 for role, result in detections.items()}
        return found, view


def cube_yaw(T):
    return np.arctan2(T[1][0], T[0][0])


def read_q(robot, names):
    """Arm joints (array, in `names` order) and gripper opening."""
    values = robot.bus.sync_read('Present_Position')
    return np.array([float(values[n]) for n in names]), float(values['gripper'])


def observe(cap, robot, scene, output):
    output.mkdir(parents=True, exist_ok=True)
    if not scene.c.get('auto'):
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
                q0, _ = read_q(robot, scene.fk.names)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Camera read failed')
                q1, _ = read_q(robot, scene.fk.names)
                readings.extend([q0, q1])
                q = (q0 + q1) / 2
                found, view = scene.detect(frame, q)
                cv2.imshow('Pick/place', view)
                if cv2.waitKey(1) & 255 in (27, ord('q')):
                    raise KeyboardInterrupt
                if not required.issubset(found):
                    continue
                # Keep the last complete detection together with its own
                # image and robot readings, even if later frames miss targets.
                accepted = ({'time': time.time(), 'joints_deg': dict(zip(scene.fk.names, q.tolist())), 'targets': found}, frame.copy())
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
            # Small pull toward the seed: a 5-DOF arm cannot match every orientation, and without it
            # tiny target shifts let the wrist wander tens of degrees between equally poor fits.
            return np.r_[(actual[:3, 3] - target[:3, 3]) * 100,
                         cv2.Rodrigues(target[:3, :3].T @ actual[:3, :3])[0].ravel(),
                         (q - seed) * .003]
        result = least_squares(residual, np.clip(seed, self.lo + 1e-6, self.hi - 1e-6), bounds=(self.lo, self.hi), max_nfev=180,
                               ftol=1e-9, xtol=1e-9, gtol=1e-9)
        actual = self.tcp(result.x)
        pe = np.linalg.norm(actual[:3, 3] - target[:3, 3])
        re = np.rad2deg(np.linalg.norm(cv2.Rodrigues(target[:3, :3].T @ actual[:3, :3])[0]))
        if pe > .003:  # ponytail: position only; a 5-DOF arm cannot always point straight down, accept a tilted grasp
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

    def refine(self, plan, T, after='approach'):
        """Apply T (base frame, old cube pose -> new) to the pick stages after `after`;
        place stages keep their tray targets. Recompute every later joint path."""
        idx = next(i for i, s in enumerate(plan) if s['stage'] == after)
        q = np.array(plan[idx]['joints_deg'][-1]); prev = np.array(plan[idx]['T_base_tcp'])
        out = plan[:idx + 1]
        for s in plan[idx + 1:]:
            target = np.array(s['T_base_tcp'])
            if s['stage'] in ('hover', 'descend', 'grasp', 'lift'):
                target = T @ target
            rows, q = self.segment(prev, target, q)
            out.append(dict(s, joints_deg=rows, T_base_tcp=target.tolist()))
            prev = target
        return out

    def plan(self, obs):
        c = self.c
        cube = matrix(obs['targets']['cube']['T_base_object'])
        pick_only = c.get('pick_only', False)
        pick = cube[:3, 3] - np.array([0, 0, c['cube']['grasp_depth_mm']/1000]) + np.array(c['pick_offset_mm'])/1000  # ponytail: manual bias correction until hand-eye is fixed
        # ponytail: hand-eye error can put the target below its true height; never plan the tool into the table.
        floor = c['table_z_m'] + max(c['tcp_lowest_extent_mm'], c['cube']['side_mm'] - c['cube']['grasp_depth_mm'])/1000 + .002
        if pick[2] < floor:
            print(f'Pick height clamped from {pick[2]*1000:.0f} mm to {floor*1000:.0f} mm (table limit)')
            pick[2] = floor
        if pick_only:
            safe_z = cube[2, 3] + c['clearance_m']
        else:
            plate = matrix(obs['targets']['plate']['T_base_object'])
            drop = (plate @ np.r_[np.array(c['plate']['drop_offset_mm'])/1000, 1])[:3]
            # The cube hangs -pick_offset (xy, base frame) from the TCP: offset the TCP the same way so the cube lands on target.
            drop[:2] += np.array(c['pick_offset_mm'][:2])/1000
            drop[2] += (c['cube']['side_mm'] - c['cube']['grasp_depth_mm'] + c['release_gap_mm'])/1000
            safe_z = max(cube[2, 3], plate[2, 3] + c['plate']['rim_height_mm']/1000) + c['clearance_m'] + c['cube']['side_mm']/1000
        q0 = np.array([obs['joints_deg'][n] for n in self.fk.names])
        # ponytail: no hard rejection; widen the bounds so the observed pose is always inside them.
        self.lo, self.hi = np.minimum(self.lo, q0 - 1), np.maximum(self.hi, q0 + 1)
        self.clearance(q0)
        start = self.tcp(q0)
        candidates = []
        failures = []
        for yaw_offset in (0, np.pi/2, np.pi, 3*np.pi/2):
            yaw = np.arctan2(cube[1, 0], cube[0, 0]) + yaw_offset
            x = np.array([np.cos(yaw), np.sin(yaw), 0])
            z = np.array([0., 0., -1.])
            R = np.column_stack([x, np.cross(z, x), z])
            pose = lambda p: transform(R, p)
            above_pick = pick.copy()
            above_pick[2] = safe_z
            align = start.copy()
            align[:3, :3] = R
            align[2, 3] = max(start[2, 3], safe_z)
            hover = pick.copy(); hover[2] += .015
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


def execute(robot, planner, plan, obs, cap, out, scene):
    c = planner.c
    log = {'status': 'running', 'events': []}
    frame_last, banner = None, ''
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
        if banner:
            frame = frame.copy()
            cv2.putText(frame, banner, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                        (0, 0, 255) if banner.startswith('FAILED') else (0, 255, 0), 3)
        cv2.imshow('Pick/place', frame)
        if cv2.waitKey(1) & 255 in (27, ord('q')):
            raise KeyboardInterrupt
    def goal(q, grip):
        return dict(zip(names, q)) | {'gripper': grip}
    def move(path, grip, stage_name, check_gripper):
        """One continuous smooth motion through all waypoints of a stage; settle only at the end."""
        old, oldgrip = read_q(robot, names)
        pts = np.vstack([old] + [np.asarray(p, dtype=float) for p in path])
        target = pts[-1]
        cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        duration = max(.3, np.sum(np.max(np.abs(np.diff(pts, axis=0)), axis=1))/c['speed_deg_s'], abs(grip-oldgrip)/20)
        steps = max(2, int(np.ceil(duration*25)))
        for u in np.linspace(0, 1, steps+1)[1:]:
            s = 3*u*u-2*u*u*u
            cmd = target.copy() if cum[-1] == 0 else np.array([np.interp(s*cum[-1], cum, col) for col in pts.T])
            planner.clearance(cmd)
            robot.bus.sync_write('Goal_Position', goal(cmd, oldgrip+(grip-oldgrip)*s))
            check_stop()
            if np.max(np.abs(read_q(robot, names)[0] - cmd)) > 8:
                raise RuntimeError('Joint tracking error >8 deg')
            time.sleep(duration/steps)
        deadline = time.monotonic()+3
        boosts, boosted = 0, target.copy()
        while True:
            measured, measured_grip = read_q(robot, names)
            err = measured - target
            worst = np.max(np.abs(err))
            grip_ok = not check_gripper or abs(measured_grip-grip) < 10
            # Loaded and stretched out, the arm sags ~5 deg; compensate anything up to the 8 deg tracking limit.
            can_boost = worst < 8 and boosts < 3 and time.monotonic() < deadline
            if grip_ok and (worst < 2 or (worst < 4 and not can_boost)):
                break
            if can_boost:
                # Gravity / load droop compensation: push the goal past the target by the measured error.
                # Ramp the goal over ~0.3 s: a single jump of up to 8 deg makes the arm visibly twitch.
                new = np.clip(boosted - err, planner.lo, planner.hi)
                for u in np.linspace(0, 1, 9)[1:]:
                    robot.bus.sync_write('Goal_Position', goal(boosted + (new - boosted) * u, grip))
                    time.sleep(.04)
                boosted = new
                boosts += 1
                time.sleep(.1)
                continue
            if time.monotonic() > deadline:
                # Continue anyway (operator's choice); the per-step 8 deg tracking check still guards the motion.
                diff = dict(zip(names, np.round(err, 1).tolist())) | {'gripper': round(measured_grip - grip, 1)}
                print(f'  [{stage_name}] not settled, continuing: measured-target = {diff}', flush=True)
                break
            check_stop()
            time.sleep(.05)
    def close_until_contact():
        hold, command = read_q(robot, names)  # drift is measured from where the arm actually settled
        history = []
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            command = max(c['gripper_closed'], command - .5)
            robot.bus.write('Goal_Position', 'gripper', command)
            check_stop()
            q, actual = read_q(robot, names)
            if np.max(np.abs(q - hold)) > 3:
                raise RuntimeError('Arm moved while closing gripper')
            history.append(actual)
            if contact_candidate(history, command, c['gripper_closed']):
                # Press grip_squeeze past contact: firmer hold, and if the cube slips out the
                # gripper closes visibly further, which is what the pick-up check looks for.
                command = max(c['gripper_closed'], actual - c['grip_squeeze'])
                robot.bus.write('Goal_Position', 'gripper', command)
                print(f'  contact at gripper {actual:.1f}, holding at {command:.1f}', flush=True)
                event('contact', measured_opening=actual, held_command=command)
                return actual, command
            # Stalled near closed (fingers on each other, or a pinched sliver): stop before the servo overloads.
            if actual <= c['gripper_closed'] + 2 or contact_candidate(history, command, -np.inf):
                robot.bus.write('Goal_Position', 'gripper', actual)
                event('empty_grasp', measured_opening=actual)
                raise RuntimeError('Gripper reached closed endpoint: possible empty grasp; no lift')
            time.sleep(.05)
        raise RuntimeError('No stable contact before timeout; no lift')
    def locate_cube(q_now):
        """Mean cube position (base frame) and yaw of 3 detections, trying up to 15 frames; None if fewer than 3."""
        seen, yaws = [], []
        for _ in range(15):
            check_stop()
            found, _ = scene.detect(frame_last, q_now)
            if 'cube' in found:
                T = np.array(found['cube']['T_base_object'])
                seen.append(T[:3, 3])
                yaws.append(cube_yaw(T))
                if len(seen) == 3:
                    print(f'  refine: 3 detections, spread {np.ptp(seen, axis=0)[:2]*1000} mm')
                    # Square: yaw is only defined mod 90 deg, so average on the 4x circle.
                    return np.mean(seen, axis=0), np.angle(np.mean(np.exp(4j * np.array(yaws)))) / 4
        return None
    def refine(i):
        """One close-range look above the cube (camera ~10 cm away, so hand-eye rotation error matters far less):
        correct the cube xy and yaw once, at safe height (at hover the open fingers are already below the cube top).
        Pick stages follow the cube; place stages keep the tray. No re-check: views from new poses disagree
        by more than they help, and chasing them oscillates."""
        time.sleep(.5)
        for _ in range(3):
            check_stop()  # flush stale camera frames
        q_now = read_q(robot, names)[0]
        located = locate_cube(q_now)
        if located is None:
            print('  refine: cube not seen reliably, keeping plan')
            return
        seen, seen_yaw = located
        old = np.array(obs['targets']['cube']['T_base_object'])
        cube_p = old[:3, 3]
        delta = np.r_[(seen - cube_p)[:2], 0.0]
        dyaw = (seen_yaw - cube_yaw(old) + np.pi/4) % (np.pi/2) - np.pi/4  # same grasp side, no wrist flip
        print(f'  refine: off by {np.round(delta[:2]*1000, 1).tolist()} mm, {np.degrees(dyaw):.1f} deg')
        if np.linalg.norm(delta) < .003 and abs(dyaw) < np.radians(3):
            return
        if np.linalg.norm(delta) >= .06 or abs(dyaw) > np.radians(30):
            print('  refine: correction too large, keeping plan')
            return
        # Rotate about the vertical through the cube, then move it to where it was seen.
        Rz = cv2.Rodrigues(np.array([0, 0, dyaw]))[0]
        T = transform(Rz, cube_p + delta - Rz @ cube_p)
        stage = plan[i]
        here = np.array(stage['T_base_tcp'])
        there = T @ here
        try:
            rows, _ = planner.segment(here, there, q_now)
            shifted = planner.refine(plan[:i] + [dict(stage, joints_deg=rows, T_base_tcp=there.tolist())] + plan[i+1:], T)
        except ValueError as e:
            print('  refine failed, keeping plan:', e)
            return
        move(rows, stage['gripper'], 'refine', True)
        plan[:] = shifted
        event('refine', delta_mm=(delta*1000).tolist(), dyaw_deg=float(np.degrees(dyaw)))

    names = planner.fk.names
    try:
        current, grip = read_q(robot, names)
        if np.max(np.abs(current - [obs['joints_deg'][n] for n in names])) > .5:
            raise ValueError('Robot changed after observation; observe again')
        # Existing calibrated LeRobot controller setup; caller must support arm during configure.
        robot.configure()
        robot.bus.sync_write('Goal_Position', goal(current, grip))
        robot.bus.enable_torque(num_retry=5)
        contact, held_grip = None, None
        for i in range(len(plan)):
            stage = plan[i]  # refine may replace later stages in place
            print(f'[{i+1}/{len(plan)}] {stage["stage"]}', flush=True)
            if stage['stage'] == 'grasp':
                contact, held_grip = close_until_contact()
                continue
            # While holding, keep the squeeze command instead of closing to gripper_closed.
            holding = held_grip is not None and stage['gripper'] == c['gripper_closed']
            move(stage['joints_deg'], held_grip if holding else stage['gripper'], stage['stage'], not holding)
            event(stage['stage'], measured=robot.bus.sync_read('Present_Position'))
            if stage['stage'] == 'approach' and c['refine_at_approach']:
                refine(i)
            if stage['stage'] == 'lift':
                opening = read_q(robot, names)[1]
                # Cube still between the fingers holds them near the contact opening; empty fingers close to held_grip.
                if opening < contact - c['grip_squeeze'] / 2:
                    raise RuntimeError(f'Cube not held after lift: gripper {opening:.1f}, contact was {contact:.1f}')
                banner = 'CUBE PICKED UP'
                print(f'>>> CUBE PICKED UP (gripper {opening:.1f}, contact {contact:.1f})', flush=True)
                event('picked_up', opening=opening, contact=contact)
        if c.get('pick_only'):
            log['status'] = 'picked_up_holding'
        else:
            # Retracted above the tray, looking down: most cube pixels inside the tray footprint = placed.
            votes = []
            for _ in range(5):
                check_stop()
                _, _, debug = detect_objects(frame_last, c)
                votes.append(cube_overlaps_tray(debug['cube']['mask'], debug['plate']['mask']))
            success = sum(votes) >= 3
            banner = 'SUCCESS' if success else 'FAILED'
            log['status'] = 'success' if success else 'failed'
            event('result', success=success, votes=votes)
            print(f'>>> RESULT: {banner}' + ('' if success else ' (cube not inside tray)'), flush=True)
            for _ in range(10):
                check_stop()  # keep the verdict on screen for a moment
    except BaseException as e:
        log['status'] = 'failed'
        log['reason'] = str(e) or type(e).__name__
        # Hold current arm position; do not release a potentially loaded arm.
        try:
            hold = robot.bus.sync_read('Present_Position')
            robot.bus.sync_write('Goal_Position', hold)
        except Exception as hold_error:
            log['hold_error'] = str(hold_error)
        banner = 'FAILED: ' + log['reason'][:40]
        try:
            for _ in range(50):
                check_stop()  # show the verdict for ~2 s; Q skips
        except BaseException:
            pass
        raise
    finally:
        save_json(out / 'execution.json', log)


def scan_color(c):
    """Camera only: live cube/tray detection and masks, to check the color preset and lighting."""
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


def teach_tcp_click(c, config_path):
    """Camera only: click the point between the fingertips; rewrites T_gripper_tcp translation."""
    scene = Scene(c, URDFFK(ROOT/c['urdf']))
    cap = open_camera(c['camera'], *scene.size)
    tcp = matrix(c['T_gripper_tcp'])
    tcp_cam = np.linalg.inv(scene.X) @ tcp  # model TCP in camera frame
    proj, _ = cv2.projectPoints(tcp_cam[:3, 3].reshape(1, 1, 3), np.zeros(3), np.zeros(3), scene.K, scene.D)
    p0 = tuple(np.round(proj.reshape(2)).astype(int))
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
                saved = json.loads(config_path.read_text())
                T = np.array(saved['T_gripper_tcp']); T[:3, 3] = new_grip
                saved['T_gripper_tcp'] = T.tolist()
                save_json(config_path, saved)
                print(f'Saved T_gripper_tcp: moved {np.round(shift*1000,1).tolist()} mm in gripper frame; was pixel {p0}, now {state["click"]}')
    finally:
        cap.release(); cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'pick_place_config.json')
    p.add_argument('--output', type=Path, default=ROOT/'pick_place_output')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--pick-only', action='store_true', help='Detect the orange cube, grasp and lift; no tray required')
    p.add_argument('--scan-color', action='store_true', help='Preview cube/tray detection and masks; no robot or calibration needed')
    p.add_argument('--teach-tcp-click', action='store_true', help='Camera only: click the point between the fingertips; sets T_gripper_tcp translation')
    p.add_argument('--ignore-calibration-gate', action='store_true', help='Execute even if hand-eye consistency is poor (expect misses)')
    p.add_argument('--auto', action='store_true', help='No keypresses or confirmations: capture when still, run the plan straight through')
    p.add_argument('--no-refine', action='store_true', help='Skip the second cube detection above the cube; grasp where first seen')
    a = p.parse_args()
    if sum([a.execute, a.scan_color, a.teach_tcp_click]) > 1:
        p.error('Choose only one mode')
    c = json.loads(a.config.read_text())
    c['pick_only'] = a.pick_only
    c['auto'] = a.auto
    c['refine_at_approach'] = not a.no_refine
    if a.scan_color:
        return scan_color(c)
    if a.teach_tcp_click:
        return teach_tcp_click(c, a.config)
    handeye = json.loads((ROOT/c['handeye_json']).read_text())
    fk = URDFFK(ROOT/c['urdf'], offsets=handeye.get('joint_offsets_deg'))
    if a.execute and not a.ignore_calibration_gate:
        calibration_gate(handeye)
    elif a.execute:
        print('WARNING: calibration gate ignored; expect misses if hand-eye error is large.')
    check_config(c)
    scene = Scene(c, fk)
    robot, cap = None, None
    try:
        robot = robot_reader(argparse.Namespace(port=c['port'], robot_id=c['robot_id'],
                    calibration_dir=Path(c['calibration_dir']) if c['calibration_dir'] else None, disable_torque=False))
        cap = open_camera(c['camera'], *scene.size)
        if a.execute:
            # Hold the observation pose with torque so the operator can let go (steadier camera, no hand tremor).
            robot.bus.sync_write('Goal_Position', robot.bus.sync_read('Present_Position'))
            robot.bus.enable_torque(num_retry=5)
            print('Torque ON: arm is holding itself. Let go, keep hands out of view, observing...')
            time.sleep(1)
        obs = observe(cap, robot, scene, a.output)
        planner = Planner(fk, c, robot.calibration)
        plan = planner.plan(obs)
        save_json(a.output/'plan.json', {'config': c, 'observation': obs, 'stages': plan})
        if not a.auto:
            for s in plan:
                print(s['stage'], 'TCP m:', np.round(np.array(s['T_base_tcp'])[:3, 3], 4), 'samples:', len(s['joints_deg']))
        if not a.execute:
            print('Plan saved. No motion commands sent. Inspect plan and physical clearance before --execute.')
            return
        if not a.auto:
            print('Support arm during controller setup. Clear path/table/plate rim and check gripper geometry. Q/ESC in camera window or Ctrl-C stops commands and attempts hold.')
            if input('Scene unchanged and path reviewed? Type execute: ').strip() != 'execute':
                return
            # Reject stale target positions after human review; plan must describe this observation.
            fresh = observe(cap, robot, scene, a.output/'recheck')
            for role in roles(c):
                old = matrix(obs['targets'][role]['T_base_object'])
                new = matrix(fresh['targets'][role]['T_base_object'])
                if np.linalg.norm(old[:3, 3]-new[:3, 3]) > .02 or np.linalg.norm(cv2.Rodrigues(old[:3, :3].T @ new[:3, :3])[0]) > np.deg2rad(10):
                    raise ValueError('Scene changed; rerun observation and planning')
        execute(robot, planner, plan, obs, cap, a.output, scene)
    except (RuntimeError, ValueError, KeyboardInterrupt) as e:
        # Task failures (missed grasp, unreachable target, Q/Ctrl-C) are a result, not a crash: no traceback.
        print(f'>>> RESULT: FAILED ({str(e) or type(e).__name__})', flush=True)
        raise SystemExit(1)
    finally:
        if cap is not None:
            cap.release()
        if robot is not None:
            robot.bus.disconnect(disable_torque=False)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
