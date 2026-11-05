import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import cv2
import numpy as np
from handeye_calibration import URDFFK
from pick_place import Planner, check_config, calibration_gate, contact_candidate, detect_objects, execute, ROOT
from placement_success import cube_overlaps_tray


def config():
    c = json.loads((ROOT/'pick_place_config.json').read_text())
    c.update(T_gripper_tcp=np.eye(4).tolist(), table_z_m=-1, workspace_min_m=[-1, -1, -1], workspace_max_m=[1, 1, 1],
             gripper_open=70, gripper_closed=15, tcp_lowest_extent_mm=5)
    c['cube'].update(side_mm=30, grasp_depth_mm=15)
    c['plate'].update(rim_height_mm=15)
    return c


def vertical_tool(fk, q):
    """Tool frame whose +Z points straight down at joints q, so a vertical grasp is reachable."""
    tool = np.eye(4)
    tool[:3, :3] = fk(dict(zip(fk.names, q)))[:3, :3].T @ np.diag([1., -1., -1.])
    return tool


class PickTests(unittest.TestCase):
    def test_ik_recovery_and_unreachable(self):
        planner = Planner(URDFFK(ROOT/'so101_new_calib.urdf'), config())
        q = np.array([10., -25, 30, 15, -20])
        target = planner.tcp(q)
        answer = planner.ik(target, q + .5)
        np.testing.assert_allclose(planner.tcp(answer)[:3, 3], target[:3, 3], atol=.002)
        target[:3, 3] = [4, 4, 4]
        with self.assertRaises(ValueError):
            planner.ik(target, q)

    def test_gates(self):
        with self.assertRaises(ValueError):
            check_config(dict(config(), table_z_m=None))
        check_config(config())
        with self.assertRaises(ValueError):
            calibration_gate({'board_position_rms_mm': 31.68, 'board_rotation_rms_deg': 5.8})
        calibration_gate({'board_position_rms_mm': 2, 'board_rotation_rms_deg': 1})

    def test_full_sequence(self):
        c = config()
        fk = URDFFK(ROOT/'so101_new_calib.urdf')
        q = np.array([0., -30, 50, -20, 0])
        c['T_gripper_tcp'] = vertical_tool(fk, q).tolist()
        c['cube'].update(side_mm=10, grasp_depth_mm=5)
        c.update(tcp_lowest_extent_mm=2)
        c['plate'].update(rim_height_mm=5)
        planner = Planner(fk, c)
        B = np.eye(4)
        B[:3, 3] = planner.tcp(q)[:3, 3] + [0, 0, -.1]
        P = B.copy()
        P[:3, 3] += [-.06, 0, -.01]
        obs = {'joints_deg': dict(zip(fk.names, q)), 'targets': {
            'cube': {'T_base_object': B.tolist()}, 'plate': {'T_base_object': P.tolist()}}}
        plan = planner.plan(obs)
        self.assertEqual([s['stage'] for s in plan], ['align', 'approach', 'hover', 'descend', 'grasp', 'lift', 'transfer', 'place', 'release', 'retract'])
        self.assertEqual(plan[4]['gripper'], c['gripper_closed'])
        self.assertEqual(plan[8]['gripper'], c['gripper_open'])

    def test_pick_only_plan(self):
        c = config()
        c.pop('plate')
        c['pick_only'] = True
        check_config(c)
        fk = URDFFK(ROOT/'so101_new_calib.urdf')
        q = np.array([0., -30, 50, -20, 0])
        c['T_gripper_tcp'] = vertical_tool(fk, q).tolist()
        c['cube'].update(side_mm=10, grasp_depth_mm=5)
        c.update(tcp_lowest_extent_mm=2)
        planner = Planner(fk, c)
        B = np.eye(4)
        B[:3, 3] = planner.tcp(q)[:3, 3] + [0, 0, -.1]
        plan = planner.plan({'joints_deg': dict(zip(fk.names, q)), 'targets': {'cube': {'T_base_object': B.tolist()}}})
        self.assertEqual([s['stage'] for s in plan], ['align', 'approach', 'hover', 'descend', 'grasp', 'lift'])

    def test_workspace_and_joint_checks(self):
        c = config()
        planner = Planner(URDFFK(ROOT/'so101_new_calib.urdf'), c)
        planner.clearance(np.zeros(5))
        with self.assertRaises(ValueError):
            planner.clearance(np.ones(5)*1000)
        c['workspace_max_m'] = [-.9, -.9, -.9]
        with self.assertRaises(ValueError):
            planner.clearance(np.zeros(5))

    def run_pick(self, drop_cube=False, cube_error_mm=None, views_mm=None):
        """Servos reach every goal instantly; the gripper stalls at 30 on the cube.
        drop_cube: the cube slips out once the lift starts, so the gripper is free to close.
        cube_error_mm: the real cube is this far (base xy) from the first observation; the camera sees it exactly.
        views_mm: instead, what each refine measurement (10 frames) reports, relative to the first observation."""
        c = config()
        c.pop('plate')
        c.update(pick_only=True, auto=True, refine_at_approach=cube_error_mm is not None or views_mm is not None,
                 tcp_lowest_extent_mm=2)
        fk = URDFFK(ROOT/'so101_new_calib.urdf')
        q = np.array([0., -30, 50, -20, 0])
        c['T_gripper_tcp'] = vertical_tool(fk, q).tolist()
        c['cube'].update(side_mm=10, grasp_depth_mm=5)
        planner = Planner(fk, c)
        B = np.eye(4)
        B[:3, 3] = planner.tcp(q)[:3, 3] + [0, 0, -.1]
        obs = {'joints_deg': dict(zip(fk.names, q)), 'targets': {'cube': {'T_base_object': B.tolist()}}}
        plan = planner.plan(obs)

        class Bus:
            pos = dict(zip(fk.names, q)) | {'gripper': 40.}
            def sync_read(self, _):
                return dict(self.pos)
            def sync_write(self, _, goal):  # arm motion: the cube falls out here when drop_cube
                self.pos.update({k: v if drop_cube and k == 'gripper' else max(v, 30.) if k == 'gripper' else v
                                 for k, v in goal.items()})
            def write(self, _, name, value):  # gripper-only closing: blocked by the cube
                self.pos[name] = max(value, 30.)
            def enable_torque(self, **_):
                pass
        robot = SimpleNamespace(bus=Bus(), configure=lambda: None)
        cap = SimpleNamespace(read=lambda: (True, np.zeros((4, 4, 3), np.uint8)))
        real = B.copy()
        if cube_error_mm is not None:
            real[:2, 3] += np.array(cube_error_mm) / 1000
        calls = []
        def detect(frame, q):
            seen = real.copy()
            if views_mm is not None:
                seen[:2, 3] += np.array(views_mm[min(len(calls) // 10, len(views_mm) - 1)]) / 1000
            calls.append(1)
            return {'cube': {'T_base_object': seen.tolist()}}, frame
        scene = SimpleNamespace(detect=detect)
        with tempfile.TemporaryDirectory() as out, mock.patch('cv2.imshow'), mock.patch('cv2.waitKey', return_value=0), \
                mock.patch('time.sleep'):
            try:
                execute(robot, planner, plan, obs, cap, Path(out), scene)
            finally:
                log = json.loads((Path(out)/'execution.json').read_text())
        return fk, plan, robot, log, real

    def test_execute_pick_reports_picked_up(self):
        fk, plan, robot, log, _ = self.run_pick()
        self.assertEqual(log['status'], 'picked_up_holding')
        self.assertEqual([e['stage'] for e in log['events']][-3:], ['contact', 'lift', 'picked_up'])
        np.testing.assert_allclose([robot.bus.pos[n] for n in fk.names], plan[-1]['joints_deg'][-1], atol=1e-6)
        self.assertEqual(robot.bus.pos['gripper'], 30)  # squeezing, but the cube keeps the fingers at contact

    def test_execute_pick_detects_dropped_cube(self):
        with self.assertRaisesRegex(RuntimeError, 'Cube not held after lift'):
            self.run_pick(drop_cube=True)

    def test_refine_moves_over_the_real_cube(self):
        fk, plan, robot, log, real = self.run_pick(cube_error_mm=[5, -4])
        stages = [e['stage'] for e in log['events']]
        self.assertEqual(stages.count('refine'), 1)
        self.assertEqual(stages[-1], 'picked_up')
        descend = next(s for s in plan if s['stage'] == 'descend')
        # pick_offset_mm has no xy part, so the corrected grasp sits right over the real cube
        np.testing.assert_allclose(np.array(descend['T_base_tcp'])[:2, 3], real[:2, 3], atol=1e-6)
        np.testing.assert_allclose([robot.bus.pos[n] for n in fk.names], plan[-1]['joints_deg'][-1], atol=1e-6)

    def test_refine_corrects_once(self):
        """Later views (here: 20 mm the other way) must not move the grasp again."""
        _, plan, _, log, real = self.run_pick(views_mm=[[9, 0], [-11, 0]])
        self.assertEqual([e['stage'] for e in log['events']].count('refine'), 1)
        descend = next(s for s in plan if s['stage'] == 'descend')
        np.testing.assert_allclose(np.array(descend['T_base_tcp'])[:2, 3], real[:2, 3] + [.009, 0], atol=1e-6)

    def test_purple_cube_and_tray_placement(self):
        """Synthetic top view: blue tray + purple cube -> both detected; inside = placed, outside = not."""
        c = json.loads((ROOT/'pick_place_config.json').read_text())
        c['cube']['color'] = 'purple'
        def scene(cube_x):
            img = np.full((400, 600, 3), 255, np.uint8)
            cv2.rectangle(img, (100, 100), (300, 300), (200, 80, 0), -1)  # blue tray
            cv2.rectangle(img, (cube_x, 170), (cube_x + 60, 230), (200, 0, 150), -1)  # purple cube top
            return detect_objects(img, c)
        found, _, debug = scene(170)
        self.assertEqual(set(found), {'cube', 'plate'})
        np.testing.assert_allclose(found['cube']['center_px'], [200, 200], atol=2)
        np.testing.assert_allclose(found['plate']['center_px'], [200, 200], atol=2)
        self.assertTrue(cube_overlaps_tray(debug['cube']['mask'], debug['plate']['mask']))
        _, _, debug = scene(420)
        self.assertFalse(cube_overlaps_tray(debug['cube']['mask'], debug['plate']['mask']))
        with self.assertRaises(ValueError):
            check_config(dict(config(), cube=dict(config()['cube'], color='pink')))

    def test_stable_closed_is_not_contact(self):
        self.assertFalse(contact_candidate([0.] * 10, 0, 0))
        self.assertFalse(contact_candidate([30.] * 10, 30, 0))
        self.assertFalse(contact_candidate(list(range(10)), 0, 0))
        self.assertTrue(contact_candidate([30.] * 10, 26, 0))


if __name__ == '__main__':
    unittest.main()
