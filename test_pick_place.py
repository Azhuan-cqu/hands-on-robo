import json
import unittest
import cv2
import numpy as np
from handeye_calibration import transform, URDFFK
from pick_place import marker_pose, pivot_tcp, Planner, check_config, calibration_gate, ROOT


class PickTests(unittest.TestCase):
    def test_marker_pnp_direction_and_scale(self):
        K = np.array([[900., 0, 640], [0, 900, 360], [0, 0, 1]])
        obj = np.array([[-.01, .01, 0], [.01, .01, 0], [.01, -.01, 0], [-.01, -.01, 0]])
        r, t = np.array([2.7, .2, .1]), np.array([.03, -.04, .32])
        corners = cv2.projectPoints(obj, r, t, K, np.zeros(5))[0]
        error, T = marker_pose(corners, .02, K, np.zeros(5))
        self.assertLess(error, 1e-6)
        np.testing.assert_allclose(T, transform(cv2.Rodrigues(r)[0], t), atol=1e-7)

    def test_pivot_and_degeneracy(self):
        rng = np.random.default_rng(3)
        tool, fixed = np.array([.01, -.08, .02]), np.array([.2, .05, .03])
        poses = []
        for _ in range(10):
            R = cv2.Rodrigues(rng.normal(size=3))[0]
            poses.append(transform(R, fixed - R @ tool))
        t, rms = pivot_tcp(poses)
        np.testing.assert_allclose(t, tool, atol=1e-10)
        self.assertLess(rms, 1e-10)
        with self.assertRaises(ValueError):
            pivot_tcp([poses[0]]*8)

    def config(self):
        c = json.loads((ROOT/'pick_place_config.json').read_text())
        c.update(detector='aruco', T_gripper_tcp=np.eye(4).tolist(), table_z_m=-1,
                 workspace_min_m=[-1, -1, -1], workspace_max_m=[1, 1, 1], gripper_open=70,
                 gripper_closed=15, tool_radius_mm=15, tcp_lowest_extent_mm=5)
        c['cube'].update(id=1, marker_mm=18, side_mm=30, grasp_depth_mm=15)
        c['plate'].update(id=2, marker_mm=18, rim_height_mm=15, inner_half_size_mm=[70, 70], outer_half_size_mm=[80, 80])
        return c

    def test_ik_recovery_and_unreachable(self):
        planner = Planner(URDFFK(ROOT/'so101_new_calib.urdf'), self.config())
        q = np.array([10., -25, 30, 15, -20])
        target = planner.tcp(q)
        answer = planner.ik(target, q + .5)
        np.testing.assert_allclose(planner.tcp(answer)[:3, 3], target[:3, 3], atol=.002)
        target[:3, 3] = [4, 4, 4]
        with self.assertRaises(ValueError):
            planner.ik(target, q)

    def test_gates(self):
        with self.assertRaises(ValueError):
            check_config(dict(json.loads((ROOT/'pick_place_config.json').read_text()), detector='aruco', table_z_m=None))
        check_config(self.config())
        with self.assertRaises(ValueError):
            calibration_gate({'board_position_rms_mm':31.68, 'board_rotation_rms_deg':5.8})
        calibration_gate({'board_position_rms_mm':2, 'board_rotation_rms_deg':1})

    def test_full_sequence_and_plate_rim(self):
        c = self.config()
        fk = URDFFK(ROOT/'so101_new_calib.urdf')
        q = np.array([0., -30, 50, -20, 0])
        G = fk(dict(zip(fk.names, q)))
        tool = np.eye(4)
        tool[:3, :3] = G[:3, :3].T @ np.diag([1., -1., -1.])
        c['T_gripper_tcp'] = tool.tolist()
        c['cube'].update(side_mm=10, grasp_depth_mm=5)
        c.update(tool_radius_mm=5, tcp_lowest_extent_mm=2)
        c['plate'].update(inner_half_size_mm=[30, 30], outer_half_size_mm=[35, 35], rim_height_mm=5)
        planner = Planner(fk, c)
        B = np.eye(4)
        B[:3, 3] = planner.tcp(q)[:3, 3] + [0, 0, -.1]
        P = B.copy()
        P[:3, 3] += [-.06, 0, -.01]
        obs = {'joints_deg': dict(zip(fk.names, q)), 'targets': {
            'cube': {'T_base_marker': B.tolist()}, 'plate': {'T_base_marker': P.tolist()}}}
        plan = planner.plan(obs)
        self.assertEqual([s['stage'] for s in plan], ['align', 'approach', 'hover', 'descend', 'grasp', 'lift', 'transfer', 'place', 'release', 'retract'])
        self.assertEqual(plan[4]['gripper'], c['gripper_closed'])
        self.assertEqual(plan[8]['gripper'], c['gripper_open'])
        # Put a rim directly under the current TCP envelope.
        planner.plate = np.eye(4)
        planner.plate[:3, 3] = planner.tcp(q)[:3, 3] + [-.03, 0, -.006]
        with self.assertRaisesRegex(ValueError, 'plate rim'):
            planner.clearance(q)

    def test_workspace_and_joint_checks(self):
        c = self.config()
        planner = Planner(URDFFK(ROOT/'so101_new_calib.urdf'), c)
        planner.clearance(np.zeros(5))
        with self.assertRaises(ValueError):
            planner.clearance(np.ones(5)*1000)
        c['workspace_max_m'] = [-.9, -.9, -.9]
        with self.assertRaises(ValueError):
            planner.clearance(np.zeros(5))



class PickOnlyTests(PickTests):
    def test_auto_id_locks_target(self):
        from pick_place import select_marker
        self.assertEqual(select_marker(np.array([[7]])), 7)
        with self.assertRaises(ValueError):
            select_marker(np.array([[7], [8]]))
        with self.assertRaises(ValueError):
            select_marker(np.array([[8]]), selected=7)

    def test_stable_closed_is_not_contact(self):
        from pick_place import contact_candidate
        self.assertFalse(contact_candidate([0.] * 10, 0, 0))
        self.assertFalse(contact_candidate([30.] * 10, 30, 0))
        self.assertFalse(contact_candidate(list(range(10)), 0, 0))
        self.assertTrue(contact_candidate([30.] * 10, 26, 0))

    def test_no_plate_config_or_plan(self):
        c = self.config()
        c.pop('plate')
        c['pick_only'] = True
        c['cube']['id'] = None
        check_config(c)
        fk = URDFFK(ROOT/'so101_new_calib.urdf')
        q = np.array([0., -30, 50, -20, 0])
        G = fk(dict(zip(fk.names, q)))
        tool = np.eye(4)
        tool[:3, :3] = G[:3, :3].T @ np.diag([1., -1., -1.])
        c['T_gripper_tcp'] = tool.tolist()
        c['cube'].update(side_mm=10, grasp_depth_mm=5)
        c.update(tool_radius_mm=5, tcp_lowest_extent_mm=2)
        planner = Planner(fk, c)
        B = np.eye(4)
        B[:3, 3] = planner.tcp(q)[:3, 3] + [0, 0, -.1]
        obs = {'joints_deg': dict(zip(fk.names, q)), 'targets': {'cube': {'T_base_marker': B.tolist()}}}
        plan = planner.plan(obs)
        self.assertEqual([s['stage'] for s in plan], ['align', 'approach', 'hover', 'descend', 'grasp', 'lift'])
        self.assertIsNone(planner.plate)

if __name__ == '__main__':
    unittest.main()
