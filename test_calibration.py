import argparse
import unittest
import cv2
import numpy as np
from camera_intrinsic import make_board, detect
from handeye_calibration import URDFFK, transform, solve


class CalibrationTests(unittest.TestCase):
    def test_board_detection(self):
        b = make_board(argparse.Namespace(squares_x=7, squares_y=5, square_mm=25,
                marker_mm=18, dictionary='DICT_4X4_50', legacy_pattern=False))
        frame = cv2.cvtColor(b.generateImage((1000, 750), marginSize=30), cv2.COLOR_GRAY2BGR)
        d = detect(frame, b, cv2.aruco.CharucoDetector(b))
        self.assertIsNotNone(d)
        self.assertEqual(len(d[2]), 24)
        np.testing.assert_allclose(np.max(d[0].reshape(-1, 3), axis=0), [.15, .10, 0], atol=1e-6)

    def test_intrinsic_recovery(self):
        rng = np.random.default_rng(4)
        K = np.array([[900., 0, 640], [0, 910, 360], [0, 0, 1]])
        obj = np.array([[x*.025, y*.025, 0] for x in range(1, 7) for y in range(1, 5)], np.float32)
        images = []
        for _ in range(25):
            r = rng.uniform(-.6, .6, 3)
            t = np.array([rng.uniform(-.16, .05), rng.uniform(-.12, .04), rng.uniform(.35, .65)])
            images.append(cv2.projectPoints(obj, r, t, K, np.zeros(5))[0])
        rms, recovered, _, _, _ = cv2.calibrateCamera([obj]*25, images, (1280, 720), None, None)
        self.assertLess(rms, .001)
        np.testing.assert_allclose(recovered, K, atol=.01)

    def test_handeye_with_real_urdf_chain(self):
        fk = URDFFK('so101_new_calib.urdf')
        self.assertEqual(fk.names, ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll'])
        rng = np.random.default_rng(12)
        X = transform(cv2.Rodrigues(np.array([.2, -.15, .3]))[0], [.03, -.02, .05])
        B = transform(cv2.Rodrigues(np.array([.3, .1, -.2]))[0], [.25, .1, .15])
        samples = []
        for _ in range(25):
            G = fk(dict(zip(fk.names, rng.uniform(-60, 60, 5))))
            C = np.linalg.inv(X) @ np.linalg.inv(G) @ B
            samples.append({'T_base_gripper': G.tolist(), 'T_camera_board': C.tolist()})
        result = solve(samples)
        np.testing.assert_allclose(result['T_gripper_camera'], X, atol=1e-8)
        self.assertLess(result['board_position_rms_mm'], 1e-6)
        with self.assertRaises(ValueError):
            solve([samples[0]] * 15)


if __name__ == '__main__':
    unittest.main()
