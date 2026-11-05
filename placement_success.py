"""Image-mask checks for verifying that a cube was placed in a tray."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from camera_intrinsic import open_camera
from detect_cube import detect_cube_top
from detect_tray import detect_tray_center


def cube_overlaps_tray(cube_mask, tray_mask, min_cube_coverage=0.8):
    """Return whether most detected cube pixels lie inside the tray mask.

    The cube and tray use different colors, so their raw HSV masks normally
    do not share pixels. The filled convex hull of the tray mask represents
    its visible footprint and is used as the placement region.
    """
    cube = np.asarray(cube_mask)
    tray = np.asarray(tray_mask)
    if cube.ndim != 2 or tray.ndim != 2 or cube.shape != tray.shape:
        raise ValueError('Cube and tray masks must be 2D arrays with the same shape')
    if not 0 < min_cube_coverage <= 1:
        raise ValueError('min_cube_coverage must be in (0, 1]')

    cube_pixels = cube > 0
    if not np.any(cube_pixels) or not np.any(tray > 0):
        return False

    tray_contours, _ = cv2.findContours(
        (tray.astype(np.uint8) > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not tray_contours:
        return False
    tray_hull = cv2.convexHull(np.concatenate(tray_contours))
    tray_region = np.zeros(tray.shape, np.uint8)
    cv2.fillConvexPoly(tray_region, tray_hull, 255)
    coverage = np.mean(tray_region[cube_pixels] > 0)
    return bool(coverage >= min_cube_coverage)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Test cube placement from a live camera stream')
    parser.add_argument('--config', type=Path, default=root / 'pick_place_config.json')
    parser.add_argument('--camera', default=None, help='Camera index or device path; defaults to config')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    cube_config = config['cube']
    tray_config = config['plate']
    camera = config['camera'] if args.camera is None else args.camera
    cap = open_camera(camera, args.width, args.height)
    last_status = None

    def size(cfg):
        return dict(min_area=cfg['min_area_px'], canny_low=cfg['canny_low'], canny_high=cfg['canny_high'])

    try:
        print('Placement test started. Press Q or ESC to quit.')
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Camera read failed')

            _, cube_debug = detect_cube_top(frame, color=cube_config['color'], **size(cube_config))
            _, tray_debug = detect_tray_center(frame, **size(tray_config))
            success = cube_overlaps_tray(cube_debug['mask'], tray_debug['mask'])
            if success != last_status:
                print('SUCCESS' if success else 'FAILED', flush=True)
                last_status = success

            status = 'SUCCESS' if success else 'FAILED'
            status_color = (0, 200, 0) if success else (0, 0, 255)
            cv2.putText(frame, status, (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, status_color, 3, cv2.LINE_AA)
            cv2.imshow('Placement test', frame)
            if cv2.waitKey(1) & 255 in (27, ord('q')):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
