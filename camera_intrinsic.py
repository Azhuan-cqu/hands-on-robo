#!/usr/bin/env python3
"""ChArUco intrinsics; shared board/camera helpers for handeye_calibration.py."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def board_args(p):
    p.add_argument('--squares-x', type=int, default=7)
    p.add_argument('--squares-y', type=int, default=5)
    p.add_argument('--square-mm', type=float, default=25)
    p.add_argument('--marker-mm', type=float, default=18)
    p.add_argument('--dictionary', default='DICT_4X4_50')
    p.add_argument('--legacy-pattern', action='store_true')


def board_config(a):
    return {k: getattr(a, k) for k in ('squares_x', 'squares_y', 'square_mm', 'marker_mm', 'dictionary', 'legacy_pattern')}


def make_board(a):
    if not 0 < a.marker_mm < a.square_mm or min(a.squares_x, a.squares_y) < 3:
        raise ValueError('Invalid board geometry')
    code = getattr(cv2.aruco, a.dictionary, None)
    if code is None or not a.dictionary.startswith('DICT_'):
        raise ValueError('Unknown ArUco dictionary: ' + a.dictionary)
    b = cv2.aruco.CharucoBoard((a.squares_x, a.squares_y), a.square_mm / 1000, a.marker_mm / 1000,
                             cv2.aruco.getPredefinedDictionary(code))
    b.setLegacyPattern(a.legacy_pattern)
    return b


def detect(frame, board, detector):
    corners, ids, _, _ = detector.detectBoard(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is None or len(ids) < 6 or board.checkCharucoCornersCollinear(ids):
        return None
    obj, img = board.matchImagePoints(corners, ids)
    return obj.astype(np.float32), img.astype(np.float32), ids


def load_camera(path):
    """camera.yml -> K, D, (width, height), board config dict."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise ValueError(f'Cannot open {path}; calibrate intrinsics first')
    K, D = fs.getNode('K').mat(), fs.getNode('D').mat()
    size = int(fs.getNode('image_width').real()), int(fs.getNode('image_height').real())
    board = json.loads(fs.getNode('board_json').string())
    fs.release()
    return K, D, size, board


def open_camera(source, width, height):
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera {source}')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Some cameras ignore this; lock focus manually.
    return cap


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def reprojection(obj, img, r, t, K, D):
    projected, _ = cv2.projectPoints(obj, r, t, K, D)
    return float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) - img.reshape(-1, 2)) ** 2, axis=1))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    board_args(p)
    p.add_argument('--camera', default='0')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--images', type=Path, help='Offline directory of PNG/JPG images')
    p.add_argument('--output', type=Path, default=Path('intrinsic_output'))
    p.add_argument('--min-views', type=int, default=15)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    b = make_board(a)
    detector = cv2.aruco.CharucoDetector(b)
    samples, objects, pixels = [], [], []
    size = None

    def add(frame, name):
        nonlocal size
        detected = detect(frame, b, detector)
        current = (frame.shape[1], frame.shape[0])
        if size is not None and size != current:
            raise ValueError('All images must have the same resolution')
        if detected is None:
            print('Rejected: fewer than 6 non-collinear ChArUco corners')
            return
        obj, img, ids = detected
        size = current
        objects.append(obj)
        pixels.append(img)
        samples.append({'image': name, 'ids': ids.ravel().tolist(), 'corners': img.reshape(-1, 2).tolist()})
        print(f'Saved view {len(samples)}: {len(ids)} corners')

    if a.images:
        for path in sorted(a.images.iterdir()):
            if path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                frame = cv2.imread(str(path))
                if frame is None:
                    raise ValueError(f'Unreadable image: {path}')
                add(frame, str(path.resolve()))
    else:
        raw = a.output / 'images'
        raw.mkdir(exist_ok=True)
        if list(raw.glob('*.png')):
            raise ValueError('Output already contains images. Use --images to solve or choose a new --output.')
        cap = open_camera(a.camera, a.width, a.height)
        print('SPACE: save; C: calibrate; Q/ESC: exit without solving. Collect 20–30 diverse views.')
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Camera read failed')
                view = frame.copy()
                d = detect(frame, b, detector)
                if d:
                    cv2.aruco.drawDetectedCornersCharuco(view, d[1], d[2])
                cv2.putText(view, f'views={len(samples)} SPACE save | C solve | Q quit', (15, 30), 0, .65, (0, 255, 0), 2)
                cv2.imshow('Intrinsics', view)
                key = cv2.waitKey(1) & 255
                if key in (27, ord('q')):
                    return
                if key == ord('c'):
                    break
                if key == 32 and d:
                    path = raw / f'{len(samples):04d}.png'
                    if not cv2.imwrite(str(path), frame):
                        raise IOError(f'Cannot save {path}')
                    add(frame, str(path.resolve()))
        finally:
            cap.release()
            cv2.destroyAllWindows()
    if len(samples) < max(5, a.min_views):
        raise ValueError(f'Need at least {max(5, a.min_views)} usable views; got {len(samples)}')
    rms, K, D, rs, ts = cv2.calibrateCamera(objects, pixels, size, None, None)
    errors = [reprojection(o, i, r, t, K, D) for o, i, r, t in zip(objects, pixels, rs, ts)]
    if not np.isfinite(K).all() or not np.isfinite(D).all():
        raise ValueError('Non-finite calibration')
    fs = cv2.FileStorage(str(a.output / 'camera.yml'), cv2.FILE_STORAGE_WRITE)
    fs.write('K', K)
    fs.write('D', D)
    fs.write('image_width', size[0])
    fs.write('image_height', size[1])
    fs.write('rms_px', rms)
    fs.write('board_json', json.dumps(board_config(a)))
    fs.release()
    save_json(a.output / 'report.json', {'rms_px': rms, 'views': [dict(s, rms_px=e) for s, e in zip(samples, errors)]})
    print(f'RMS={rms:.4f} px; per-view errors: {np.round(errors, 3)}')
    print(f'Saved {a.output / "camera.yml"}; inspect report.json and validate using new views.')


if __name__ == '__main__':
    main()
