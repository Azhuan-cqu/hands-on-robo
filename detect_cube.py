#!/usr/bin/env python3
"""Colored cube preview; image coordinates only, no robot connection.

With pose context, selects an upright cube top using square PnP, table normal
and height. Without calibration, only a 2D preview hypothesis is returned.
Occlusion or weak face boundaries can cause failure.
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np


# Calibrated to IMG_4468.HEIC, bottom-to-top labels supplied by the user.
# OpenCV H is 0..179; S/V are 0..255. Gaps intentionally reject ambiguous hues.
# "cyan" is the fifth, yellow-green block in THIS set, not generic cyan.
COLOR_HSV_RANGES = {
    "red": [
        ((170, 100, 80), (179, 255, 255)),
    ],
    "orange": [((0, 100, 100), (19, 255, 255))],
    "yellow": [((20, 100, 100), (50, 255, 255))],
    "green": [((34, 70, 70), (84, 255, 255))],
    "cyan": [((34, 70, 70), (100, 255, 255))],
    "blue": [((100, 80, 70), (129, 255, 255))],
    "purple": [((120, 70, 50), (169, 255, 255))],
}


def diagonal_center(quad):
    """Perspective-correct image of a square's center, using cyclic corners."""
    a, b, c, d = np.asarray(quad, dtype=float)
    system = np.column_stack((c - a, -(d - b)))
    if abs(np.linalg.det(system)) < 1e-6:
        return None
    t, u = np.linalg.solve(system, b - a)
    return a + t * (c - a) if 0 < t < 1 and 0 < u < 1 else None


def line_quadrilaterals(edges, object_size):
    """Bridge broken boundaries with intersections of supported Hough lines.

    A line intersection only proposes a corner; the caller validates all four
    full sides against original Canny edges and checks orange coverage.
    """
    minimum = max(12, int(object_size * .25))
    segments = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(12, minimum // 2),
                               minLineLength=minimum, maxLineGap=max(4, int(object_size * .1)))
    if segments is None:
        return []
    segments = sorted(segments.reshape(-1, 4),
                      key=lambda s: -np.linalg.norm(s[2:] - s[:2]))
    lines = []
    for segment in segments:
        a, b = segment.reshape(2, 2).astype(float)
        direction = (b - a) / np.linalg.norm(b - a)
        normal = np.array([-direction[1], direction[0]])
        offset = normal @ a
        if any(abs(normal @ n) > .995 and abs(n @ a - d) < 3
               for n, d in lines):
            continue
        lines.append((normal, offset))
        if len(lines) == 12:
            break
    height, width = edges.shape
    quads = []
    for group in combinations(lines, 4):
        for order in ((0, 1, 2, 3), (0, 1, 3, 2), (0, 2, 1, 3)):
            corners = []
            for i in range(4):
                n1, d1 = group[order[i]]
                n2, d2 = group[order[(i + 1) % 4]]
                system = np.array([n1, n2])
                if abs(np.linalg.det(system)) < .4:
                    break
                point = np.linalg.solve(system, [d1, d2])
                if not (0 <= point[0] < width and 0 <= point[1] < height):
                    break
                corners.append(point)
            if len(corners) == 4:
                quad = np.rint(corners).astype(np.int32).reshape(4, 1, 2)
                if cv2.isContourConvex(quad):
                    quads.append(quad)
    return quads


def select_cube_face(candidates, pose_context=None):
    """Select a face without image-y ordering.

    Context: K, D, T_base_camera, side_m, top_z_m (all lengths in meters).
    Height is a soft prior by default: set enforce_height=True for a hard gate.
    Assumes an upright cube on a horizontal base-frame table. Without context,
    returns only a 2D preview hypothesis, never a verified physical top face.
    """
    if not candidates:
        return None
    if pose_context is None:
        ranked = sorted(candidates, key=lambda c: cv2.contourArea(
            c['top_corners_px'].astype(np.float32)) * c['edge_support'] ** 4, reverse=True)
        return dict(ranked[0], selection_mode='preview_2d')
    ctx = pose_context
    K = np.asarray(ctx['K'], np.float64)
    D = np.asarray(ctx.get('D', np.zeros(5)), np.float64)
    T = np.asarray(ctx['T_base_camera'], np.float64)
    side = float(ctx['side_m'])
    if (K.shape != (3, 3) or T.shape != (4, 4) or side <= 0
            or not np.isfinite(K).all() or not np.isfinite(T).all()
            or not np.isfinite(side) or not np.isfinite(ctx['top_z_m'])):
        raise ValueError('Invalid cube pose calibration/context')
    half = side / 2
    model = np.array([[-half, -half, 0], [half, -half, 0],
                      [half, half, 0], [-half, half, 0]], np.float64)
    angle_limit = float(ctx.get('max_tilt_deg', 25))
    error_limit = float(ctx.get('max_reprojection_px', 3))
    height_limit = float(ctx.get('height_tolerance_m', max(.005, side * .25)))
    if min(angle_limit, error_limit, height_limit) <= 0:
        raise ValueError('Cube pose tolerances must be positive')
    ranked = []
    for candidate in candidates:
        pixels = np.asarray(candidate['top_corners_px'], np.float64)
        try:
            solved = cv2.solvePnPGeneric(model, pixels, K, D, flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            continue
        if not solved[0]:
            continue
        hypotheses = []
        for rvec, tvec in zip(solved[1], solved[2]):
            R = cv2.Rodrigues(rvec)[0]
            camera_points = model @ R.T + tvec.reshape(3)
            if not np.isfinite(camera_points).all() or np.any(camera_points[:, 2] <= 0):
                continue
            projected = cv2.projectPoints(model, rvec, tvec, K, D)[0].reshape(4, 2)
            error = float(np.sqrt(np.mean(np.sum((projected - pixels) ** 2, axis=1))))
            normal = T[:3, :3] @ R[:, 2]
            # A cyclic contour can have either winding; plane normal sign is arbitrary.
            tilt = float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0, 1))))
            base_points = camera_points @ T[:3, :3].T + T[:3, 3]
            height_error = float(np.max(np.abs(base_points[:, 2] - ctx['top_z_m'])))
            height_consistent = height_error <= height_limit
            if (error > error_limit or tilt > angle_limit
                    or (ctx.get('enforce_height', False) and not height_consistent)):
                continue
            # Project the physical square center, including lens distortion.
            center = cv2.projectPoints(np.zeros((1, 3)), rvec, tvec, K, D)[0].reshape(2)
            # Absolute PnP height depends on size, hand-eye translation and
            # table Z. It must not veto an otherwise horizontal face by default.
            # A bounded prior cannot overwhelm the normal/reprojection evidence.
            score = error / error_limit + tilt / angle_limit + .25 * min(height_error / height_limit, 1.)
            hypotheses.append((score, dict(candidate, center_px=center,
                selection_mode='pose_3d', normal_base=normal if normal[2] >= 0 else -normal,
                reprojection_error_px=error, tilt_deg=tilt, height_error_m=height_error,
                height_consistent=height_consistent,
                height_tolerance_m=height_limit,
                estimated_top_z_m=float(np.mean(base_points[:, 2])),
                expected_top_z_m=float(ctx['top_z_m']))))
        if hypotheses:
            ranked.append(min(hypotheses, key=lambda item: item[0]))
    ranked.sort(key=lambda item: item[0])
    if not ranked:
        return None
    # Do not guess between spatially distinct faces with essentially equal fits.
    if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < .15:
        a, b = ranked[0][1], ranked[1][1]
        if np.linalg.norm(a['center_px'] - b['center_px']) > 8:
            return None
    return ranked[0][1]


def detect_cube_top(frame, hsv_low=(9, 170, 60), hsv_high=(19, 255, 255),
                min_area=400, canny_low=15, canny_high=40, *, color=None, pose_context=None):
    """Detect without drawing, GUI, camera access, or modifying the BGR input.

    Returns (result, debug). result is None when no unambiguous candidate exists,
    otherwise a dict with center_px (float array [u,v]), top_corners_px (4x2
    cyclic corner array), and edge_support (minimum supported edge fraction,
    not a probability). Coordinates are pixels in the original image, NOT 3D.
    debug contains mask, edges and cube_contour for optional visualization.

    color selects a COLOR_HSV_RANGES preset and overrides hsv_low/hsv_high.
    With color=None, custom HSV bounds apply (orange-only by default).
    Uses the largest matching color component. With pose_context, selects using
    square PnP, base-frame normal and expected top height. Without calibration,
    returns an explicitly unverified 2D preview hypothesis (no image-y ranking).
    """
    # Normalize large still photos: fixed pixel-scale Canny/Hough parameters
    # otherwise see surface texture as edges and lose the actual face borders.
    # Always return geometry/debug images in the caller's original coordinates.
    height, width = frame.shape[:2]
    if max(height, width) > 1280:
        factor = 1280 / max(height, width)
        small = cv2.resize(frame, (round(width * factor), round(height * factor)),
                           interpolation=cv2.INTER_AREA)
        small_context = None
        if pose_context is not None:
            small_context = dict(pose_context)
            resize_scale = np.diag([small.shape[1] / width, small.shape[0] / height, 1.])
            small_context['K'] = resize_scale @ np.asarray(pose_context['K'])
            small_context['max_reprojection_px'] = pose_context.get('max_reprojection_px', 3) * factor
        result, debug = detect_cube_top(small, hsv_low, hsv_high,
                                       min_area * factor ** 2, canny_low, canny_high,
                                       color=color, pose_context=small_context)
        scale = np.array([width / small.shape[1], height / small.shape[0]])
        if result is not None:
            result = dict(result, center_px=result['center_px'] * scale,
                          top_corners_px=np.rint(result['top_corners_px'] * scale).astype(np.int32))
        if result is not None and 'reprojection_error_px' in result:
            result['reprojection_error_px'] /= factor
        contour = debug['cube_contour']
        return result, {
            'mask': cv2.resize(debug['mask'], (width, height), interpolation=cv2.INTER_NEAREST),
            'edges': cv2.resize(debug['edges'], (width, height), interpolation=cv2.INTER_NEAREST),
            'cube_contour': None if contour is None else np.rint(contour * scale).astype(np.int32),
        }
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if color is not None and color not in COLOR_HSV_RANGES:
        raise ValueError(f'Unknown cube color: {color!r}; choose from {tuple(COLOR_HSV_RANGES)}')
    ranges = COLOR_HSV_RANGES[color] if color is not None else [(hsv_low, hsv_high)]
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for low, high in ranges:
        mask |= cv2.inRange(hsv, np.array(low, np.uint8), np.array(high, np.uint8))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    edges = np.zeros(mask.shape, np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    result = None
    blob = None
    if contours:
        blob = max(contours, key=cv2.contourArea)
        object_mask = np.zeros_like(mask)
        cv2.drawContours(object_mask, [blob], -1, 255, cv2.FILLED)
        x, y, w, h = cv2.boundingRect(blob)
        x0, y0 = max(0, x - 8), max(0, y - 8)
        x1, y1 = min(mask.shape[1], x + w + 8), min(mask.shape[0], y + h + 8)
        roi = frame[y0:y1, x0:x1]
        # Use original-image channel gradients, retaining internal face edges
        # that disappear when all orange faces merge into one binary mask.
        local_edges = np.zeros(roi.shape[:2], np.uint8)
        for channel in cv2.split(cv2.GaussianBlur(roi, (5, 5), 0)):
            local_edges |= cv2.Canny(channel, canny_low, canny_high)
        near_object = cv2.dilate(object_mask[y0:y1, x0:x1], kernel, iterations=2)
        local_edges &= near_object
        edges[y0:y1, x0:x1] = local_edges
        # Close only small edge gaps. RETR_LIST also exposes bounded inner faces.
        connected = cv2.morphologyEx(local_edges, cv2.MORPH_CLOSE, kernel)
        outlines, _ = cv2.findContours(connected, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        nearby_edges = cv2.dilate(local_edges, kernel, iterations=1)
        candidates = []
        quads = [cv2.approxPolyDP(contour, .025 * cv2.arcLength(contour, True), True)
                 for contour in outlines]
        quads.extend(line_quadrilaterals(local_edges, min(w, h)))
        for quad in quads:
            if len(quad) != 4 or not cv2.isContourConvex(quad):
                continue
            area = cv2.contourArea(quad)
            if area < max(100, .12 * cv2.contourArea(blob)):
                continue
            points = quad.reshape(4, 2)
            lengths = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)
            if lengths.min() < 8 or lengths.max() / lengths.min() > 5:
                continue
            face_mask = np.zeros_like(local_edges)
            cv2.fillConvexPoly(face_mask, points, 255)
            inside = face_mask > 0
            if np.mean(mask[y0:y1, x0:x1][inside] > 0) < .85:
                continue
            # Every proposed side must be supported by real image edges.
            support = []
            for a, b in zip(points, np.roll(points, -1, axis=0)):
                line = np.zeros_like(local_edges)
                cv2.line(line, tuple(a), tuple(b), 255, 1)
                support.append(float(np.mean(nearby_edges[line > 0] > 0)))
            if min(support) < .7:
                continue
            global_points = points + [x0, y0]
            center = diagonal_center(global_points)
            if center is None:
                continue
            candidate = dict(top_corners_px=global_points, center_px=center,
                             edge_support=min(support))
            nearby = next((i for i, c in enumerate(candidates)
                           if np.linalg.norm(center - c['center_px']) < 8), None)
            if nearby is None:
                candidates.append(candidate)
            elif candidate['edge_support'] > candidates[nearby]['edge_support']:
                # Hough can produce several slightly shifted boundaries. Keep
                # the best supported one, not whichever contour was seen first.
                candidates[nearby] = candidate
        result = select_cube_face(candidates, pose_context)
        if result is not None:
            result['color'] = color or 'custom'
    return result, {'mask': mask, 'edges': edges, 'cube_contour': blob}


def draw_cube_detection(frame, result, debug=None):
    """Return an annotated copy; draw an existing result without re-detecting.

    debug is optional and comes from detect_cube_top. Without it, only the
    candidate top face, diagonals, center and status are drawn.
    """
    view = frame.copy()
    if debug is not None:
        if debug['cube_contour'] is not None:
            cv2.drawContours(view, [debug['cube_contour']], -1, (255, 255, 0), 2)
        view[debug['edges'] > 0] = (255, 100, 0)
    if result is not None:
        points = result['top_corners_px']
        cv2.polylines(view, [points.astype(np.int32)], True, (0, 255, 0), 2)
        for a, b in ((points[0], points[2]), (points[1], points[3])):
            cv2.line(view, tuple(a), tuple(b), (0, 255, 255), 1)
        cx, cy = np.rint(result['center_px']).astype(int)
        cv2.drawMarker(view, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.circle(view, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(view, f'center ({cx}, {cy}) px', (max(0, cx - 80), max(45, cy - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
    status = (f"{result.get('color', 'custom')}: {result.get('selection_mode', 'preview_2d')}"
              if result else 'No reliable top-face candidate')
    if result is not None and result.get('height_consistent') is False:
        status += f" | height mismatch {result['height_error_m'] * 1000:.0f}mm"
    cv2.putText(view, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55,
                (0, 255, 0) if result else (0, 200, 255), 2)
    return view


def detect_cube(frame, hsv_low=(9, 170, 60), hsv_high=(19, 255, 255),
                min_area=400, canny_low=15, canny_high=40, *, color=None, pose_context=None):
    """Preview compatibility wrapper: return (result, view, mask, edges).

    In pick_place, use detect_cube_top directly for pixel coordinates and call
    draw_cube_detection only when a visualization is needed.
    """
    result, debug = detect_cube_top(frame, hsv_low, hsv_high, min_area,
                                    canny_low, canny_high, color=color, pose_context=pose_context)
    view = draw_cube_detection(frame, result, debug)
    return result, view, debug['mask'], debug['edges']


def main():
    from camera_intrinsic import open_camera

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, help='Read a single image instead of camera')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('pick_place_config.json'))
    parser.add_argument('--preview-2d', action='store_true', help='Camera preview without robot connection or 3D verification')
    parser.add_argument('--camera', default=None)
    parser.add_argument('--pose-context', type=Path,
                        help='JSON with K, D, T_base_camera, side_m, top_z_m; pose must match the frame')
    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--color', choices=tuple(COLOR_HSV_RANGES),
                        help='Override the cube color in --config')
    parser.add_argument('--hsv-low', nargs=3, type=int, default=None, metavar=('H', 'S', 'V'))
    parser.add_argument('--hsv-high', nargs=3, type=int, default=None, metavar=('H', 'S', 'V'))
    parser.add_argument('--min-area', type=float, default=None)
    parser.add_argument('--canny-low', type=int, default=None)
    parser.add_argument('--canny-high', type=int, default=None)
    parser.add_argument('--output', type=Path, default=Path('cube_detection.png'))
    parser.add_argument('--no-display', action='store_true', help='Single-image mode: save without opening windows')
    args = parser.parse_args()
    if args.color and (args.hsv_low is not None or args.hsv_high is not None):
        parser.error('--color cannot be combined with --hsv-low/--hsv-high')
    if args.preview_2d and args.pose_context:
        parser.error('--preview-2d cannot be combined with --pose-context')
    config = json.loads(args.config.read_text())
    cfg = config['cube']
    custom_hsv = args.hsv_low is not None or args.hsv_high is not None
    if args.color is None and not custom_hsv:
        args.color = cfg.get('color')
    args.hsv_low = args.hsv_low if args.hsv_low is not None else cfg.get('hsv_low', [9, 170, 60])
    args.hsv_high = args.hsv_high if args.hsv_high is not None else cfg.get('hsv_high', [19, 255, 255])
    args.camera = args.camera if args.camera is not None else config.get('camera', '0')
    for name, key, default in [('min_area', 'min_area_px', 400), ('canny_low', 'canny_low', 15),
                               ('canny_high', 'canny_high', 40)]:
        if getattr(args, name) is None:
            setattr(args, name, cfg.get(key, default))
    live_pose = not (args.image or args.preview_2d or args.pose_context)
    if args.no_display and not args.image:
        parser.error('--no-display requires --image')
    if any(not 0 <= lo <= hi <= limit for lo, hi, limit in
           zip(args.hsv_low, args.hsv_high, [179, 255, 255])):
        parser.error('HSV bounds must satisfy 0 <= low <= high <= (179,255,255)')
    if args.min_area <= 0 or not 0 <= args.canny_low < args.canny_high:
        parser.error('Require min-area > 0 and 0 <= canny-low < canny-high')

    pose_context = None
    if args.pose_context:
        pose_context = json.loads(args.pose_context.read_text())
    elif not live_pose:
        print('2D preview only: no calibrated top-face verification.')

    def process(frame):
        context = pose_context
        if live_pose:
            if (frame.shape[1], frame.shape[0]) != calibration_size:
                raise ValueError('Camera resolution differs from calibration')
            positions = robot.bus.sync_read('Present_Position')
            q = {name: float(positions[name]) for name in fk.names}
            if not np.isfinite(list(q.values())).all():
                raise ValueError('Non-finite robot joint positions')
            context = dict(pose_context, T_base_camera=fk(q) @ handeye_matrix)
        return detect_cube(frame, args.hsv_low, args.hsv_high, args.min_area,
                           args.canny_low, args.canny_high, color=args.color, pose_context=context)

    def save(view):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), view):
            raise RuntimeError(f'Cannot save {args.output}')
        print(f'Saved {args.output}')

    cap = None
    robot = None
    try:
        if live_pose:
            from handeye_calibration import URDFFK, robot_reader

            root = args.config.resolve().parent
            handeye = json.loads((root / config['handeye_json']).read_text())
            if handeye.get('translation_unit') != 'meter' or handeye.get('gripper_frame') != 'gripper_link':
                raise ValueError('Hand-eye units/frame mismatch')
            handeye_matrix = np.asarray(handeye['T_gripper_camera'], np.float64)
            if (handeye_matrix.shape != (4, 4) or not np.isfinite(handeye_matrix).all()
                    or not np.allclose(handeye_matrix[3], [0, 0, 0, 1])
                    or not np.allclose(handeye_matrix[:3, :3].T @ handeye_matrix[:3, :3], np.eye(3), atol=1e-5)
                    or np.linalg.det(handeye_matrix[:3, :3]) < .99):
                raise ValueError('Invalid hand-eye transform')
            fs = cv2.FileStorage(str(root / config['camera_yml']), cv2.FILE_STORAGE_READ)
            try:
                if not fs.isOpened():
                    raise ValueError('Cannot read camera calibration')
                K, D = fs.getNode('K').mat(), fs.getNode('D').mat()
                calibration_size = (int(fs.getNode('image_width').real()),
                                    int(fs.getNode('image_height').real()))
            finally:
                fs.release()
            if K is None or K.shape != (3, 3) or min(calibration_size) <= 0:
                raise ValueError('Invalid camera calibration')
            if ((args.width is not None and args.width != calibration_size[0])
                    or (args.height is not None and args.height != calibration_size[1])):
                raise ValueError('Requested resolution differs from calibration')
            args.width, args.height = calibration_size
            side = float(cfg['side_mm']) / 1000
            if not np.isfinite(side) or side <= 0:
                raise ValueError('Cube side_mm must be positive')
            pose_context = dict(K=K, D=D if D is not None else np.zeros(5),
                                side_m=side, top_z_m=float(config['table_z_m']) + side)
            for key in ('enforce_height', 'height_tolerance_m', 'max_tilt_deg', 'max_reprojection_px'):
                if key in cfg:
                    pose_context[key] = cfg[key]
            fk = URDFFK(root / config['urdf'], offsets=handeye.get('joint_offsets_deg'))
            calibration_dir = config.get('calibration_dir')
            robot = robot_reader(argparse.Namespace(
                port=config['port'], robot_id=config['robot_id'],
                calibration_dir=root / calibration_dir if calibration_dir else None,
                disable_torque=False))
            print('3D detection: reading current joints; no motion commands or torque changes.')
        args.width = args.width if args.width is not None else 1280
        args.height = args.height if args.height is not None else 720
        if args.image:
            frame = cv2.imread(str(args.image))
            if frame is None:
                raise ValueError(f'Cannot read {args.image}')
            result, view, mask, edges = process(frame)
            save(view)
            print('Center (pixels):', result['center_px'].tolist() if result else 'not detected')
            if not args.no_display:
                cv2.imshow('Cube detection', view)
                cv2.imshow('Selected color mask', mask)
                cv2.imshow('Original image edges', edges)
                cv2.waitKey(0)
        else:
            cap = open_camera(args.camera, args.width, args.height)
            print('Q / ESC: quit; S: save annotated frame. Green top, cyan outline, blue edges, red center.')
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Camera frame read failed')
                _, view, mask, edges = process(frame)
                cv2.imshow('Cube detection', view)
                cv2.imshow('Selected color mask', mask)
                cv2.imshow('Original image edges', edges)
                key = cv2.waitKey(1) & 0xff
                if key in (27, ord('q')):
                    break
                if key == ord('s'):
                    save(view)
    finally:
        try:
            if cap is not None:
                cap.release()
        finally:
            try:
                if robot is not None:
                    robot.bus.disconnect(disable_torque=False)
            finally:
                if not args.no_display:
                    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
