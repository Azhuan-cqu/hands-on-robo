#!/usr/bin/env python3
"""Orange cube preview; image coordinates only, no robot connection.

Assumes an upright image viewing a cube from above. With several visible faces,
the uppermost edge-bounded orange quadrilateral is a top-face hypothesis, not a
geometrically verified pose. Occlusion or weak face boundaries can cause failure.
"""
import argparse
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np


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


def detect_cube_top(frame, hsv_low=(0, 80, 50), hsv_high=(25, 255, 255),
                min_area=400, canny_low=15, canny_high=40):
    """Detect without drawing, GUI, camera access, or modifying the BGR input.

    Returns (result, debug). result is None when no unambiguous candidate exists,
    otherwise a dict with center_px (float array [u,v]), top_corners_px (4x2
    cyclic corner array), and edge_support (minimum supported edge fraction,
    not a probability). Coordinates are pixels in the original image, NOT 3D.
    debug contains mask, edges and cube_contour for optional visualization.

    Uses the largest orange component and an upright, above-cube viewpoint
    heuristic to choose a top face; this does not verify physical cube geometry.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_low, np.uint8), np.array(hsv_high, np.uint8))
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
        # Explicit viewpoint heuristic: camera looks down, image is upright.
        candidates.sort(key=lambda c: c['center_px'][1])
        if candidates and (len(candidates) == 1 or
                           candidates[1]['center_px'][1] - candidates[0]['center_px'][1] > .08 * h):
            result = candidates[0]
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
    status = 'TOP FACE CANDIDATE (viewpoint heuristic)' if result else 'No reliable top-face candidate'
    cv2.putText(view, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55,
                (0, 255, 0) if result else (0, 200, 255), 2)
    return view


def detect_cube(frame, hsv_low=(0, 80, 50), hsv_high=(25, 255, 255),
                min_area=400, canny_low=15, canny_high=40):
    """Preview compatibility wrapper: return (result, view, mask, edges).

    In pick_place, use detect_cube_top directly for pixel coordinates and call
    draw_cube_detection only when a visualization is needed.
    """
    result, debug = detect_cube_top(frame, hsv_low, hsv_high, min_area,
                                    canny_low, canny_high)
    view = draw_cube_detection(frame, result, debug)
    return result, view, debug['mask'], debug['edges']


def main():
    from camera_intrinsic import open_camera

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, help='Read a single image instead of camera')
    parser.add_argument('--camera', default='0')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--hsv-low', nargs=3, type=int, default=[0, 80, 50], metavar=('H', 'S', 'V'))
    parser.add_argument('--hsv-high', nargs=3, type=int, default=[25, 255, 255], metavar=('H', 'S', 'V'))
    parser.add_argument('--min-area', type=float, default=400)
    parser.add_argument('--canny-low', type=int, default=15)
    parser.add_argument('--canny-high', type=int, default=40)
    parser.add_argument('--output', type=Path, default=Path('cube_detection.png'))
    parser.add_argument('--no-display', action='store_true', help='Single-image mode: save without opening windows')
    args = parser.parse_args()
    if args.no_display and not args.image:
        parser.error('--no-display requires --image')
    if any(not 0 <= lo <= hi <= limit for lo, hi, limit in
           zip(args.hsv_low, args.hsv_high, [179, 255, 255])):
        parser.error('HSV bounds must satisfy 0 <= low <= high <= (179,255,255)')
    if args.min_area <= 0 or not 0 <= args.canny_low < args.canny_high:
        parser.error('Require min-area > 0 and 0 <= canny-low < canny-high')

    def process(frame):
        return detect_cube(frame, args.hsv_low, args.hsv_high, args.min_area,
                           args.canny_low, args.canny_high)

    def save(view):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), view):
            raise RuntimeError(f'Cannot save {args.output}')
        print(f'Saved {args.output}')

    cap = None
    try:
        if args.image:
            frame = cv2.imread(str(args.image))
            if frame is None:
                raise ValueError(f'Cannot read {args.image}')
            result, view, mask, edges = process(frame)
            save(view)
            print('Center (pixels):', result['center_px'].tolist() if result else 'not detected')
            if not args.no_display:
                cv2.imshow('Cube detection', view)
                cv2.imshow('Orange mask', mask)
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
                cv2.imshow('Orange mask', mask)
                cv2.imshow('Original image edges', edges)
                key = cv2.waitKey(1) & 0xff
                if key in (27, ord('q')):
                    break
                if key == ord('s'):
                    save(view)
    finally:
        if cap is not None:
            cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
