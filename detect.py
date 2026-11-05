#!/usr/bin/env python3
"""Colour-blob + edge-supported quadrilateral detection for the orange cube and blue tray.

Pixel coordinates only; no camera, GUI or robot access. Returns None when there is no
unambiguous candidate. Cube: uppermost supported orange quad is the top-face hypothesis
(camera looks down, image upright). Tray: largest supported blue quad is the outer border.
"""
from itertools import combinations

import cv2
import numpy as np

KERNEL = np.ones((3, 3), np.uint8)


def diagonal_center(quad):
    """Perspective-correct image of a square's center, using cyclic corners."""
    a, b, c, d = np.asarray(quad, dtype=float)
    system = np.column_stack((c - a, -(d - b)))
    if abs(np.linalg.det(system)) < 1e-6:
        return None
    t, u = np.linalg.solve(system, b - a)
    return a + t * (c - a) if 0 < t < 1 and 0 < u < 1 else None


def line_quadrilaterals(edges, object_size):
    """Bridge broken boundaries with intersections of supported Hough lines; caller validates sides."""
    minimum = max(12, int(object_size * .25))
    segments = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(12, minimum // 2),
                               minLineLength=minimum, maxLineGap=max(4, int(object_size * .1)))
    if segments is None:
        return []
    segments = sorted(segments.reshape(-1, 4), key=lambda s: -np.linalg.norm(s[2:] - s[:2]))
    lines = []
    for segment in segments:
        a, b = segment.reshape(2, 2).astype(float)
        direction = (b - a) / np.linalg.norm(b - a)
        normal = np.array([-direction[1], direction[0]])
        if any(abs(normal @ n) > .995 and abs(n @ a - d) < 3 for n, d in lines):
            continue
        lines.append((normal, normal @ a))
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


def quad_candidates(frame, hsv_low, hsv_high, min_area, canny_low, canny_high, area_frac, fill, dedup=False):
    """Largest colour blob -> edge-supported convex quads inside it.

    Each candidate: corners_px (4x2 cyclic), center_px, edge_support (min supported side
    fraction, not a probability), area_px. debug: mask, blob height.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_low, np.uint8), np.array(hsv_high, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL)
    contours = [c for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                if cv2.contourArea(c) >= min_area]
    debug = {'mask': mask, 'h': 0}
    if not contours:
        return [], debug
    blob = max(contours, key=cv2.contourArea)
    object_mask = np.zeros_like(mask)
    cv2.drawContours(object_mask, [blob], -1, 255, cv2.FILLED)
    x, y, w, h = cv2.boundingRect(blob)
    debug['h'] = h
    x0, y0 = max(0, x - 8), max(0, y - 8)
    x1, y1 = min(mask.shape[1], x + w + 8), min(mask.shape[0], y + h + 8)
    roi = frame[y0:y1, x0:x1]
    # Original-image channel gradients keep internal face edges that merge in the binary mask.
    local_edges = np.zeros(roi.shape[:2], np.uint8)
    for channel in cv2.split(cv2.GaussianBlur(roi, (5, 5), 0)):
        local_edges |= cv2.Canny(channel, canny_low, canny_high)
    local_edges &= cv2.dilate(object_mask[y0:y1, x0:x1], KERNEL, iterations=2)
    connected = cv2.morphologyEx(local_edges, cv2.MORPH_CLOSE, KERNEL)
    outlines, _ = cv2.findContours(connected, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    nearby_edges = cv2.dilate(local_edges, KERNEL, iterations=1)
    quads = [cv2.approxPolyDP(c, .025 * cv2.arcLength(c, True), True) for c in outlines]
    quads.extend(line_quadrilaterals(local_edges, min(w, h)))
    candidates = []
    for quad in quads:
        if len(quad) != 4 or not cv2.isContourConvex(quad):
            continue
        area = cv2.contourArea(quad)
        if area < max(100, area_frac * cv2.contourArea(blob)):
            continue
        points = quad.reshape(4, 2)
        lengths = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)
        if lengths.min() < 8 or lengths.max() / lengths.min() > 5:
            continue
        face_mask = np.zeros_like(local_edges)
        cv2.fillConvexPoly(face_mask, points, 255)
        if np.mean(mask[y0:y1, x0:x1][face_mask > 0] > 0) < fill:
            continue
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
        candidate = dict(corners_px=global_points, center_px=center, edge_support=min(support), area_px=float(area))
        near = next((i for i, c in enumerate(candidates) if np.linalg.norm(center - c['center_px']) < 8), None) if dedup else None
        if near is None:
            candidates.append(candidate)
        elif candidate['edge_support'] > candidates[near]['edge_support']:
            candidates[near] = candidate  # Hough gives shifted duplicates; keep the best supported
    return candidates, debug


def detect_cube_top(frame, hsv_low=(0, 80, 50), hsv_high=(25, 255, 255), min_area=400, canny_low=15, canny_high=40):
    """Uppermost orange quad = top face; ambiguous when another candidate is nearly as high."""
    candidates, debug = quad_candidates(frame, hsv_low, hsv_high, min_area, canny_low, canny_high, .12, .85, dedup=True)
    candidates.sort(key=lambda c: c['center_px'][1])
    ok = candidates and (len(candidates) == 1 or
                         candidates[1]['center_px'][1] - candidates[0]['center_px'][1] > .08 * debug['h'])
    return (candidates[0] if ok else None), debug


def detect_tray_center(frame, hsv_low=(90, 70, 40), hsv_high=(135, 255, 255), min_area=400, canny_low=15, canny_high=40):
    """Largest blue quad = outer tray border (center of the border, not of an asymmetric interior)."""
    candidates, debug = quad_candidates(frame, hsv_low, hsv_high, min_area, canny_low, canny_high, .65, .65)
    return (max(candidates, key=lambda c: (c['area_px'], c['edge_support'])) if candidates else None), debug


def draw_detection(frame, result):
    """Annotated copy: boundary, diagonals and center of an existing result."""
    view = frame.copy()
    if result is not None:
        points = result['corners_px']
        cv2.polylines(view, [points.astype(np.int32)], True, (0, 255, 0), 2)
        for a, b in ((points[0], points[2]), (points[1], points[3])):
            cv2.line(view, tuple(a), tuple(b), (0, 255, 255), 1)
        cx, cy = np.rint(result['center_px']).astype(int)
        cv2.drawMarker(view, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(view, f'({cx}, {cy})', (max(0, cx - 40), max(45, cy - 15)), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
    return view


if __name__ == '__main__':  # ponytail: self-check on a synthetic frame
    img = np.full((400, 600, 3), 255, np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (200, 80, 0), -1)   # blue tray
    cv2.rectangle(img, (170, 170), (230, 230), (0, 120, 255), -1)  # orange cube top
    cube, _ = detect_cube_top(img)
    tray, _ = detect_tray_center(img)
    assert cube is not None and np.allclose(cube['center_px'], [200, 200], atol=2), cube
    assert tray is not None and np.allclose(tray['center_px'], [200, 200], atol=2), tray
    print('ok')
