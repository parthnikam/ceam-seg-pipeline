"""Camera overlay and readable ego-BEV / map drawing."""

from __future__ import annotations

import cv2
import numpy as np

from bev import Grid
from boundaries import TrackFit, link_polylines
from camera import Camera
from ground import Plane, plane_to_pixels

ROAD = (70, 150, 70)
BG = (28, 28, 28)
GRID = (48, 48, 48)
EDGE = (230, 230, 230)
CENTER = (80, 220, 255)
CART = (0, 210, 255)
START = (80, 80, 255)
PATH = (240, 240, 240)


def overlay_camera(
    frame: np.ndarray,
    drivable: np.ndarray,
    obstacle: np.ndarray,
    camera: Camera,
    plane: Plane | None,
    fit: TrackFit | None,
) -> np.ndarray:
    color = np.zeros_like(frame)
    color[drivable > 0] = (0, 180, 0)
    out = cv2.addWeighted(frame, 1.0, color, 0.28, 0)
    if fit is not None and plane is not None:
        left, right, center = link_polylines(fit, plane)
        _draw_link_poly(out, camera, left, (0, 80, 255), 3)
        _draw_link_poly(out, camera, right, (255, 80, 0), 3)
        _draw_link_poly(out, camera, center, (0, 220, 255), 2)
        _draw_side_extents(out, camera, plane, fit)
        label = f"ey={fit.e_y:+.2f}m  epsi={np.degrees(fit.e_psi):+.1f}deg  w={fit.width:.1f}m  k={fit.kappa:+.3f}/m"
        cv2.putText(out, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(out, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA)
    return out


def _draw_link_poly(img: np.ndarray, camera: Camera, xyz_link: np.ndarray, bgr: tuple[int, int, int], thick: int) -> None:
    xyz_opt = Camera.link_to_optical(xyz_link)
    u, v, ok = camera.project(xyz_opt)
    # Far plane hits sit on the horizon; keep lines on the visible pavement.
    h = img.shape[0]
    ok = ok & (v >= 0.45 * h) & (v < h - 1)
    pts = np.stack([u[ok], v[ok]], axis=1)
    if len(pts) < 2:
        return
    cv2.polylines(img, [np.int32(pts)], False, bgr, thick, cv2.LINE_AA)


def _draw_side_extents(
    img: np.ndarray,
    camera: Camera,
    plane: Plane,
    fit: TrackFit,
) -> None:
    """Draw nearest->farthest lines for the left and right road edges.

    Shows where the detected road starts (bottom of frame) and ends (top of
    frame / vanishing point) on each side, so it is easy to see whether YOLO
    or the planar fit is dropping pavement at either extreme.
    """
    h = img.shape[0]
    # Left side: nearest (idx 0) -> farthest (idx -1)
    xs_l = np.array([fit.x[0], fit.x[-1]], dtype=np.float64)
    ys_l = np.array([fit.y_left[0], fit.y_left[-1]], dtype=np.float64)
    # Right side: nearest (idx 0) -> farthest (idx -1)
    xs_r = np.array([fit.x[0], fit.x[-1]], dtype=np.float64)
    ys_r = np.array([fit.y_right[0], fit.y_right[-1]], dtype=np.float64)

    for xs, ys, color in ((xs_l, ys_l, (0, 80, 255)), (xs_r, ys_r, (255, 80, 0))):
        u, v, ok = plane_to_pixels(camera, plane, xs, ys)
        ok = ok & np.isfinite(u) & np.isfinite(v) & (v >= 0.45 * h) & (v < h - 1)
        if not np.all(ok):
            continue
        pts = np.int32(np.stack([u[ok], v[ok]], axis=1))
        if len(pts) >= 2:
            cv2.line(img, tuple(pts[0]), tuple(pts[1]), color, 3, cv2.LINE_AA)


def cart_triangle(px: float, py: float, yaw: float, size: float) -> np.ndarray:
    """Wedge pointing along yaw. Image: +x (forward) is up, so yaw=0 → tip up."""
    s, c = np.sin(yaw), np.cos(yaw)
    tip = (px - s * size, py - c * size)
    left = (px + c * size * 0.45 + s * size * 0.35, py - s * size * 0.45 + c * size * 0.35)
    right = (px - c * size * 0.45 + s * size * 0.35, py + s * size * 0.45 + c * size * 0.35)
    return np.int32([tip, left, right])


def draw_cart(
    img: np.ndarray,
    px: int,
    py: int,
    yaw: float,
    size: int,
    label: str,
    color=CART,
    label_side: int = 1,
) -> None:
    tri = cart_triangle(px, py, yaw, size)
    cv2.fillConvexPoly(img, tri, color)
    cv2.polylines(img, [tri], True, (20, 20, 20), 1, cv2.LINE_AA)
    tx = px + size if label_side >= 0 else px - size - 8 * len(label)
    cv2.putText(img, label, (tx, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def render_bev(
    drive_bev: np.ndarray,
    obst_bev: np.ndarray,
    grid: Grid,
    fit: TrackFit | None = None,
    scale: int = 4,
) -> np.ndarray:
    vis = np.full((grid.nx, grid.ny, 3), BG, np.uint8)
    vis[drive_bev > 0] = ROAD
    vis = np.flipud(vis)
    vis = cv2.resize(vis, (grid.ny * scale, grid.nx * scale), interpolation=cv2.INTER_NEAREST)
    _draw_metre_grid(vis, grid, scale)
    if fit is not None:
        def to_px(x, y):
            col = (grid.ymax - y) / grid.cell * scale
            row = (grid.xmax - x) / grid.cell * scale
            return np.stack([col, row], axis=1)

        cv2.polylines(vis, [np.int32(to_px(fit.x, fit.y_left))], False, EDGE, 2, cv2.LINE_AA)
        cv2.polylines(vis, [np.int32(to_px(fit.x, fit.y_right))], False, EDGE, 2, cv2.LINE_AA)
        cv2.polylines(vis, [np.int32(to_px(fit.x, fit.y_center))], False, CENTER, 2, cv2.LINE_AA)
    cart_px = vis.shape[1] // 2
    cart_py = vis.shape[0] - 14
    draw_cart(vis, cart_px, cart_py, 0.0, 16, "CART")
    _legend(vis, [("CART", CART), ("road", ROAD), ("edge", EDGE)])
    for metres in (4, 8, 12):
        if metres > grid.xmax:
            continue
        row = int((grid.xmax - metres) / grid.cell * scale)
        cv2.putText(vis, f"{metres}m", (6, max(14, row)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    return vis


def _draw_metre_grid(vis: np.ndarray, grid: Grid, scale: int) -> None:
    h, w = vis.shape[:2]
    for x in np.arange(np.ceil(grid.xmin / 5) * 5, grid.xmax, 5):
        row = int((grid.xmax - x) / grid.cell * scale)
        if 0 <= row < h:
            cv2.line(vis, (0, row), (w - 1, row), GRID, 1)
    for y in np.arange(np.ceil(grid.ymin / 5) * 5, grid.ymax, 5):
        col = int((grid.ymax - y) / grid.cell * scale)
        if 0 <= col < w:
            cv2.line(vis, (col, 0), (col, h - 1), GRID, 1)


def _legend(img: np.ndarray, items: list[tuple[str, tuple[int, int, int]]]) -> None:
    x, y = 8, img.shape[0] - 12
    for name, color in reversed(items):
        cv2.circle(img, (x + 6, y - 4), 5, color, -1)
        cv2.putText(img, name, (x + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (210, 210, 210), 1, cv2.LINE_AA)
        y -= 16


def _letterbox(img: np.ndarray | None, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), BG, np.uint8)
    if img is None or img.size == 0 or width < 8 or height < 8:
        return canvas
    ih, iw = img.shape[:2]
    scale = min(width / max(iw, 1), height / max(ih, 1))
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    y0 = (height - nh) // 2
    x0 = (width - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def compose_hud(
    overlay: np.ndarray,
    world_map: np.ndarray,
    ego_bev: np.ndarray,
    panel_w: int = 720,
) -> np.ndarray:
    """Camera frame with a live map + ego BEV column on the right."""
    h, _w = overlay.shape[:2]
    panel_w = max(480, panel_w)
    panel = np.full((h, panel_w, 3), 16, np.uint8)
    title_h = 26
    split = int(h * 0.70)
    panel[title_h:split] = _letterbox(world_map, panel_w, max(1, split - title_h))
    panel[split + title_h :] = _letterbox(ego_bev, panel_w, max(1, h - split - title_h))
    cv2.putText(panel, "MAP  (NOW = you)", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(panel, "EGO  (cart at bottom)", (10, split + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.line(panel, (0, split), (panel_w - 1, split), (50, 50, 50), 1)
    cv2.line(panel, (0, 0), (0, h - 1), (70, 70, 70), 2)
    return np.hstack([overlay, panel])
