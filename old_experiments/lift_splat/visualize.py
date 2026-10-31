"""BEV occupancy / colour maps and a side-by-side panel."""

from __future__ import annotations

import numpy as np
import cv2

from .geometry import gen_dx_bx
from .lift import GridConf


EGO_GREEN = (0, 185, 118)
GRID_GRAY = (55, 55, 55)
AXIS_GRAY = (90, 90, 90)


def _grid_arrays(grid: GridConf):
    dx, bx, nx = gen_dx_bx(grid.xbound, grid.ybound, grid.zbound)
    return dx.numpy(), bx.numpy(), nx.numpy()


def occupancy_to_bgr(occupancy: np.ndarray, grid: GridConf, scale: int = 10) -> np.ndarray:
    """Car at the bottom, left of car on the left of the image."""
    occ = cv2.GaussianBlur(occupancy.astype(np.float32), (0, 0), 0.6)
    peak = float(np.percentile(occ, 99)) if occ.size else 1.0
    vis = np.clip(occ / max(peak, 1e-6), 0.0, 1.0)
    vis = vis[::-1, ::-1]
    heat = (vis * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
    color[vis < 1e-3] = (24, 24, 24)
    color = cv2.resize(
        color,
        (color.shape[1] * scale, color.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    return _annotate(color, grid, scale)


def colour_to_bgr(colour: np.ndarray, grid: GridConf, scale: int = 10) -> np.ndarray:
    vis = cv2.GaussianBlur(colour.astype(np.float32), (0, 0), 0.5)
    vis = vis[::-1, ::-1]
    present = vis.sum(axis=2) > 1e-4
    bgr = (np.clip(vis[..., ::-1], 0.0, 1.0) * 255.0).astype(np.uint8)
    bgr[~present] = (24, 24, 24)
    bgr = cv2.resize(
        bgr,
        (bgr.shape[1] * scale, bgr.shape[0] * scale),
        interpolation=cv2.INTER_NEAREST,
    )
    return _annotate(bgr, grid, scale)


def _annotate(img: np.ndarray, grid: GridConf, scale: int) -> np.ndarray:
    dx, _, _ = _grid_arrays(grid)
    h = img.shape[0]

    xmin, xmax, _ = grid.xbound
    ymin, ymax, _ = grid.ybound

    def world_to_px(x_fwd: float, y_left: float) -> tuple[int, int]:
        col = int(round((ymax - y_left) / dx[1] * scale))
        row = int(round((xmax - x_fwd) / dx[0] * scale))
        return col, row

    for x_m in range(int(np.ceil(xmin / 10.0)) * 10, int(xmax) + 1, 10):
        c0, r0 = world_to_px(x_m, ymin)
        c1, r1 = world_to_px(x_m, ymax)
        cv2.line(img, (c0, r0), (c1, r1), GRID_GRAY, 1)
    for y_m in range(int(np.ceil(ymin / 5.0)) * 5, int(ymax) + 1, 5):
        c0, r0 = world_to_px(xmin, y_m)
        c1, r1 = world_to_px(xmax, y_m)
        cv2.line(img, (c0, r0), (c1, r1), GRID_GRAY, 1)

    # origin axes
    c0, r0 = world_to_px(0.0, 0.0)
    c1, r1 = world_to_px(min(5.0, xmax), 0.0)
    cv2.arrowedLine(img, (c0, r0), (c1, r1), AXIS_GRAY, 1, tipLength=0.15)

    car_len, car_wid = 5.4, 2.0
    # halo cam is ~2 m behind the nose
    nose = 2.2
    tail = nose - car_len
    pts = np.array(
        [
            world_to_px(nose, car_wid / 2.0),
            world_to_px(nose, -car_wid / 2.0),
            world_to_px(tail, -car_wid / 2.0),
            world_to_px(tail, car_wid / 2.0),
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(img, pts, EGO_GREEN)

    label = f"{xmax:.0f}m fwd  {ymax - ymin:.0f}m wide  {dx[0]:.2f}m/px"
    cv2.putText(img, label, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, "BEV", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return img


def panel(frame_bgr: np.ndarray, bev_bgr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Camera stacked over the mask on the left, BEV on the right."""
    left_w = 960
    fh, fw = frame_bgr.shape[:2]
    left_h = max(1, int(left_w * fh / fw))
    cam = cv2.resize(frame_bgr, (left_w, left_h), interpolation=cv2.INTER_AREA)
    if mask is None:
        left = cam
    else:
        mask_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
        mask_bgr = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
        mask_bgr = cv2.resize(mask_bgr, (left_w, left_h), interpolation=cv2.INTER_NEAREST)
        left = np.concatenate([cam, mask_bgr], axis=0)
    bh, bw = bev_bgr.shape[:2]
    bev_h = left.shape[0]
    bev_w = max(1, int(bw * (bev_h / bh)))
    bev = cv2.resize(bev_bgr, (bev_w, bev_h), interpolation=cv2.INTER_NEAREST)
    return np.concatenate([left, bev], axis=1)


