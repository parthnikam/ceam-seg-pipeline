"""ZED X geometry: turbo-depth invert, unproject, optical ↔ camera_link."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DEPTH_MAX_M = 15.0

# camera_link (x forward, y left, z up) → optical (x right, y down, z forward).
# Matches the ROS optical quaternion in calib.yaml (0.5, -0.5, 0.5, -0.5).
R_OPT_FROM_LINK = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)
R_LINK_FROM_OPT = R_OPT_FROM_LINK.T

# Extrinsic between the two ZED X camera_link frames, from tf_static in calib.yaml:
#   zedx_camera_center -> left_camera_frame:  t = [-0.01,  0.06, 0.0]
#   zedx_camera_center -> right_camera_frame: t = [-0.01, -0.06, 0.0]
# Both optical frames share the same rotation, so camera_link frames differ only
# by a pure translation: right -> left = [0.0, 0.12, 0.0] (right is 0.12 m to the
# right / negative-Y of left).
CAMERA_LINK_LEFT_FROM_RIGHT = np.array([0.0, 0.12, 0.0], dtype=np.float64)
CAMERA_LINK_RIGHT_FROM_LEFT = -CAMERA_LINK_LEFT_FROM_RIGHT


def _turbo_bgr_lut() -> np.ndarray:
    gray = np.arange(256, dtype=np.uint8).reshape(1, 256)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)[0].astype(np.int16)


_TURBO_BGR = _turbo_bgr_lut()


def depth_from_turbo(bgr: np.ndarray) -> np.ndarray:
    """Invert COLORMAP_TURBO encoding (0–15 m). Black / near-black → NaN."""
    h, w = bgr.shape[:2]
    packed = (
        bgr[:, :, 0].astype(np.uint32) << 16
        | bgr[:, :, 1].astype(np.uint32) << 8
        | bgr[:, :, 2].astype(np.uint32)
    )
    unique, inverse = np.unique(packed, return_inverse=True)
    colors = np.stack(
        [(unique >> 16) & 255, (unique >> 8) & 255, unique & 255],
        axis=1,
    ).astype(np.int32)
    lut = _TURBO_BGR.astype(np.int32)
    dist = np.sum((colors[:, None, :] - lut[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(dist, axis=1)
    min_d = dist[np.arange(dist.shape[0]), nearest]
    metres = nearest.astype(np.float32) / 255.0 * DEPTH_MAX_M
    # Compressed black/sky is off the colormap; drop it.
    metres[min_d > 40 ** 2] = np.nan
    hsv_like_sat = colors.max(axis=1) - colors.min(axis=1)
    metres[(colors.max(axis=1) < 18) | (hsv_like_sat < 12)] = np.nan
    depth = metres[inverse].reshape(h, w)
    valid = np.isfinite(depth).astype(np.uint8) * 255
    # Median in index space, keep holes as NaN.
    index = np.zeros((h, w), np.uint8)
    finite = np.isfinite(depth)
    index[finite] = np.clip(np.rint(depth[finite] / DEPTH_MAX_M * 255.0), 0, 255).astype(np.uint8)
    index = cv2.medianBlur(index, 9)
    valid = cv2.medianBlur(valid, 9)
    out = index.astype(np.float32) / 255.0 * DEPTH_MAX_M
    out[valid < 128] = np.nan
    return out


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = qx * qx + qy * qy + qz * qz + qw * qw
    s = 2.0 / n if n > 1e-12 else 0.0
    x, y, z, w = qx, qy, qz, qw
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / np.linalg.norm(out)
    theta = np.arccos(dot)
    s0 = np.sin((1.0 - t) * theta) / np.sin(theta)
    s1 = np.sin(t * theta) / np.sin(theta)
    return s0 * q0 + s1 * q1


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, xyzw. `q1 * q2` applies q2 first (body-frame increment)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    out = np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )
    n = float(np.linalg.norm(out))
    return out / n if n > 1e-12 else np.array([0.0, 0.0, 0.0, 1.0])


def quat_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = rotvec / theta
    s = np.sin(theta * 0.5)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(theta * 0.5)])


def integrate_gyro(
    quat_xyzw: np.ndarray,
    t_imu: np.ndarray,
    gyro: np.ndarray,
    t0: float,
    t1: float,
) -> np.ndarray:
    """Propagate world-from-body quat with body-frame gyro from t0 to t1."""
    q = np.asarray(quat_xyzw, dtype=np.float64).copy()
    if t1 <= t0 or t_imu.size == 0:
        return q
    i0 = int(np.searchsorted(t_imu, t0, side="left"))
    i1 = int(np.searchsorted(t_imu, t1, side="right"))
    t_prev = t0
    start = max(i0 - 1, 0)
    end = min(max(i1, start + 1), len(t_imu))
    for k in range(start, end):
        t = float(t_imu[k])
        if t <= t0:
            t_prev = t
            continue
        w = gyro[max(k - 1, 0)]
        t_clip = t if t < t1 else t1
        dt = t_clip - t_prev
        if dt > 0.0:
            q = quat_mul(q, quat_from_rotvec(w * dt))
        t_prev = t_clip
        if t >= t1:
            break
    if t_prev < t1:
        w = gyro[min(max(end - 1, 0), len(gyro) - 1)]
        q = quat_mul(q, quat_from_rotvec(w * (t1 - t_prev)))
    return q


def pose_matrix(xyz: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_rot(*quat_xyzw)
    T[:3, 3] = xyz
    return T


def yaw_from_rot(R: np.ndarray) -> float:
    """Yaw of camera_link x-axis in the odom xy plane."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


@dataclass
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_k(cls, width: int, height: int, k: list[float]) -> Camera:
        return cls(width, height, float(k[0]), float(k[4]), float(k[2]), float(k[5]))

    @classmethod
    def from_calib_info(cls, info: dict) -> "Camera":
        return cls(
            int(info["width"]),
            int(info["height"]),
            float(info["K"][0]),
            float(info["K"][4]),
            float(info["K"][2]),
            float(info["K"][5]),
        )

    def unproject(self, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Pixels + metric depth → Nx3 in the optical frame."""
        u = np.asarray(u, dtype=np.float64).ravel()
        v = np.asarray(v, dtype=np.float64).ravel()
        z = np.asarray(z, dtype=np.float64).ravel()
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return np.stack([x, y, z], axis=1)

    def project(self, xyz_opt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = xyz_opt[:, 2]
        ok = z > 0.15
        u = self.fx * xyz_opt[:, 0] / np.maximum(z, 1e-6) + self.cx
        v = self.fy * xyz_opt[:, 1] / np.maximum(z, 1e-6) + self.cy
        ok &= np.isfinite(u) & np.isfinite(v)
        return u, v, ok

    @staticmethod
    def optical_to_link(xyz_opt: np.ndarray) -> np.ndarray:
        return xyz_opt @ R_LINK_FROM_OPT.T

    @staticmethod
    def link_to_optical(xyz_link: np.ndarray) -> np.ndarray:
        return xyz_link @ R_OPT_FROM_LINK.T
