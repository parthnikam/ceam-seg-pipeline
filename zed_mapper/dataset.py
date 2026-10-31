"""Load a gocart ZED extract (test4 / test5 layout) with left + right cameras."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from camera import Camera, depth_from_turbo, integrate_gyro, pose_matrix, slerp

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class Frame:
    index: int
    bgr_left: np.ndarray
    bgr_right: np.ndarray | None
    depth_m_left: np.ndarray
    T_odom_cam: np.ndarray
    yolo_road_left: np.ndarray | None
    yolo_road_right: np.ndarray | None


class ExtractSequence:
    def __init__(self, root: Path, mask_dir: Path | None = None):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"extract not found: {self.root}")
        calib_path = self.root / "calib.yaml"
        pose_path = self.root / "pose_tum.txt"
        depth_path = self.root / "depth.mp4"
        if not calib_path.is_file():
            raise FileNotFoundError(f"missing {calib_path}")
        if not pose_path.is_file():
            raise FileNotFoundError(f"missing {pose_path}")
        if not depth_path.is_file():
            raise FileNotFoundError(f"missing {depth_path}")

        with calib_path.open() as f:
            calib = yaml.safe_load(f)
        cam_info = calib["camera_info"]

        # Left camera (always present; depth is registered to this camera).
        left_info = cam_info["/zedx/zed_node/left/color/rect/image/camera_info"]
        self.camera_left = Camera.from_calib_info(left_info)

        # Right camera (optional; no depth stream registered to it).
        right_key = "/zedx/zed_node/right/color/rect/image/camera_info"
        self.camera_right: Camera | None = None
        if right_key in cam_info:
            self.camera_right = Camera.from_calib_info(cam_info[right_key])

        self._t, self._xyz, self._quat = _load_tum(pose_path)
        self.n_tum = len(self._t)
        self._imu_t, self._imu_gyro = _load_imu(self.root / "imu.csv")
        self.n_imu = 0 if self._imu_t is None else len(self._imu_t)

        # ---- Discover left RGB source ----
        left_img_dir = self.root / "images" / "left"
        left_video = self.root / "left.mp4"
        if left_img_dir.is_dir():
            left_images = sorted(p for p in left_img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        else:
            left_images = []
        if left_images:
            self._rgb_left_kind = "images"
            self._rgb_left_items = left_images
            n_left = len(left_images)
        elif left_video.is_file():
            self._rgb_left_kind = "video"
            self._rgb_left_items = [left_video]
            cap = cv2.VideoCapture(str(left_video))
            n_left = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
        else:
            raise FileNotFoundError(f"no left images or left.mp4 in {self.root}")

        # ---- Discover right RGB source (optional) ----
        right_img_dir = self.root / "images" / "right"
        right_video = self.root / "right.mp4"
        if right_img_dir.is_dir():
            right_images = sorted(p for p in right_img_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        else:
            right_images = []
        self.has_right = False
        if right_images:
            self._rgb_right_kind = "images"
            self._rgb_right_items = right_images
            n_right = len(right_images)
            self.has_right = True
        elif right_video.is_file():
            self._rgb_right_kind = "video"
            self._rgb_right_items = [right_video]
            cap = cv2.VideoCapture(str(right_video))
            n_right = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            self.has_right = True

        # Depth is only registered to the left camera.
        cap = cv2.VideoCapture(str(depth_path))
        n_depth = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.depth_fps = float(cap.get(cv2.CAP_PROP_FPS) or 4.0)
        cap.release()

        self.n_frames = min(n_left, n_depth)
        if self.n_frames <= 0:
            raise RuntimeError(f"no frames in {self.root}")
        if self.has_right:
            self.n_frames = min(self.n_frames, n_right)

        # ---- YOLO mask directories ----
        if mask_dir is not None:
            self.yolo_dir_left = Path(mask_dir)
            if not self.yolo_dir_left.is_dir():
                raise FileNotFoundError(f"mask folder not found: {self.yolo_dir_left}")
        else:
            self.yolo_dir_left = self.root / "images" / "left" / "output" / "road"
            if not self.yolo_dir_left.is_dir():
                fallback = self.root / "output" / "road"
                self.yolo_dir_left = fallback if fallback.is_dir() else None

        self.yolo_dir_right: Path | None = None
        if self.has_right:
            self.yolo_dir_right = self.root / "images" / "right" / "output" / "road"
            if not self.yolo_dir_right.is_dir():
                fallback = self.root / "output_right" / "road"
                self.yolo_dir_right = fallback if fallback.is_dir() else None

    def __len__(self) -> int:
        return self.n_frames

    def iter_frames(self, max_frames: int | None = None, start: int = 0):
        start = max(0, min(start, self.n_frames - 1))
        end = self.n_frames if max_frames is None else min(self.n_frames, start + max_frames)
        depth_cap = cv2.VideoCapture(str(self.root / "depth.mp4"))
        if not depth_cap.isOpened():
            raise RuntimeError(f"could not open {self.root / 'depth.mp4'}")
        left_cap = None
        if self._rgb_left_kind == "video":
            left_cap = cv2.VideoCapture(str(self._rgb_left_items[0]))
            if not left_cap.isOpened():
                depth_cap.release()
                raise RuntimeError(f"could not open {self._rgb_left_items[0]}")
        right_cap = None
        if self.has_right and self._rgb_right_kind == "video":
            right_cap = cv2.VideoCapture(str(self._rgb_right_items[0]))
            if not right_cap.isOpened():
                depth_cap.release()
                if left_cap is not None:
                    left_cap.release()
                raise RuntimeError(f"could not open {self._rgb_right_items[0]}")
        if start > 0:
            depth_cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            if left_cap is not None:
                left_cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            if right_cap is not None:
                right_cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        try:
            for i in range(start, end):
                # ---- Left frame ----
                if self._rgb_left_kind == "images":
                    bgr_left = cv2.imread(str(self._rgb_left_items[i]), cv2.IMREAD_COLOR)
                    if bgr_left is None:
                        raise RuntimeError(f"could not read {self._rgb_left_items[i]}")
                else:
                    ok, bgr_left = left_cap.read()
                    if not ok:
                        break
                ok, depth_bgr = depth_cap.read()
                if not ok:
                    break
                if depth_bgr.shape[:2] != bgr_left.shape[:2]:
                    depth_bgr = cv2.resize(depth_bgr, (bgr_left.shape[1], bgr_left.shape[0]), interpolation=cv2.INTER_NEAREST)
                depth_m = depth_from_turbo(depth_bgr)

                # ---- Right frame (optional) ----
                bgr_right = None
                if self.has_right:
                    if self._rgb_right_kind == "images":
                        bgr_right = cv2.imread(str(self._rgb_right_items[i]), cv2.IMREAD_COLOR)
                        if bgr_right is None:
                            raise RuntimeError(f"could not read {self._rgb_right_items[i]}")
                    else:
                        ok_r, bgr_right = right_cap.read()
                        if not ok_r:
                            bgr_right = None

                T = self._pose_at(i)
                yolo_left = self._yolo_left(i, bgr_left.shape)
                yolo_right = self._yolo_right(i, bgr_right.shape) if bgr_right is not None else None

                yield Frame(i, bgr_left, bgr_right, depth_m, T, yolo_left, yolo_right)
        finally:
            depth_cap.release()
            if left_cap is not None:
                left_cap.release()
            if right_cap is not None:
                right_cap.release()

    def _time_of(self, index: int) -> float:
        alpha = index / max(self.n_frames - 1, 1)
        return float(self._t[0] + alpha * (self._t[-1] - self._t[0]))

    def _pose_at(self, index: int) -> np.ndarray:
        t_query = self._time_of(index)
        j = int(np.searchsorted(self._t, t_query, side="left"))
        if j <= 0:
            return pose_matrix(self._xyz[0], self._quat[0])
        if j >= len(self._t):
            xyz, quat = self._xyz[-1], self._quat[-1]
            t_left = float(self._t[-1])
            if self._imu_t is not None:
                quat = integrate_gyro(quat, self._imu_t, self._imu_gyro, t_left, t_query)
            return pose_matrix(xyz, quat)
        t0, t1 = float(self._t[j - 1]), float(self._t[j])
        u = 0.0 if t1 <= t0 else (t_query - t0) / (t1 - t0)
        xyz = (1.0 - u) * self._xyz[j - 1] + u * self._xyz[j]
        if self._imu_t is not None:
            quat = integrate_gyro(self._quat[j - 1], self._imu_t, self._imu_gyro, t0, t_query)
        else:
            quat = slerp(self._quat[j - 1], self._quat[j], u)
        return pose_matrix(xyz, quat)

    def _yolo_left(self, index: int, shape: tuple[int, ...]) -> np.ndarray | None:
        if self.yolo_dir_left is None:
            return None
        path = self.yolo_dir_left / f"left_{index:06d}.png"
        if not path.is_file():
            return None
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        if mask.shape[:2] != shape[:2]:
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return (mask > 0).astype(np.uint8) * 255

    def _yolo_right(self, index: int, shape: tuple[int, ...]) -> np.ndarray | None:
        if self.yolo_dir_right is None:
            return None
        path = self.yolo_dir_right / f"right_{index:06d}.png"
        if not path.is_file():
            return None
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        if mask.shape[:2] != shape[:2]:
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return (mask > 0).astype(np.uint8) * 255


def _load_tum(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            rows.append([float(x) for x in parts[:8]])
    if len(rows) < 2:
        raise RuntimeError(f"need at least 2 TUM poses in {path}")
    data = np.asarray(rows, dtype=np.float64)
    return data[:, 0], data[:, 1:4], data[:, 4:8]


def _load_imu(path: Path) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if not path.is_file():
        return None, None
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.size < 2:
        return None, None
    t = np.asarray(data["t"], dtype=np.float64).reshape(-1)
    gyro = np.stack([data["wx"], data["wy"], data["wz"]], axis=1).astype(np.float64)
    if t.size < 2:
        return None, None
    return t, gyro
