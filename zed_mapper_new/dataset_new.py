"""Loader for the `allnew` gocart extract (public/gocart_extract/allnew).

Sources, all in per-topic folders holding `data.csv` rows of
`index,timestamp_ns,timestamp_utc,topic,message_json`:

- `zed_left/zed_lean_camera_node/rgb/camera_info` — K for the left camera
- `zed_right/...` — K for the right camera
- `images/left/*.png`, `images/right/*.png` — rectified color; the filename
  (`20260905_071723_285318`) is a UTC wall-clock stamp, so images are paired
  with /odom and with each other by time, not by frame index.
- `odom/data.csv` — nav_msgs/Odometry (odom → base_link), ~6 Hz, epoch stamps.
- `imu/data/data.csv` — sensor_msgs/Imu, ~316 Hz on a zero-based sim clock;
  aligned to the epoch clock with a first/last linear map calibrated against
  the livox IMU. Axes are assumed base-aligned (no imu_link mount exists).
- `livox/imu/data.csv` — 200 Hz IMU on the epoch clock, mount known from
  tf_static, gyro remapped into base axes.
- `tf_static/data.csv` — base_footprint/base_link and the zedxgs camera
  mounts, which give the camera height (ground plane for the IPM, this
  extract has **no depth stream**) and the stereo baseline.

There is no ZED VIO here: pose is /odom, with the selected IMU's gyro filling
heading between (and after the last) odom samples, exactly like the TUM+IMU
pair in the test4/test5 loader.
"""

from __future__ import annotations

import bisect
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from camera import Camera, integrate_gyro, pose_matrix, quat_to_rot, slerp, yaw_from_rot
from ground import Plane

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_ANGULAR_RE = re.compile(
    r'""angular_velocity"": \{""x"": ([-+0-9.eE]+), ""y"": ([-+0-9.eE]+), ""z"": ([-+0-9.eE]+)\}'
)

# zed_mapper's camera_link (x fwd, y left, z up) differs from the URDF
# zedxgs_*_camera_link (x fwd, y right, z down) by 180 degrees about X; the
# mount quaternion in tf_static already carries that flip, so removing it here
# yields a base-aligned forward camera.
_URDF_LINK_FLIP = np.diag([1.0, -1.0, -1.0])

# Fallback if public/dataset/data.yaml is absent (Cityscapes order).
CITYSCAPES_FALLBACK_NAMES = {0: "road", 1: "sidewalk"}


@dataclass
class Frame:
    index: int
    t: float                          # epoch seconds (from the image filename)
    stem: str                         # left image filename stem
    bgr_left: np.ndarray
    bgr_right: np.ndarray | None
    road_left: np.ndarray | None       # uint8 0/255 YOLO road mask
    road_right: np.ndarray | None
    T_odom_cam: np.ndarray            # odom → left camera_link (zed_mapper frame)


def _parse_floats(field) -> list[float]:
    if isinstance(field, list):
        return [float(v) for v in field]
    inner = str(field).strip().lstrip("[").rstrip("]")
    return [float(v) for v in inner.replace("\n", " ").split()]


class AllnewSequence:
    def __init__(
        self,
        root: Path,
        segmented_dir: Path | None = None,
        imu: str = "auto",
        pair_tol: float = 0.25,
    ):
        if imu not in {"auto", "livox", "imu", "none"}:
            raise ValueError(f"unknown imu mode {imu!r}")
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"extract not found: {self.root}")

        left_dir = self.root / "images" / "left"
        right_dir = self.root / "images" / "right"
        if not left_dir.is_dir():
            raise FileNotFoundError(f"no {left_dir}")
        self._left = self._stamped_images(left_dir)
        self._right = self._stamped_images(right_dir) if right_dir.is_dir() else []
        if not self._left:
            raise RuntimeError(f"no images in {left_dir}")
        self.n_frames = len(self._left)
        self.left_count = self.n_frames
        self.right_count = len(self._right)

        self.camera_left = Camera(*self._camera_from_info(
            self.root / "zed_left" / "zed_lean_camera_node" / "rgb" / "camera_info" / "data.csv"
        ))
        right_info = self.root / "zed_right" / "zed_lean_camera_node" / "rgb" / "camera_info" / "data.csv"
        self.camera_right: Camera | None = None
        if right_info.is_file():
            self.camera_right = Camera(*self._camera_from_info(right_info))

        tf = self._load_tf_static(self.root / "tf_static" / "data.csv")
        self._init_mounts(tf)

        self._odom_t, self._odom_xyz, self._odom_quat = self._load_odom(
            self.root / "odom" / "data.csv"
        )
        if len(self._odom_t) < 2:
            raise RuntimeError(f"need at least 2 /odom poses in {self.root / 'odom' / 'data.csv'}")
        self.n_odom = len(self._odom_t)
        self.n_frames_past_odom = int(np.sum(self._left_t() > self._odom_t[-1]))

        self.imu_note = ""
        self.imu_yaw_err_rad = float("nan")
        self.imu_source, self._imu_t, self._imu_gyro = self._select_imu(imu)
        self.n_imu = 0 if self._imu_t is None else len(self._imu_t)

        self.pair_tol = float(pair_tol)
        self._right_pair = self._pair_right(self.pair_tol)

        seg = Path(segmented_dir) if segmented_dir is not None else self.root / "images" / "segmented"
        self.mask_dir_left = seg / "left"
        self.mask_dir_right = seg / "right"
        self.mask_count_left = self._count_masks(self.mask_dir_left)
        self.mask_count_right = self._count_masks(self.mask_dir_right)

    # ---------- discovery / parsing ----------

    @staticmethod
    def _stamped_images(folder: Path) -> list[tuple[float, Path]]:
        out = []
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                t = datetime.strptime(path.stem, "%Y%m%d_%H%M%S_%f").replace(
                    tzinfo=timezone.utc
                ).timestamp()
            except ValueError:
                continue
            out.append((t, path))
        if out:
            out.sort(key=lambda item: item[0])
        return out

    def _left_t(self) -> np.ndarray:
        return np.asarray([t for t, _ in self._left], dtype=np.float64)

    @staticmethod
    def _camera_from_info(path: Path) -> tuple[int, int, float, float, float, float]:
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        with path.open(encoding="utf-8", newline="") as f:
            row = next(csv.DictReader(f))
        msg = json.loads(row["message_json"])
        k = _parse_floats(msg["k"])
        return int(msg["width"]), int(msg["height"]), k[0], k[4], k[2], k[5]

    @staticmethod
    def _load_tf_static(path: Path) -> dict[str, dict]:
        out: dict[str, dict] = {}
        if not path.is_file():
            return out
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                for tfm in json.loads(row["message_json"])["transforms"]:
                    tr = tfm["transform"]
                    out[tfm["child_frame_id"]] = {
                        "parent": tfm["header"]["frame_id"],
                        "t": np.array(
                            [tr["translation"]["x"], tr["translation"]["y"], tr["translation"]["z"]],
                            dtype=np.float64,
                        ),
                        "q": (
                            tr["rotation"]["x"], tr["rotation"]["y"],
                            tr["rotation"]["z"], tr["rotation"]["w"],
                        ),
                    }
        return out

    def _init_mounts(self, tf: dict[str, dict]) -> None:
        def find(pred) -> tuple[str, dict] | None:
            for key in ("zedxgs_left_camera_link", "zedxgs_right_camera_link", "livox_frame"):
                if key in tf and pred(key):
                    return key, tf[key]
            for key, val in tf.items():
                if pred(key):
                    return key, val
            return None

        mount_l = find(lambda k: "left_camera_link" in k)
        mount_r = find(lambda k: "right_camera_link" in k)
        if mount_l is None:
            raise RuntimeError("no left camera mount (base_link -> *left_camera_link) in tf_static")
        if mount_r is None:
            raise RuntimeError("no right camera mount (base_link -> *right_camera_link) in tf_static")
        base = tf.get("base_link")
        self.base_link_height_m = float(base["t"][2]) if base is not None else 0.1

        R_l = quat_to_rot(*mount_l[1]["q"]) @ _URDF_LINK_FLIP
        R_r = quat_to_rot(*mount_r[1]["q"]) @ _URDF_LINK_FLIP
        t_l = np.asarray(mount_l[1]["t"], dtype=np.float64)
        t_r = np.asarray(mount_r[1]["t"], dtype=np.float64)
        for name, R in (("left", R_l), ("right", R_r)):
            if abs(R - np.eye(3)).max() > 0.05:
                print(f"warning: {name} camera mount is not level/forward: R={R.tolist()}")
        self.T_base_caml = np.eye(4)
        self.T_base_caml[:3, :3] = R_l
        self.T_base_caml[:3, 3] = t_l
        self.T_base_camr = np.eye(4)
        self.T_base_camr[:3, :3] = R_r
        self.T_base_camr[:3, 3] = t_r

        # Ground plane (base_footprint z=0) expressed in the left camera_link.
        normal = R_l.T @ np.array([0.0, 0.0, 1.0])
        self.cam_height_m = float(self.base_link_height_m + t_l[2])
        self.plane = Plane(normal, self.cam_height_m)
        # Grid-point shift that maps left-camera coords to right-camera coords.
        delta = R_l.T @ (t_l - t_r)
        self.baseline_offset = (float(delta[0]), float(delta[1]))

        livox = tf.get("livox_frame")
        self._R_base_livox = (
            quat_to_rot(*livox["q"]) if livox is not None else np.eye(3)
        )

    @staticmethod
    def _load_odom(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not path.is_file():
            raise FileNotFoundError(f"missing {path}")
        ts, xyzs, quats = [], [], []
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                msg = json.loads(row["message_json"])
                stamp = msg["header"]["stamp"]
                p = msg["pose"]["pose"]
                pos, ori = p["position"], p["orientation"]
                ts.append(int(stamp["sec"]) + int(stamp["nanosec"]) * 1e-9)
                xyzs.append((pos["x"], pos["y"], pos["z"]))
                quats.append((ori["x"], ori["y"], ori["z"], ori["w"]))
        if not ts:
            raise RuntimeError(f"no /odom messages in {path}")
        t = np.asarray(ts, dtype=np.float64)
        xyz = np.asarray(xyzs, dtype=np.float64)
        quat = np.asarray(quats, dtype=np.float64)
        order = np.argsort(t)
        return t[order], xyz[order], quat[order]

    @staticmethod
    def _load_imu_csv(path: Path) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Fast raw-line parse: timestamp_ns column + angular_velocity regex."""
        if not path.is_file():
            return None, None
        ts, ws = [], []
        with path.open(encoding="utf-8") as f:
            f.readline()  # header
            for line in f:
                parts = line.split(",", 4)
                if len(parts) < 5:
                    continue
                m = _ANGULAR_RE.search(parts[4])
                if m is not None:
                    w = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
                else:
                    txt = parts[4].replace('""', '"').strip().strip('"')
                    av = json.loads(txt)["angular_velocity"]
                    w = (float(av["x"]), float(av["y"]), float(av["z"]))
                ts.append(int(parts[1]) * 1e-9)
                ws.append(w)
        if len(ts) < 10:
            return None, None
        return np.asarray(ts, dtype=np.float64), np.asarray(ws, dtype=np.float64)

    @staticmethod
    def _first_last_stamp_s(path: Path) -> tuple[float, float] | None:
        """First and last timestamp_ns without parsing the whole file."""
        if not path.is_file():
            return None
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
            if row is None:
                return None
            first = int(row["timestamp_ns"]) * 1e-9
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - (1 << 20)))
            tail = f.read().decode("utf-8", "ignore")
        lines = [ln for ln in tail.split("\n") if ln.strip()]
        if not lines:
            return None
        last = int(lines[-1].split(",")[1]) * 1e-9
        return first, last

    # ---------- IMU selection ----------

    def _imu_paths(self) -> dict[str, Path]:
        return {
            "imu": self.root / "imu" / "data" / "data.csv",
            "livox": self.root / "livox" / "imu" / "data.csv",
        }

    def _aligned_imu(self, name: str) -> tuple[np.ndarray, np.ndarray, str]:
        """Load one IMU, map its clock to epoch and its gyro to base axes."""
        paths = self._imu_paths()
        t, w = self._load_imu_csv(paths[name])
        if t is None:
            return None, None, f"{name} csv missing at {paths[name]}"
        note = ""
        if name == "imu":
            livox = self._first_last_stamp_s(paths["livox"])
            if livox is not None:
                lo0, lo1 = livox
                span = t[-1] - t[0]
                a = (lo1 - lo0) / span if span > 0 else 1.0
                t = a * t + (lo0 - a * t[0])
                note = "clock calibrated against livox first/last stamps"
            else:
                t = t + (self._odom_t[0] - t[0])
                note = "clock anchored to first /odom stamp (no livox to calibrate)"
            # imu_link has no mount in tf_static; assume axes = base_link.
            gyro = w
            note += "; axes assumed base-aligned"
        else:
            gyro = w @ self._R_base_livox.T
            note = "epoch clock; gyro rotated into base axes by the livox mount"
        return t, gyro, note

    def _validate_imu(self, t: np.ndarray, gyro: np.ndarray) -> float | None:
        """|gyro-integrated yaw change - /odom yaw change| over the odom span."""
        yaws = np.array(
            [yaw_from_rot(quat_to_rot(*q)) for q in self._odom_quat], dtype=np.float64
        )
        dtheta_odom = float(np.unwrap(yaws)[-1] - yaws[0])
        lo, hi = self._odom_t[0], self._odom_t[-1]
        sel = (t >= lo) & (t <= hi)
        if int(sel.sum()) < 10:
            return None
        dtheta_imu = float(np.trapezoid(gyro[sel, 2], t[sel]))
        return abs(dtheta_imu - dtheta_odom)

    def _select_imu(self, mode: str) -> tuple[str, np.ndarray | None, np.ndarray | None]:
        if mode == "none":
            return "none", None, None
        order = {"imu": ["imu"], "livox": ["livox"], "auto": ["imu", "livox"]}[mode]
        yaws = np.array(
            [yaw_from_rot(quat_to_rot(*q)) for q in self._odom_quat], dtype=np.float64
        )
        dtheta_odom = float(np.unwrap(yaws)[-1] - yaws[0])
        tol = max(0.5, 0.3 * abs(dtheta_odom))

        best: tuple[float, str, np.ndarray, np.ndarray, str] | None = None
        for name in order:
            t, gyro, note = self._aligned_imu(name)
            if t is None:
                print(f"warning: could not load IMU '{name}': {note}")
                continue
            err = self._validate_imu(t, gyro)
            if err is None:
                print(f"warning: IMU '{name}' does not overlap /odom enough to validate")
                continue
            if best is None or err < best[0]:
                best = (err, name, t, gyro, note)
            if mode == "auto" and name == "imu" and err < 0.10:
                break  # good enough; skip parsing the other IMU

        if best is None:
            print("warning: no usable IMU; /odom quaternions will be slerped only")
            self.imu_note = "none loaded; /odom slerp only"
            return "none", None, None
        err, name, t, gyro, note = best
        self.imu_note = note
        if err > tol:
            if mode == "auto":
                print(
                    f"warning: auto IMU check failed (err={err:.2f} rad > tol={tol:.2f} rad); "
                    "using slerp only"
                )
                self.imu_note = "none validated; /odom slerp only"
                return "none", None, None
            print(
                f"warning: IMU '{name}' yaw check failed "
                f"(err={err:.2f} rad > tol={tol:.2f} rad); using it anyway"
            )
        self.imu_yaw_err_rad = err
        return name, t, gyro

    # ---------- pairing / masks / poses ----------

    def _pair_right(self, tol: float) -> list[int | None]:
        rts = [t for t, _ in self._right]
        pairs: list[int | None] = []
        for t, _path in self._left:
            j = bisect.bisect_left(rts, t)
            best, best_dt = None, tol
            for cand in (j - 1, j):
                if 0 <= cand < len(rts):
                    dt = abs(rts[cand] - t)
                    if dt <= best_dt:
                        best, best_dt = cand, dt
            pairs.append(best)
        return pairs

    @staticmethod
    def _count_masks(folder: Path) -> int:
        if not folder.is_dir():
            return 0
        return sum(1 for p in folder.iterdir() if p.suffix.lower() == ".png")

    @staticmethod
    def _load_mask(path: Path, shape: tuple[int, ...]) -> np.ndarray | None:
        if not path.is_file():
            return None
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        if mask.shape[:2] != shape[:2]:
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return (mask > 0).astype(np.uint8) * 255

    def _pose_at(self, t: float) -> np.ndarray:
        """T_odom_base at time t: lerp xyz, slerp or gyro-integrate heading."""
        t_odom = self._odom_t
        j = int(np.searchsorted(t_odom, t, side="left"))
        if j <= 0:
            xyz, quat = self._odom_xyz[0], self._odom_quat[0]
        elif j >= len(t_odom):
            xyz, quat = self._odom_xyz[-1], self._odom_quat[-1]
            if self._imu_t is not None:
                quat = integrate_gyro(quat, self._imu_t, self._imu_gyro, float(t_odom[-1]), t)
        else:
            t0, t1 = float(t_odom[j - 1]), float(t_odom[j])
            u = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
            xyz = (1.0 - u) * self._odom_xyz[j - 1] + u * self._odom_xyz[j]
            if self._imu_t is not None:
                quat = integrate_gyro(self._odom_quat[j - 1], self._imu_t, self._imu_gyro, t0, t)
            else:
                quat = slerp(self._odom_quat[j - 1], self._odom_quat[j], u)
        T = pose_matrix(xyz, quat)
        return T @ self.T_base_caml

    # ---------- iteration ----------

    def __len__(self) -> int:
        return self.n_frames

    @property
    def odom_xy(self) -> np.ndarray:
        """Driven /odom positions (n, 2) — the map window is sized from these."""
        return self._odom_xyz[:, :2]

    def iter_frames(self, max_frames: int | None = None, start: int = 0):
        start = max(0, min(start, self.n_frames - 1))
        end = self.n_frames if max_frames is None else min(self.n_frames, start + max_frames)
        for i in range(start, end):
            t, left_path = self._left[i]
            bgr_left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
            if bgr_left is None:
                raise RuntimeError(f"could not read {left_path}")
            bgr_right = None
            road_right = None
            j = self._right_pair[i]
            if j is not None:
                right_path = self._right[j][1]
                bgr_right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
                if bgr_right is None:
                    bgr_right = None
                else:
                    road_right = self._load_mask(
                        self.mask_dir_right / f"{right_path.stem}.png", bgr_right.shape
                    )
            road_left = self._load_mask(self.mask_dir_left / f"{left_path.stem}.png", bgr_left.shape)
            yield Frame(
                index=i,
                t=float(t),
                stem=left_path.stem,
                bgr_left=bgr_left,
                bgr_right=bgr_right,
                road_left=road_left,
                road_right=road_right,
                T_odom_cam=self._pose_at(float(t)),
            )
