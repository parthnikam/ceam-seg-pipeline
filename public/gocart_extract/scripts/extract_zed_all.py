#!/usr/bin/env python3
"""Extract every ZED-related stream that is actually in these rosbag2 files.

The bags do not contain confidence, disparity, native point cloud, unrectified
images, or object tracking. Those SDK outputs were never recorded.
This writes: colorized depth video, stereo SBS, RGB-D SBS, IMU CSV, VIO pose,
camera calibration, static extrinsics, and 1 Hz colored point clouds from RGB-D.
"""
from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import subprocess
import time
from typing import Optional

import cv2
import numpy as np
import yaml
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo, Image, Imu
from tf2_msgs.msg import TFMessage

FFMPEG = "/home/parth/miniconda3/bin/ffmpeg"
LEFT = "/zedx/zed_node/left/color/rect/image"
RIGHT = "/zedx/zed_node/right/color/rect/image"
DEPTH = "/zedx/zed_node/depth/depth_registered"
IMU = "/zedx/zed_node/imu/data"
TF = "/tf"
TF_STATIC = "/tf_static"
LEFT_INFO = "/zedx/zed_node/left/color/rect/image/camera_info"
RIGHT_INFO = "/zedx/zed_node/right/color/rect/image/camera_info"
DEPTH_INFO = "/zedx/zed_node/depth/depth_registered/camera_info"
DEPTH_INFO_ALT = "/zedx/zed_node/depth/camera_info"
PAIR_NS = 50_000_000  # 50 ms
DEPTH_MAX_M = 15.0
PC_STRIDE = 4
PC_EVERY = 4  # every 4th RGB-D pair ≈ 1 Hz at ~4 fps


def log(msg: str) -> None:
    print(msg, flush=True)


def start_ffmpeg(path: str, width: int, height: int, fps: float) -> subprocess.Popen:
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error", "-stats", "-y",
        "-vaapi_device", "/dev/dri/renderD128",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", f"{fps:.6f}", "-i", "-",
        "-vf", "format=nv12,hwupload",
        "-c:v", "h264_vaapi", "-qp", "18",
        "-movflags", "+faststart", path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def close_ffmpeg(proc: subprocess.Popen, path: str, label: str) -> None:
    proc.stdin.close()
    err = proc.stderr.read().decode(errors="replace")
    rc = proc.wait()
    size = os.path.getsize(path) if os.path.exists(path) else 0
    log(f"{label}: ffmpeg rc={rc} size={size} {err.strip()[-200:]}")
    if rc != 0 or size < 1000:
        raise SystemExit(f"{label}: encode failed {path}")


def bgr_from_color(msg: Image) -> np.ndarray:
    row = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    if msg.encoding == "bgra8":
        img = row[:, : msg.width * 4].reshape(msg.height, msg.width, 4)
        return np.ascontiguousarray(img[:, :, :3])
    if msg.encoding == "bgr8":
        return np.ascontiguousarray(row[:, : msg.width * 3].reshape(msg.height, msg.width, 3))
    raise SystemExit(f"unsupported color encoding {msg.encoding}")


def depth_from_msg(msg: Image) -> np.ndarray:
    row = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.step // 4)
    return row[:, : msg.width]


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0.0) & (depth <= DEPTH_MAX_M)
    u8 = np.zeros(depth.shape, dtype=np.uint8)
    u8[valid] = np.clip(depth[valid] / DEPTH_MAX_M * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    bgr[~valid] = 0
    return bgr


def write_ply_xyzrgb(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    n = xyz.shape[0]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    packed = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                ("r", "u1"), ("g", "u1"), ("b", "u1")])
    packed["x"] = xyz[:, 0]
    packed["y"] = xyz[:, 1]
    packed["z"] = xyz[:, 2]
    packed["r"] = rgb[:, 0]
    packed["g"] = rgb[:, 1]
    packed["b"] = rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        packed.tofile(f)


def unproject_rgbd(depth: np.ndarray, bgr: np.ndarray, K: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    d = depth[::stride, ::stride]
    color = bgr[::stride, ::stride]
    h, w = d.shape
    us, vs = np.meshgrid(np.arange(0, depth.shape[1], stride),
                         np.arange(0, depth.shape[0], stride))
    valid = np.isfinite(d) & (d > 0.2) & (d <= DEPTH_MAX_M)
    z = d[valid]
    x = (us[valid] - cx) * z / fx
    y = (vs[valid] - cy) * z / fy
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    # bgr -> rgb
    rgb = color[valid][:, ::-1].astype(np.uint8)
    return xyz, rgb


def info_to_dict(msg: CameraInfo) -> dict:
    return {
        "frame_id": str(msg.header.frame_id),
        "width": int(msg.width),
        "height": int(msg.height),
        "distortion_model": str(msg.distortion_model),
        "D": [float(x) for x in msg.d],
        "K": [float(x) for x in msg.k],
        "R": [float(x) for x in msg.r],
        "P": [float(x) for x in msg.p],
    }


def stamp_sec(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


class FfmpegOut:
    def __init__(self, path: str, fps: float):
        self.path = path
        self.fps = fps
        self.proc: Optional[subprocess.Popen] = None
        self.wh = None
        self.n = 0

    def write(self, bgr: np.ndarray) -> None:
        h, w, _ = bgr.shape
        if self.proc is None:
            self.proc = start_ffmpeg(self.path, w, h, self.fps)
            self.wh = (w, h)
        if (w, h) != self.wh:
            raise SystemExit(f"size change {self.path}: {self.wh} -> {(w, h)}")
        try:
            self.proc.stdin.write(np.ascontiguousarray(bgr).tobytes())
        except BrokenPipeError as exc:
            err = self.proc.stderr.read().decode(errors="replace")[-800:]
            raise SystemExit(f"ffmpeg died {self.path}: {err}") from exc
        self.n += 1

    def close(self, label: str) -> None:
        if self.proc is None:
            log(f"{label}: no frames for {self.path}")
            return
        close_ffmpeg(self.proc, self.path, label)


def extract(bag_dir: str, out_dir: str, prefix: str, duration_s: float, counts: dict, max_frames: int) -> None:
    meta_path = os.path.join(bag_dir, "metadata.yaml")
    with open(meta_path, encoding="utf-8") as f:
        info = yaml.safe_load(f)["rosbag2_bagfile_information"]
    db = os.path.join(bag_dir, info["relative_file_paths"][0])
    os.makedirs(out_dir, exist_ok=True)
    pc_dir = os.path.join(out_dir, f"{prefix}_pointcloud")
    os.makedirs(pc_dir, exist_ok=True)

    fps_depth = max(counts.get(DEPTH, 1), 1) / duration_s
    fps_left = max(counts.get(LEFT, 1), 1) / duration_s
    log(f"{prefix}: depth_fps={fps_depth:.3f} left_fps={fps_left:.3f}")

    depth_v = FfmpegOut(os.path.join(out_dir, f"{prefix}_depth.mp4"), fps_depth)
    stereo_v = FfmpegOut(os.path.join(out_dir, f"{prefix}_stereo.mp4"), fps_left)
    rgbd_v = FfmpegOut(os.path.join(out_dir, f"{prefix}_rgbd.mp4"), fps_left)

    imu_path = os.path.join(out_dir, f"{prefix}_imu.csv")
    pose_path = os.path.join(out_dir, f"{prefix}_pose_tum.txt")
    imu_f = open(imu_path, "w", newline="")
    imu_w = csv.writer(imu_f)
    imu_w.writerow(["t", "qx", "qy", "qz", "qw", "wx", "wy", "wz", "ax", "ay", "az"])
    pose_f = open(pose_path, "w")
    pose_f.write("# TUM: t tx ty tz qx qy qz qw  (odom -> zedx_camera_link)\n")

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    name_to_id = {n: i for i, n in con.execute("SELECT id, name FROM topics")}
    wanted_names = [LEFT, RIGHT, DEPTH, IMU, TF, TF_STATIC, LEFT_INFO, RIGHT_INFO]
    if DEPTH_INFO in name_to_id:
        wanted_names.append(DEPTH_INFO)
    elif DEPTH_INFO_ALT in name_to_id:
        wanted_names.append(DEPTH_INFO_ALT)
    missing = [n for n in (LEFT, RIGHT, DEPTH, IMU, TF) if n not in name_to_id]
    if missing:
        raise SystemExit(f"{prefix}: missing {missing}")
    id_to_name = {name_to_id[n]: n for n in wanted_names if n in name_to_id}
    ids = tuple(id_to_name.keys())
    q = f"SELECT topic_id, timestamp, data FROM messages WHERE topic_id IN ({','.join('?'*len(ids))}) ORDER BY timestamp"

    last_left = None  # (t_ns, bgr)
    last_right = None
    last_depth = None  # (t_ns, depth, bgr_colorized)
    stereo_left_used = set()
    rgbd_left_used = set()
    calib = {}
    tf_static_list = []
    K = None
    n_imu = n_pose = n_pc = 0
    n_img = {LEFT: 0, RIGHT: 0, DEPTH: 0}
    t0 = time.time()

    log(f"{prefix}: scanning {db}")
    for topic_id, t_ns, data in con.execute(q, ids):
        name = id_to_name[topic_id]
        if name == IMU:
            msg = deserialize_message(data, Imu)
            imu_w.writerow([
                f"{stamp_sec(msg.header):.9f}",
                msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w,
                msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
                msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
            ])
            n_imu += 1
            continue
        if name == TF:
            msg = deserialize_message(data, TFMessage)
            for tr in msg.transforms:
                if tr.header.frame_id.lstrip("/") == "odom" and tr.child_frame_id.lstrip("/") == "zedx_camera_link":
                    t = tr.transform.translation
                    r = tr.transform.rotation
                    pose_f.write(
                        f"{stamp_sec(tr.header):.9f} {t.x:.6f} {t.y:.6f} {t.z:.6f} "
                        f"{r.x:.6f} {r.y:.6f} {r.z:.6f} {r.w:.6f}\n"
                    )
                    n_pose += 1
            continue
        if name == TF_STATIC:
            msg = deserialize_message(data, TFMessage)
            for tr in msg.transforms:
                t = tr.transform.translation
                r = tr.transform.rotation
                tf_static_list.append({
                    "parent": str(tr.header.frame_id),
                    "child": str(tr.child_frame_id),
                    "translation": [float(t.x), float(t.y), float(t.z)],
                    "rotation_xyzw": [float(r.x), float(r.y), float(r.z), float(r.w)],
                })
            continue
        if name in (LEFT_INFO, RIGHT_INFO, DEPTH_INFO, DEPTH_INFO_ALT):
            if name not in calib:
                ci = deserialize_message(data, CameraInfo)
                calib[name] = info_to_dict(ci)
                if name in (LEFT_INFO, DEPTH_INFO, DEPTH_INFO_ALT) and K is None:
                    K = np.array(ci.k, dtype=np.float64).reshape(3, 3)
            continue

        msg = deserialize_message(data, Image)
        if name == LEFT:
            bgr = bgr_from_color(msg)
            last_left = (t_ns, bgr)
            n_img[LEFT] += 1
        elif name == RIGHT:
            bgr = bgr_from_color(msg)
            last_right = (t_ns, bgr)
            n_img[RIGHT] += 1
        else:
            depth = depth_from_msg(msg)
            dvis = colorize_depth(depth)
            last_depth = (t_ns, depth, dvis)
            depth_v.write(dvis)
            n_img[DEPTH] += 1

        if last_left and last_right:
            tl, bl = last_left
            tr, br = last_right
            if abs(tl - tr) <= PAIR_NS and tl not in stereo_left_used:
                if bl.shape == br.shape:
                    stereo_v.write(np.concatenate([bl, br], axis=1))
                    stereo_left_used.add(tl)

        if last_left and last_depth:
            tl, bl = last_left
            td, depth, dvis = last_depth
            if abs(tl - td) <= PAIR_NS and tl not in rgbd_left_used:
                if bl.shape[:2] == dvis.shape[:2]:
                    rgbd_v.write(np.concatenate([bl, dvis], axis=1))
                    rgbd_left_used.add(tl)
                    if K is not None and (len(rgbd_left_used) % PC_EVERY == 1):
                        xyz, rgb = unproject_rgbd(depth, bl, K, PC_STRIDE)
                        ply = os.path.join(pc_dir, f"{len(rgbd_left_used)-1:05d}.ply")
                        write_ply_xyzrgb(ply, xyz, rgb)
                        n_pc += 1

        if max_frames and n_img[DEPTH] >= max_frames and n_img[LEFT] >= max_frames:
            log(f"{prefix}: reached --max-frames {max_frames}")
            break
        total = n_img[LEFT] + n_img[RIGHT] + n_img[DEPTH]
        if total and total % 60 == 0:
            log(
                f"{prefix}: L={n_img[LEFT]} R={n_img[RIGHT]} D={n_img[DEPTH]} "
                f"stereo={stereo_v.n} rgbd={rgbd_v.n} pc={n_pc} imu={n_imu} pose={n_pose} "
                f"{time.time()-t0:.1f}s"
            )

    imu_f.close()
    pose_f.close()
    depth_v.close(f"{prefix} depth")
    stereo_v.close(f"{prefix} stereo")
    rgbd_v.close(f"{prefix} rgbd")

    calib_path = os.path.join(out_dir, f"{prefix}_calib.yaml")
    with open(calib_path, "w", encoding="utf-8") as f:
        yaml.safe_dump({"camera_info": calib, "tf_static": tf_static_list}, f, sort_keys=False)

    log(
        f"{prefix}: wrote depth={depth_v.n} stereo={stereo_v.n} rgbd={rgbd_v.n} "
        f"imu={n_imu} pose={n_pose} pointclouds={n_pc} calib={calib_path}"
    )
    log(f"{prefix}: done in {time.time()-t0:.1f}s")
    con.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bag_dir")
    ap.add_argument("out_dir")
    ap.add_argument("prefix")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    with open(os.path.join(args.bag_dir, "metadata.yaml"), encoding="utf-8") as f:
        info = yaml.safe_load(f)["rosbag2_bagfile_information"]
    duration_s = info["duration"]["nanoseconds"] / 1e9
    counts = {t["topic_metadata"]["name"]: t["message_count"] for t in info["topics_with_message_count"]}
    extract(args.bag_dir, args.out_dir, args.prefix, duration_s, counts, args.max_frames)


if __name__ == "__main__":
    main()
