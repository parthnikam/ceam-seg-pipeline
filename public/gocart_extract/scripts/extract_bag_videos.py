#!/usr/bin/env python3
"""Extract ZED left/right color streams from a rosbag2 sqlite bag to H.264 mp4."""
from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
import time

import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image

FFMPEG = "/home/parth/miniconda3/bin/ffmpeg"
LEFT_TOPIC = "/zedx/zed_node/left/color/rect/image"
RIGHT_TOPIC = "/zedx/zed_node/right/color/rect/image"
PIX_FMT = {
    "bgra8": "bgra",
    "rgba8": "rgba",
    "bgr8": "bgr24",
    "rgb8": "rgb24",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def image_to_packed(msg: Image) -> bytes:
    nchan = {"bgra8": 4, "rgba8": 4, "bgr8": 3, "rgb8": 3}[msg.encoding]
    bpp = nchan
    expected_step = msg.width * bpp
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.step == expected_step:
        return buf[: msg.height * expected_step].tobytes()
    row_major = buf.reshape(msg.height, msg.step)
    return np.ascontiguousarray(row_major[:, :expected_step]).tobytes()


def start_ffmpeg(path: str, width: int, height: int, fps: float, pix_fmt: str) -> subprocess.Popen:
    cmd = [
        FFMPEG,
        "-hide_banner",
        "-loglevel",
        "error",
        "-stats",
        "-y",
        "-vaapi_device",
        "/dev/dri/renderD128",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-vf",
        "format=nv12,hwupload",
        "-c:v",
        "h264_vaapi",
        "-qp",
        "18",
        "-movflags",
        "+faststart",
        path,
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def extract_bag(
    db_path: str,
    out_dir: str,
    prefix: str,
    duration_s: float,
    counts: dict[str, int],
    max_frames: int = 0,
) -> None:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    topics = {name: tid for tid, name in con.execute("SELECT id, name FROM topics")}
    wanted = {}
    for name in (LEFT_TOPIC, RIGHT_TOPIC):
        if name not in topics:
            raise SystemExit(f"{prefix}: missing topic {name}")
        wanted[topics[name]] = name

    procs: dict[str, subprocess.Popen] = {}
    written = {LEFT_TOPIC: 0, RIGHT_TOPIC: 0}
    meta = {}
    t0 = time.time()
    ids = tuple(wanted.keys())
    placeholders = ",".join("?" * len(ids))
    query = (
        f"SELECT topic_id, timestamp, data FROM messages "
        f"WHERE topic_id IN ({placeholders}) ORDER BY timestamp"
    )
    log(f"{prefix}: scanning {db_path}")
    for topic_id, ts, data in con.execute(query, ids):
        topic = wanted[topic_id]
        msg = deserialize_message(data, Image)
        enc = msg.encoding
        if enc not in PIX_FMT:
            raise SystemExit(f"{prefix}: unsupported encoding {enc} on {topic}")
        if topic not in meta:
            fps = counts[topic] / duration_s if duration_s > 0 else 4.0
            out = os.path.join(
                out_dir,
                f"{prefix}_{'left' if topic == LEFT_TOPIC else 'right'}.mp4",
            )
            log(
                f"{prefix}: {topic} {msg.width}x{msg.height} {enc} "
                f"{counts[topic]} frames @ {fps:.3f} fps -> {out}"
            )
            proc = start_ffmpeg(out, msg.width, msg.height, fps, PIX_FMT[enc])
            procs[topic] = proc
            meta[topic] = (msg.width, msg.height, enc, out)
        w, h, e, _out = meta[topic]
        if msg.width != w or msg.height != h or msg.encoding != e:
            raise SystemExit(f"{prefix}: frame size/encoding changed on {topic}")
        packed = image_to_packed(msg)
        try:
            procs[topic].stdin.write(packed)
        except BrokenPipeError as exc:
            err = procs[topic].stderr.read().decode(errors="replace")[-1000:]
            raise SystemExit(f"{prefix}: ffmpeg died for {topic}: {err}") from exc
        written[topic] += 1
        total = written[LEFT_TOPIC] + written[RIGHT_TOPIC]
        if max_frames and written[LEFT_TOPIC] >= max_frames and written[RIGHT_TOPIC] >= max_frames:
            log(f"{prefix}: reached --max-frames {max_frames}")
            break
        if total % 50 == 0:
            elapsed = time.time() - t0
            log(
                f"{prefix}: left={written[LEFT_TOPIC]}/{counts[LEFT_TOPIC]} "
                f"right={written[RIGHT_TOPIC]}/{counts[RIGHT_TOPIC]} "
                f"elapsed={elapsed:.1f}s"
            )

    for topic, proc in procs.items():
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        rc = proc.wait()
        _, _, _, out = meta[topic]
        size = os.path.getsize(out) if os.path.exists(out) else 0
        log(f"{prefix}: {topic} ffmpeg rc={rc} size={size} bytes {err.strip()}")
        if rc != 0 or size < 1000:
            raise SystemExit(f"{prefix}: encode failed for {topic}")
        if written[topic] != counts[topic]:
            log(
                f"{prefix}: WARNING wrote {written[topic]} frames, "
                f"metadata said {counts[topic]}"
            )
    con.close()
    log(f"{prefix}: done in {time.time() - t0:.1f}s")


def parse_metadata(bag_dir: str) -> tuple[float, dict[str, int]]:
    import yaml

    with open(os.path.join(bag_dir, "metadata.yaml"), encoding="utf-8") as f:
        info = yaml.safe_load(f)["rosbag2_bagfile_information"]
    duration_s = info["duration"]["nanoseconds"] / 1e9
    counts = {}
    for t in info["topics_with_message_count"]:
        counts[t["topic_metadata"]["name"]] = t["message_count"]
    rel = info["relative_file_paths"][0]
    db = os.path.join(bag_dir, rel)
    return duration_s, counts, db


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bag_dir")
    ap.add_argument("out_dir")
    ap.add_argument("prefix")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    duration_s, counts, db = parse_metadata(args.bag_dir)
    extract_bag(db, args.out_dir, args.prefix, duration_s, counts, args.max_frames)


if __name__ == "__main__":
    main()
