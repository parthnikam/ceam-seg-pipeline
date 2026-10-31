#!/usr/bin/env python3
"""Visualize colored RGB-D PLY point clouds from a gocart extract."""

from __future__ import annotations

import argparse
import glob
import os
import sys

# Open3D 0.19's GLFW/GLEW path cannot create a GL window on native Wayland.
# Drop onto X11/XWayland before importing open3d so GLEW can initialize.
os.environ["GDK_BACKEND"] = "x11"
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["GLFW_PLATFORM"] = "x11"
os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ.pop("WAYLAND_DISPLAY", None)

import open3d as o3d

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACT_ROOT = os.path.dirname(SCRIPT_DIR)

# GLFW key codes
KEY_SPACE = 32
KEY_N = ord("N")
KEY_P = ord("P")
KEY_Q = ord("Q")
KEY_R = ord("R")
KEY_ESCAPE = 256
KEY_RIGHT = 262
KEY_LEFT = 263


def resolve_folder(dataset: str) -> str:
    if os.path.isdir(dataset):
        return os.path.abspath(dataset)
    return os.path.abspath(os.path.join(EXTRACT_ROOT, dataset, "pointcloud"))


def load_cloud(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise RuntimeError(f"empty cloud: {path}")
    return pcd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="?",
        default="test4",
        help="Extract name (test4, test5) or a folder of .ply files. Default: test4",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start at this 0-based index in the sorted PLY list.",
    )
    args = parser.parse_args()

    folder_path = resolve_folder(args.dataset)
    ply_files = sorted(glob.glob(os.path.join(folder_path, "*.ply")))
    if not ply_files:
        print(f"No .ply files in {folder_path}", file=sys.stderr)
        sys.exit(1)

    idx = max(0, min(args.start, len(ply_files) - 1))
    state = {"idx": idx, "pcd": load_cloud(ply_files[idx])}

    vis = o3d.visualization.VisualizerWithKeyCallback()
    if not vis.create_window(
        window_name=f"{os.path.basename(ply_files[idx])}  [{idx + 1}/{len(ply_files)}]",
        width=1280,
        height=800,
    ):
        print(
            "Failed to create an OpenGL window. "
            "Need a working X11/XWayland display (DISPLAY is set, GPU GL works).",
            file=sys.stderr,
        )
        sys.exit(1)

    vis.add_geometry(state["pcd"], reset_bounding_box=True)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = (0.05, 0.05, 0.05)

    def show(reset_view: bool) -> None:
        i = state["idx"]
        path = ply_files[i]
        print(f"[{i + 1}/{len(ply_files)}] {os.path.basename(path)}")
        state["pcd"] = load_cloud(path)
        vis.clear_geometries()
        vis.add_geometry(state["pcd"], reset_bounding_box=reset_view)
        vis.update_renderer()

    def step(delta: int) -> bool:
        nxt = (state["idx"] + delta) % len(ply_files)
        if nxt == state["idx"]:
            return False
        state["idx"] = nxt
        show(reset_view=False)
        return False

    def next_cloud(_vis) -> bool:
        return step(1)

    def prev_cloud(_vis) -> bool:
        return step(-1)

    def reset_view(vis_ref) -> bool:
        vis_ref.reset_view_point(True)
        return False

    def quit_viewer(vis_ref) -> bool:
        vis_ref.close()
        return False

    print(f"Found {len(ply_files)} .ply files in {folder_path}")
    print("Keys: N / Right / Space = next,  P / Left = prev,  R = reset view,  Q / Esc = quit")
    print(f"[{idx + 1}/{len(ply_files)}] {os.path.basename(ply_files[idx])}")

    for key in (KEY_N, KEY_RIGHT, KEY_SPACE):
        vis.register_key_callback(key, next_cloud)
    for key in (KEY_P, KEY_LEFT):
        vis.register_key_callback(key, prev_cloud)
    vis.register_key_callback(KEY_R, reset_view)
    vis.register_key_callback(KEY_Q, quit_viewer)
    vis.register_key_callback(KEY_ESCAPE, quit_viewer)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
