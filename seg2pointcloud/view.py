#!/usr/bin/env python3
"""Interactive 3D viewer for pycolmap Structure-from-Motion reconstructions.

Loads the sparse reconstruction produced by seg2pointcloud/sfm.py and opens an
interactive 3D window showing:
  - Reconstructed 3D landmarks (with RGB colors or height/error coloring)
  - Left and right ZED X GS camera frustums and poses
  - Left and right camera motion trajectories

Supported interactive backends (--backend):
  - auto (default): tries Open3D desktop window -> Viser -> self-contained Three.js browser viewer
  - open3d: native OpenGL desktop window via Open3D
  - viser: interactive web viewer via Viser (http://localhost:8080)
  - web: zero-dependency interactive Three.js WebGL window (writes viewer.html & opens browser)
  - matplotlib: interactive Matplotlib 3D window

Usage:
    python seg2pointcloud/view_sfm.py --sparse seg2pointcloud/out/sparse
    python seg2pointcloud/view_sfm.py --sparse seg2pointcloud/out/sparse --backend web
    python seg2pointcloud/view_sfm.py --sparse seg2pointcloud/out/sparse --backend open3d
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pycolmap

NUM_THREADS = max(1, os.cpu_count() or 1)


@dataclass
class CameraPoseData:
    """Extracted pose and intrinsics for a single registered image."""

    image_id: int
    name: str
    side: str  # "left" or "right"
    width: int
    height: int
    K: np.ndarray  # (3, 3) calibration matrix
    R_cam_from_world: np.ndarray  # (3, 3)
    t_cam_from_world: np.ndarray  # (3,)
    center_world: np.ndarray  # (3,) camera center in world coordinates
    forward_world: np.ndarray  # (3,) principal axis in world coordinates


def find_best_reconstruction(
    sparse_path: Path, model_idx: int | None = None
) -> tuple[Path, pycolmap.Reconstruction]:
    """Load the specified or largest sub-model from a COLMAP sparse directory."""
    if not sparse_path.exists():
        raise FileNotFoundError(f"Sparse path not found: {sparse_path}")

    # Direct reconstruction directory (contains cameras.bin / cameras.txt)
    if any(
        (sparse_path / fname).exists()
        for fname in ("cameras.bin", "cameras.txt", "images.bin", "images.txt")
    ):
        return sparse_path, pycolmap.Reconstruction(str(sparse_path))

    if model_idx is not None:
        model_dir = sparse_path / str(model_idx)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Reconstruction sub-model not found: {model_dir}")
        return model_dir, pycolmap.Reconstruction(str(model_dir))

    # Search numbered subdirectories (0, 1, ...) and pick the one with most registered images
    candidates: list[tuple[int, Path, pycolmap.Reconstruction]] = []
    for sub in sorted(sparse_path.iterdir()):
        if not sub.is_dir():
            continue
        try:
            rec = pycolmap.Reconstruction(str(sub))
            candidates.append((rec.num_reg_images(), sub, rec))
        except Exception:
            continue

    if not candidates:
        raise FileNotFoundError(
            f"No valid COLMAP reconstructions found inside {sparse_path}"
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, best_dir, best_rec = candidates[0]
    return best_dir, best_rec


def _get_cam_from_world(image: pycolmap.Image) -> pycolmap.Rigid3d:
    """Retrieve cam_from_world Rigid3d across pycolmap 3.x and 4.x."""
    cfw = image.cam_from_world
    return cfw() if callable(cfw) else cfw


def extract_scene_data(
    rec: pycolmap.Reconstruction,
    max_reproj_error: float = 4.0,
    min_track_length: int = 2,
    clip_percentile: float = 98.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[CameraPoseData]]:
    """Extract filtered 3D points, colors, errors, and sorted camera poses."""
    points = list(rec.points3D.values())

    if points:
        # Vectorized filtering and stacking (much faster than per-point Python ops)
        def _track_len(pt) -> int:
            track_len = pt.track.length() if callable(getattr(pt.track, "length", None)) else len(pt.track.elements)
            return int(track_len)

        xyz = np.stack([np.asarray(pt.xyz, dtype=np.float64) for pt in points])
        rgb = np.stack([np.asarray(pt.color, dtype=np.uint8) for pt in points])
        errors = np.fromiter((float(pt.error) for pt in points), dtype=np.float64, count=len(points))
        track_lens = np.fromiter((_track_len(pt) for pt in points), dtype=np.int64, count=len(points))

        mask = (errors <= max_reproj_error) & (track_lens >= min_track_length)

        # Filter extreme spatial outliers so the interactive camera centers cleanly
        if 0.0 < clip_percentile < 100.0 and int(mask.sum()) > 20:
            xyz_f = xyz[mask]
            median = np.median(xyz_f, axis=0)
            dists = np.linalg.norm(xyz_f - median, axis=1)
            cutoff = np.percentile(dists, clip_percentile)
            mask_idx = np.flatnonzero(mask)
            mask_idx = mask_idx[dists <= max(cutoff, 1e-6)]
            mask = np.zeros_like(mask)
            mask[mask_idx] = True

        xyz = xyz[mask]
        rgb = rgb[mask]
        errors = errors[mask]
    else:
        xyz = np.zeros((0, 3), dtype=np.float64)
        rgb = np.zeros((0, 3), dtype=np.uint8)
        errors = np.zeros((0,), dtype=np.float64)

    poses: list[CameraPoseData] = []
    for image in rec.images.values():
        has_pose = image.has_pose() if callable(getattr(image, "has_pose", None)) else getattr(image, "has_pose", True)
        if not has_pose:
            continue

        cam = rec.cameras[image.camera_id]
        K = np.asarray(cam.calibration_matrix(), dtype=np.float64)
        cfw = _get_cam_from_world(image)
        R = np.asarray(cfw.rotation.matrix(), dtype=np.float64)
        t = np.asarray(cfw.translation, dtype=np.float64).reshape(3)
        center = np.asarray(image.projection_center(), dtype=np.float64).reshape(3)
        forward = np.asarray(image.viewing_direction(), dtype=np.float64).reshape(3)
        side = "left" if image.name.startswith(("left_", "left/")) else "right"

        poses.append(
            CameraPoseData(
                image_id=int(image.image_id),
                name=image.name,
                side=side,
                width=int(cam.width),
                height=int(cam.height),
                K=K,
                R_cam_from_world=R,
                t_cam_from_world=t,
                center_world=center,
                forward_world=forward,
            )
        )

    poses.sort(key=lambda p: p.name)
    return xyz, rgb, errors, poses


def estimate_frustum_scale(xyz: np.ndarray, poses: list[CameraPoseData], user_scale: float | None) -> float:
    """Compute a sensible default camera frustum size from the scene extent."""
    if user_scale is not None and user_scale > 0:
        return user_scale
    if len(poses) >= 2:
        centers = np.vstack([p.center_world for p in poses])
        span = float(np.linalg.norm(centers.max(axis=0) - centers.min(axis=0)))
        if span > 1e-4:
            return max(span * 0.04, 0.05)
    if len(xyz) >= 2:
        span = float(np.linalg.norm(np.percentile(xyz, 90, axis=0) - np.percentile(xyz, 10, axis=0)))
        if span > 1e-4:
            return max(span * 0.025, 0.05)
    return 0.2


def view_with_open3d(
    xyz: np.ndarray,
    rgb: np.ndarray,
    poses: list[CameraPoseData],
    frustum_scale: float,
    point_size: float,
) -> None:
    """Open a native desktop Open3D visualizer window."""
    import open3d as o3d

    geometries: list[o3d.geometry.Geometry] = []

    if len(xyz) > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)
        geometries.append(pcd)

    side_colors = {
        "left": [0.15, 0.55, 1.0],   # Blue for left camera
        "right": [1.0, 0.40, 0.15],  # Orange for right camera
    }

    def _build_frustum(p):
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = p.R_cam_from_world
        extrinsic[:3, 3] = p.t_cam_from_world
        frustum = o3d.geometry.LineSet.create_camera_visualization(
            p.width, p.height, p.K, extrinsic, scale=frustum_scale
        )
        frustum.paint_uniform_color(side_colors[p.side])
        return frustum

    # Build frustum geometries in parallel (C++ geometry creation releases the GIL)
    all_frustums: list[o3d.geometry.LineSet] = []
    try:
        with ThreadPoolExecutor(max_workers=min(NUM_THREADS, max(1, len(poses)))) as pool:
            all_frustums = list(pool.map(_build_frustum, poses))
    except Exception:
        all_frustums = [_build_frustum(p) for p in poses]
    geometries.extend(all_frustums)

    for side in ("left", "right"):
        side_poses = [p for p in poses if p.side == side]
        color = side_colors[side]

        if len(side_poses) >= 2:
            centers = np.vstack([p.center_world for p in side_poses])
            lines = [[i, i + 1] for i in range(len(side_poses) - 1)]
            traj = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(centers),
                lines=o3d.utility.Vector2iVector(lines),
            )
            traj.paint_uniform_color(color)
            geometries.append(traj)

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frustum_scale * 2.0)
    geometries.append(axes)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="COLMAP SfM Reconstruction (Left=Blue, Right=Orange)", width=1440, height=900)
    for g in geometries:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    if opt is not None:
        opt.background_color = np.asarray([0.07, 0.08, 0.10])
        opt.point_size = float(point_size)

    print("Opening Open3D interactive window (press 'Q' or close window to exit)...")
    vis.run()
    vis.destroy_window()


def view_with_viser(
    xyz: np.ndarray,
    rgb: np.ndarray,
    poses: list[CameraPoseData],
    frustum_scale: float,
    point_size: float,
    port: int = 8080,
) -> None:
    """Start an interactive Viser 3D web server."""
    import time
    import viser
    import viser.transforms as vtf

    server = viser.ViserServer(port=port)
    print(f"Viser interactive server running at http://localhost:{port} (Ctrl+C to stop)")

    if len(xyz) > 0:
        server.scene.add_point_cloud(
            name="/landmarks",
            points=xyz.astype(np.float32),
            colors=rgb,
            point_size=float(point_size * 0.01),
        )

    side_colors = {
        "left": (38, 140, 255),
        "right": (255, 102, 38),
    }

    for p in poses:
        T_world_from_cam = np.eye(4, dtype=np.float64)
        T_world_from_cam[:3, :3] = p.R_cam_from_world.T
        T_world_from_cam[:3, 3] = p.center_world
        se3 = vtf.SE3.from_matrix(T_world_from_cam)
        fov = float(2.0 * np.arctan2(p.height / 2.0, p.K[1, 1]))
        aspect = float(p.width / max(p.height, 1))
        server.scene.add_camera_frustum(
            name=f"/cameras/{p.side}/{p.name}",
            fov=fov,
            aspect=aspect,
            scale=frustum_scale,
            color=side_colors[p.side],
            wxyz=se3.rotation().wxyz,
            position=se3.translation(),
        )

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping Viser server.")


def build_web_viewer_html(
    xyz: np.ndarray,
    rgb: np.ndarray,
    errors: np.ndarray,
    poses: list[CameraPoseData],
    frustum_scale: float,
    point_size: float,
    output_html: Path,
    open_browser: bool = True,
) -> Path:
    """Write a self-contained interactive Three.js HTML viewer and open it in the browser."""
    if poses:
        # Vectorized frustum corner computation for all cameras at once
        fxs = np.array([p.K[0, 0] for p in poses])
        fys = np.array([p.K[1, 1] for p in poses])
        cxs = np.array([p.K[0, 2] for p in poses])
        cys = np.array([p.K[1, 2] for p in poses])
        ws = np.array([float(p.width) for p in poses])
        hs = np.array([float(p.height) for p in poses])

        xs = np.array([0.0, 1.0, 1.0, 0.0])
        ys = np.array([0.0, 0.0, 1.0, 1.0])
        corners_cam = np.stack(
            [
                (xs[None, :] * ws[:, None] - cxs[:, None]) / fxs[:, None],
                (ys[None, :] * hs[:, None] - cys[:, None]) / fys[:, None],
                np.ones((len(poses), 4)),
            ],
            axis=2,
        )  # (n_cams, 4, 3)

        R_wc_all = np.stack([p.R_cam_from_world.T for p in poses])
        dirs_world = np.einsum("nij,nkj->nki", R_wc_all, corners_cam)  # (n_cams, 4, 3)

        frustums_payload = [
            {
                "name": p.name,
                "side": p.side,
                "center": np.round(p.center_world, 5).tolist(),
                "dirs": np.round(dirs_world[i], 5).tolist(),
            }
            for i, p in enumerate(poses)
        ]
    else:
        frustums_payload = []

    scene_json = json.dumps(
        {
            "points": np.round(xyz, 4).reshape(-1).tolist(),
            "colors": rgb.reshape(-1).tolist(),
            "errors": np.round(errors, 3).tolist(),
            "num_points": int(len(xyz)),
            "cameras": frustums_payload,
            "default_frustum_scale": float(frustum_scale),
            "default_point_size": float(point_size),
        }
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>COLMAP SfM Interactive 3D Viewer</title>
<style>
  html, body {{
    margin: 0; padding: 0; width: 100%; height: 100%;
    overflow: hidden; background: #0e1117; color: #e6edf3;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }}
  #canvas-container {{ width: 100%; height: 100%; }}
  #hud {{
    position: absolute; top: 16px; left: 16px; width: 300px;
    background: rgba(22, 27, 34, 0.90); border: 1px solid #30363d;
    border-radius: 10px; padding: 14px 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.45);
    backdrop-filter: blur(8px); font-size: 13px;
  }}
  #hud h2 {{ margin: 0 0 8px 0; font-size: 15px; font-weight: 600; color: #58a6ff; }}
  .stat-row {{ display: flex; justify-content: space-between; margin: 4px 0; color: #8b949e; }}
  .stat-row span.val {{ color: #e6edf3; font-weight: 600; font-variant-numeric: tabular-nums; }}
  .control-group {{ margin-top: 10px; padding-top: 10px; border-top: 1px solid #21262d; }}
  .control-row {{ display: flex; align-items: center; justify-content: space-between; margin: 6px 0; }}
  .control-row label {{ color: #c9d1d9; }}
  input[type="range"] {{ width: 130px; accent-color: #58a6ff; }}
  .legend-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }}
  button {{
    margin-top: 8px; width: 100%; padding: 6px 10px; border-radius: 6px;
    border: 1px solid #30363d; background: #21262d; color: #e6edf3;
    cursor: pointer; font-size: 12px; font-weight: 500;
  }}
  button:hover {{ background: #30363d; }}
  #hint {{
    position: absolute; bottom: 14px; left: 16px;
    background: rgba(22, 27, 34, 0.8); padding: 6px 12px;
    border-radius: 6px; font-size: 12px; color: #8b949e; border: 1px solid #30363d;
  }}
</style>
</head>
<body>
<div id="canvas-container"></div>
<div id="hud">
  <h2>ZED X GS — SfM Reconstruction</h2>
  <div class="stat-row"><span>3D Landmarks</span><span class="val" id="stat-pts">0</span></div>
  <div class="stat-row"><span>Left Cameras</span><span class="val" id="stat-left">0</span></div>
  <div class="stat-row"><span>Right Cameras</span><span class="val" id="stat-right">0</span></div>

  <div class="control-group">
    <div class="control-row">
      <label>Point Size</label>
      <input id="pt-size" type="range" min="0.5" max="10" step="0.25" value="2.5" />
    </div>
    <div class="control-row">
      <label>Frustum Size</label>
      <input id="frustum-scale" type="range" min="0.02" max="2.5" step="0.02" value="0.3" />
    </div>
    <div class="control-row">
      <label><span class="legend-dot" style="background:#2f81f7;"></span>Left Camera</label>
      <input id="toggle-left" type="checkbox" checked />
    </div>
    <div class="control-row">
      <label><span class="legend-dot" style="background:#f78166;"></span>Right Camera</label>
      <input id="toggle-right" type="checkbox" checked />
    </div>
    <div class="control-row">
      <label>Trajectory Lines</label>
      <input id="toggle-traj" type="checkbox" checked />
    </div>
    <button id="btn-reset">Reset Camera View</button>
  </div>
</div>
<div id="hint">Left-drag: Orbit &nbsp;|&nbsp; Right-drag: Pan &nbsp;|&nbsp; Scroll: Zoom</div>

<script type="importmap">
{{
  "imports": {{
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }}
}}
</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const DATA = {scene_json};

// Convert COLMAP world coordinates (X right, Y down, Z forward) to Three.js (X right, Y up, Z backward)
function toThree(x, y, z) {{
  return new THREE.Vector3(x, -y, -z);
}}

const container = document.getElementById('canvas-container');
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e1117);

const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.01, 5000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

// 1. Point cloud
const numPts = DATA.num_points;
document.getElementById('stat-pts').textContent = numPts.toLocaleString();

const positions = new Float32Array(numPts * 3);
const colors = new Float32Array(numPts * 3);
const bbox = new THREE.Box3();

for (let i = 0; i < numPts; i++) {{
  const v = toThree(DATA.points[3*i], DATA.points[3*i+1], DATA.points[3*i+2]);
  positions[3*i] = v.x;
  positions[3*i+1] = v.y;
  positions[3*i+2] = v.z;
  colors[3*i] = DATA.colors[3*i] / 255.0;
  colors[3*i+1] = DATA.colors[3*i+1] / 255.0;
  colors[3*i+2] = DATA.colors[3*i+2] / 255.0;
  bbox.expandByPoint(v);
}}

const ptGeom = new THREE.BufferGeometry();
ptGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
ptGeom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
const ptMat = new THREE.PointsMaterial({{
  size: DATA.default_point_size * 0.02,
  vertexColors: true,
  sizeAttenuation: true,
}});
const pointCloud = new THREE.Points(ptGeom, ptMat);
scene.add(pointCloud);

// 2. Camera frustums & trajectories
const leftGroup = new THREE.Group();
const rightGroup = new THREE.Group();
const trajGroup = new THREE.Group();
scene.add(leftGroup);
scene.add(rightGroup);
scene.add(trajGroup);

const leftCams = DATA.cameras.filter(c => c.side === 'left');
const rightCams = DATA.cameras.filter(c => c.side === 'right');
document.getElementById('stat-left').textContent = leftCams.length;
document.getElementById('stat-right').textContent = rightCams.length;

DATA.cameras.forEach(c => {{
  bbox.expandByPoint(toThree(c.center[0], c.center[1], c.center[2]));
}});

function buildFrustums(scale) {{
  leftGroup.clear();
  rightGroup.clear();

  DATA.cameras.forEach(cam => {{
    const c = toThree(cam.center[0], cam.center[1], cam.center[2]);
    const corners = cam.dirs.map(d => toThree(
      cam.center[0] + d[0] * scale,
      cam.center[1] + d[1] * scale,
      cam.center[2] + d[2] * scale
    ));
    const verts = [];
    for (let i = 0; i < 4; i++) {{
      verts.push(c.x, c.y, c.z, corners[i].x, corners[i].y, corners[i].z);
      const next = corners[(i + 1) % 4];
      verts.push(corners[i].x, corners[i].y, corners[i].z, next.x, next.y, next.z);
    }}
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
    const color = cam.side === 'left' ? 0x2f81f7 : 0xf78166;
    const line = new THREE.LineSegments(geom, new THREE.LineBasicMaterial({{ color }}));
    if (cam.side === 'left') leftGroup.add(line);
    else rightGroup.add(line);
  }});
}}

function buildTrajectories() {{
  trajGroup.clear();
  [['left', leftCams, 0x58a6ff], ['right', rightCams, 0xffa657]].forEach(([side, cams, color]) => {{
    if (cams.length < 2) return;
    const pts = cams.map(c => toThree(c.center[0], c.center[1], c.center[2]));
    const geom = new THREE.BufferGeometry().setFromPoints(pts);
    const line = new THREE.Line(geom, new THREE.LineBasicMaterial({{ color }}));
    trajGroup.add(line);
  }});
}}

const initScale = DATA.default_frustum_scale || 0.25;
const scaleInput = document.getElementById('frustum-scale');
scaleInput.max = Math.max(initScale * 5, 1.0).toFixed(2);
scaleInput.step = (initScale * 0.05).toFixed(3);
scaleInput.value = initScale;

buildFrustums(initScale);
buildTrajectories();

function resetView() {{
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  if (!bbox.isEmpty()) {{
    bbox.getCenter(center);
    bbox.getSize(size);
  }}
  const radius = Math.max(size.length() * 0.65, 2.0);
  camera.position.set(center.x + radius * 0.5, center.y + radius * 0.45, center.z + radius * 0.85);
  controls.target.copy(center);
  controls.update();
}}
resetView();

// UI Listeners
document.getElementById('pt-size').addEventListener('input', e => {{
  ptMat.size = parseFloat(e.target.value) * 0.02;
}});
scaleInput.addEventListener('input', e => {{
  buildFrustums(parseFloat(e.target.value));
}});
document.getElementById('toggle-left').addEventListener('change', e => {{
  leftGroup.visible = e.target.checked;
}});
document.getElementById('toggle-right').addEventListener('change', e => {{
  rightGroup.visible = e.target.checked;
}});
document.getElementById('toggle-traj').addEventListener('change', e => {{
  trajGroup.visible = e.target.checked;
}});
document.getElementById('btn-reset').addEventListener('click', resetView);

window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(html, encoding="utf-8")
    print(f"Interactive WebGL viewer written to: {output_html.resolve()}")
    if open_browser:
        webbrowser.open(output_html.resolve().as_uri())
    return output_html


def view_with_matplotlib(
    xyz: np.ndarray,
    rgb: np.ndarray,
    poses: list[CameraPoseData],
    frustum_scale: float,
    point_size: float,
) -> None:
    """Open an interactive Matplotlib 3D window."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("COLMAP SfM Reconstruction (Left=Blue, Right=Orange)")

    if len(xyz) > 0:
        # Subsample only if huge so matplotlib rotation stays responsive
        step = max(1, len(xyz) // 15000)
        pts = xyz[::step]
        cols = rgb[::step].astype(np.float64) / 255.0
        ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], c=cols, s=point_size, alpha=0.8)

    for side, color in (("left", "dodgerblue"), ("right", "darkorange")):
        side_poses = [p for p in poses if p.side == side]
        if not side_poses:
            continue
        centers = np.vstack([p.center_world for p in side_poses])
        ax.plot(centers[:, 0], centers[:, 2], -centers[:, 1], color=color, linewidth=2.0, label=f"{side} trajectory ({len(side_poses)})")
        dirs = np.vstack([p.forward_world for p in side_poses]) * frustum_scale
        ax.quiver(
            centers[:, 0],
            centers[:, 2],
            -centers[:, 1],
            dirs[:, 0],
            dirs[:, 2],
            -dirs[:, 1],
            color=color,
            length=1.0,
            normalize=False,
        )

    ax.set_xlabel("X (right)")
    ax.set_ylabel("Z (forward)")
    ax.set_zlabel("-Y (up)")
    ax.legend()
    plt.tight_layout()
    print("Opening Matplotlib interactive 3D window...")
    plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive 3D viewer for pycolmap SfM reconstructions."
    )
    parser.add_argument(
        "--sparse",
        type=Path,
        default=Path("seg2pointcloud/out/sparse"),
        help="Path to the sparse output folder (or a specific model subfolder like sparse/0).",
    )
    parser.add_argument(
        "--model-idx",
        type=int,
        default=None,
        help="Specific model index under --sparse (default: auto-select largest model).",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "open3d", "viser", "web", "matplotlib"),
        default="auto",
        help="Interactive viewer backend to use (default: auto).",
    )
    parser.add_argument(
        "--max-reproj-error",
        type=float,
        default=4.0,
        help="Filter out 3D points with reprojection error above this threshold in px (default: 4.0).",
    )
    parser.add_argument(
        "--min-track-length",
        type=int,
        default=2,
        help="Minimum number of observing images per 3D point (default: 2).",
    )
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=98.0,
        help="Distance percentile from scene median to clip far outlier points (default: 98.0, 100 to disable).",
    )
    parser.add_argument(
        "--frustum-scale",
        type=float,
        default=None,
        help="Camera frustum size in world units (default: auto-scaled from trajectory extent).",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.5,
        help="Rendered 3D point size (default: 2.5).",
    )
    parser.add_argument(
        "--export-ply",
        type=Path,
        default=None,
        help="Optional path to also export the sparse point cloud as a .ply file.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the Viser web server when --backend viser is used (default: 8080).",
    )
    args = parser.parse_args()

    try:
        model_dir, rec = find_best_reconstruction(args.sparse, args.model_idx)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded reconstruction from: {model_dir}")
    print(f"  Registered images: {rec.num_reg_images()} / {rec.num_images()}")
    print(f"  Raw 3D points:     {rec.num_points3D()}")

    if args.export_ply is not None:
        args.export_ply.parent.mkdir(parents=True, exist_ok=True)
        rec.export_PLY(str(args.export_ply))
        print(f"  Exported PLY to:   {args.export_ply}")

    xyz, rgb, errors, poses = extract_scene_data(
        rec,
        max_reproj_error=args.max_reproj_error,
        min_track_length=args.min_track_length,
        clip_percentile=args.clip_percentile,
    )
    left_count = sum(1 for p in poses if p.side == "left")
    right_count = sum(1 for p in poses if p.side == "right")
    print(f"  Filtered 3D pts:   {len(xyz)}")
    print(f"  Camera poses:      {left_count} left, {right_count} right")

    frustum_scale = estimate_frustum_scale(xyz, poses, args.frustum_scale)
    html_out = model_dir.parent / "viewer.html"

    if args.backend == "open3d":
        view_with_open3d(xyz, rgb, poses, frustum_scale, args.point_size)
        return 0

    if args.backend == "viser":
        view_with_viser(xyz, rgb, poses, frustum_scale, args.point_size, port=args.port)
        return 0

    if args.backend == "web":
        build_web_viewer_html(
            xyz, rgb, errors, poses, frustum_scale, args.point_size, html_out, open_browser=True
        )
        return 0

    if args.backend == "matplotlib":
        view_with_matplotlib(xyz, rgb, poses, frustum_scale, args.point_size)
        return 0

    # ---- auto mode: always generate viewer.html, then launch best available interactive window ----
    build_web_viewer_html(
        xyz, rgb, errors, poses, frustum_scale, args.point_size, html_out, open_browser=False
    )

    try:
        import open3d  # noqa: F401

        view_with_open3d(xyz, rgb, poses, frustum_scale, args.point_size)
        return 0
    except ImportError:
        pass

    try:
        import viser  # noqa: F401

        view_with_viser(xyz, rgb, poses, frustum_scale, args.point_size, port=args.port)
        return 0
    except ImportError:
        pass

    print("Opening self-contained Three.js viewer in your browser...")
    webbrowser.open(html_out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
