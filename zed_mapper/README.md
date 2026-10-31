# zed_mapper

Fast, lightweight mapping for a campus go-kart with **one forward ZED X**. LiDAR is mounted (Livox in the URDF) but not in these extracts.

This folder is the product. The F1 pipelines under `opencv_method/`, `lift_splat/`, `yolo_segmentation/`, and `f1_threshold_method/` are RGB-only baselines.

## Goal

From a single front depth camera, every frame should produce:

1. **Drivable mask** — pavement the cart may occupy. Grass is also “ground”; it is not drivable.
2. **Left and right boundaries** — asphalt vs grass / kerb / building, not painted lane lines.
3. **Ego bird’s-eye** — metric occupancy of the next ~30 m.
4. **Two poses**
   - **Track-relative:** cross-track error `e_y`, heading error `e_ψ`, width left/right, curvature `κ`. Steering can use this with no global map.
   - **Map pose:** `(x, y, yaw)` in odom, drawn on a stitched top-down map.

## Hardware

| now | later |
| --- | --- |
| Stereolabs ZED X, 1920×1200 rectified, ~4 fps in the extracts | same camera live via `zed-ros2-wrapper` |
| ZED VIO pose (`pose_tum.txt`) + IMU gyro (`imu.csv`) | EKF with map match |
| — | Livox as a second BEV layer |

There is **no** `base_link → zedx_camera_link` in these extracts (only a ZED One mount). The map is built in **odom**, not `base_link`. Do not copy the ZED One extrinsics onto the X.

ZED neural depth needs NVIDIA CUDA when live. These extracts already have depth.

## Extracts

`public/gocart_extract/test4` (~4 min) and `test5` (~12 min):

| file | contents |
| --- | --- |
| `left.mp4` / `right.mp4` | rectified color, 1920×1200 |
| `depth.mp4` | turbo colormap of depth, **0–15 m**, black = invalid |
| `pose_tum.txt` | ZED VIO `t tx ty tz qx qy qz qw` (odom → `zedx_camera_link`). **Not** the IMU. |
| `calib.yaml` | `K` and `tf_static` |
| `pointcloud/*.ply` | ~1 Hz colored XYZ in the optical frame |
| `images/left/` | left rectified JPEGs (test4) |
| `images/left/output/road/` | optional YOLO26-sem binary road masks |
| `imu.csv` | 100 Hz `qx..qw, wx..wz, ax..az`. Gyro fills heading between VIO samples. |

`depth.mp4` is **not** a float depth map. The extract script ran `COLORMAP_TURBO` over 0–15 m. We invert that LUT (~6 cm quantization). The 1 Hz PLYs are the metric check, not the runtime path.

Pair left and depth by frame index. Stamp the mosaic with VIO translation and
IMU-gyro heading. `pose_tum.txt` is already-fused visual-inertial odometry
(position + orientation, ~30 Hz with dropouts). `imu.csv` is the raw IMU
(no translation). They are not interchangeable; the map still needs TUM for x/y.

## v1 pipeline

No new neural net. test4 uses YOLO road masks when present; test5 uses an asphalt color prior. Both are fused with a depth ground plane.

```
left RGB + turbo-depth + TUM VIO + IMU gyro [+ YOLO road]
        │
        ├─ RANSAC plane from near stereo
        ├─ flat ground = |height| ≈ 0  (a wall has *uniform depth*, not uniform height)
        ├─ seed in front of the cart, flood through planar pavement
        │     far pixels with no stereo join only if they touch that blob
        ├─ inverse IPM of that mask, near field only (~12 m, not 40 m)
        ├─ L/R of the blob in front of the cart (width = the mask)
        └─ mosaic: stamp ~8 m ahead + START + path + NOW
```

A wall facing the camera has almost the same depth in a patch. The road does the opposite: depth grows toward the vanishing point, but the 3D points stay on one plane. We grow that plane from the bumper, then keep the road-looking part (YOLO / asphalt). Grass that is also flat is dropped.

The map is no longer a red/green occupancy dump. `map.png` is road + a white path + a **START** wedge and a **NOW** wedge. Ego BEV has a metre grid, range labels, and the cart at the bottom.

```bash
python zed_mapper/run.py public/gocart_extract/test4 zed_mapper/out/test4 --max-frames 80 --debug -y
python zed_mapper/run.py public/gocart_extract/test5 zed_mapper/out/test5 --start 400 --max-frames 80 --ahead 8 --debug -y
```

Writes `overlays/` (camera + live map HUD), `bev/`, `map.png`, and **`drive.mp4`**. Each video frame is the camera overlay with a right-hand panel of the accumulating world map (`NOW` = the cart) and the ego BEV. `--fps` defaults to 10 so playback isn’t stuck at the extract’s ~4 fps. `--no-video` skips the mp4. `--start` skips into the clip (the first seconds of these bags are often stationary). test4 uses YOLO road masks under `output/road/` when present; test5 uses the asphalt color prior.

HUD on the overlay: `ey` is centerline `y` at 8 m (camera_link, +left), `epsi` heading vs the centerline, `w` width, `k` curvature. Camera overlay is the road mask only. BEV/map: CART/NOW, START, path, road.

**v1 limits:** a plaza that fills the camera still has no painted kerb — we then keep the near mask width instead of inventing a lane. Far IPM is only as good as the plane (hills need a local plane or lidar). No loop closure — the mosaic is dead-reckoned ZED VIO. `--ahead` (default 8 m) is the map stamp range.

## Why the F1 methods are not this

| experiment | keep as | why it is not the cart stack |
| --- | --- | --- |
| YOLO26-sem Cityscapes | teacher / optional appearance | wrong domain, leaks sidewalk, misses the hood, too heavy to be the runtime model |
| paint ridges + IPM mosaic | mosaic idea | assumes white paint; fake pinhole; Farneback odometry |
| LSS splat | BEV splat pattern | invents depth along the ray; ZED **measures** depth |
| HLS + sliding window | classic lane detector | lighting-brittle; painted-line prior |

## Curves

The method that holds on bends is **not** a polynomial in the camera image and not a “curve CNN” on RGB.

1. Sample left/right from the **drivable mask**.
2. Back-project with ZED depth into **metric BEV**.
3. Fit a **piecewise cubic** (Catmull-Rom / B-spline) to those points, with a width prior tying the two edges.
4. Curvature in SI units: `κ = (x'y'' − y'x'') / (x'² + y'²)^{3/2}`.

| method | on real corners |
| --- | --- |
| Polynomial in the image | no — perspective warps the bend |
| One global 2nd/3rd-order poly in BEV | only a single gentle arc |
| Sliding window / UFLD / CLRNet / CurveLane-NAS / BézierLaneNet | painted lanes, not grass vs tarmac |
| 3D-LaneNet / PersFormer | infer 3D; we already measure it |
| Clothoid in BEV | yes, if the path is designed (OpenDRIVE). v1.1 |
| **Catmull-Rom / cubic B-spline in BEV** | **v1** — piecewise, so S-bends and hairpins survive |

Do not fit in image space. `opencv_method/ground.py` used a single quadratic and preferred a straight line; that model is not reused here.

## Algorithm catalog

### Drivable segmentation

- **Depth-only plane** — fast, metric, cannot tell tarmac from grass.
- **Classical RGB** — HLS asphalt vs green. Fine in even sun.
- **YOLO26-sem** — useful labels, not the cart model.
- **v1 fusion:** seed planar pavement in front of the cart (`|h|≈0` and road appearance), flood through that surface. Far no-depth pixels join only if they touch it. Uniform-depth patches that stick up are walls, not road.
- **Later:** tiny RGB-D net if appearance is the weak link. The lift stays ray–plane.

### Boundaries

Connected pavement in front of the cart, then left/right of that blob in the **near** field (~2–10 m). Rows that hit the camera FOV or whose width disagrees with the near mask are dropped, so overlay lines follow the road instead of shooting to the horizon. Catmull-Rom on those samples. Width 1.5–10 m, taken from the mask.

### Ego BEV (the actual 2D → map step)

Inverse IPM on the live RANSAC plane: for each 0.12 m cell (2–12 m forward, ±6 m) project onto the image and sample the mask. Horizon / image-border samples are ignored. The world map only **stamps ~8 m ahead** (`--ahead`), so the traversed corridor stays tight as you drive. Stereo unproject is reserved for obstacles, which are *not* on the plane.

### Track-relative pose

From the ego centerline at a lookahead (~5 m): `e_y`, `e_ψ`, `w_left`, `w_right`, `κ`. Stanley / pure-pursuit interface. Works on lap 1.

### Map pose

Stamp the ego BEV at the VIO position with IMU-propagated yaw (not optical flow). Occupancy mosaic + the live spline overlay. Cart arrow from odom yaw.

Later: pose-graph / RTAB-Map loop closure, freeze a vector map of L/R polylines, match live edges to that map (lateral lock), EKF `map → odom → zedx_camera_link`.

### LiDAR later

Camera remains the road-vs-grass expert. Livox becomes ground / obstacle / NDT. Same BEV occupancy interface; late AND (`camera_road ∧ lidar_ground ∧ ¬lidar_obstacle`). Do not assume a BEV came from the camera.

## Layout

```
dataset.py     extract loader (calib, RGB, turbo-depth, TUM pose, IMU, YOLO)
camera.py      K, optical ↔ camera_link, unproject, turbo LUT
ground.py      RANSAC plane, ray–plane IPM
segment.py     appearance − hood − stereo obstacles
boundaries.py  BEV occupancy → coupled spline → κ, e_y, e_ψ
bev.py         inverse-IPM occupancy grid
mosaic.py      stamp at VIO + IMU pose
pipeline.py    one frame
vis.py         overlays
run.py         CLI
```

## Frameworks

**v1:** Python, OpenCV, NumPy, PyYAML.

**Later:** PyTorch + TensorRT for a tiny segmenter; ROS 2 for live topics; GTSAM / RTAB-Map for loop closure; `robot_localization` EKF; Lanelet2 or a JSON polyline map; mmdet3d BEVFusion only if late fusion is not enough.

## Build order

1. ~~ZED extract loader + turbo invert + plane BEV~~ (this package, v1)
2. ~~Mask contours → spline → track-relative + odom mosaic~~ (this package, v1)
3. Tiny RGB-D segmenter on cart logs
4. Clothoid / vector map, loop closure, freeze-and-localize
5. Live ROS node
6. Livox as a second BEV source

## Files

OpenCV, NumPy, PyYAML. No SciPy, no ZED SDK at runtime.
