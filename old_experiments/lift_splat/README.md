# Lift-Splat on F1 onboard frames

This folder is the [Lift, Splat, Shoot](https://github.com/nv-tlabs/lift-splat-shoot) geometry, kept separate from the original repo and from the OpenCV homography pipeline.

The original code is a 6-camera nuScenes trainer (EfficientNet lift, BEV ResNet, vehicle labels, lidar, planning). None of that applies here: one uncalibrated halo camera, no BEV ground truth, CPU-only. What we keep is the part that is actually the method:

1. Build a frustum of 3D points along each camera ray (`create_frustum` + `get_geometry`)
2. Put a feature on those points — here the YOLO **road mask**, not learned image features
3. Splat them into a bird's-eye grid (`voxel_pooling`)

Depth along each ray is peaked at the ground-plane hit (`--depth ground`). That is the right prior for tarmac. `--depth uniform` is the uninformed LSS lift and smears the mask along the ray.

Camera extrinsics/intrinsics are **not** in the dataset. Defaults are a pinhole halo cam: 80° HFOV, 1.15 m high, 12° down. If the BEV track is too wide, drop `--hfov`; if it starts too close or too far, change `--pitch` / `--height`.

```bash
python lift_splat/run.py public/dataset/images/f1_video1 --max-images 8 -y
python lift_splat/run.py public/gocart_extract/test4/images/left --max-images 8 -y
```

The source folder is the clip: JPEG frames at the top level, YOLO products in `output/{semantic,road,overlays}/` (what `yolo_segmenter.py` writes). BEV lands in `<source>/lift_splat/`.

On F1 footage YOLO often paints the tarmac as `wall` and only the runoff as `road`. Default occupancy unions class ids `0,3,9` (road, wall, terrain) from `output/semantic/`. Pass `--occupancy mask` for the binary road PNG only. The colour BEV splats RGB at ground-plane depth, so you still see kerbs and tarmac.

Each run writes:

- `bev/` — occupancy, car at the bottom, 0.5 m cells, 30 m forward
- `bev_color/` — same grid, coloured with the source pixels
- `overlays/` — camera / mask on the left, colour BEV on the right

Taken from `lift-splat-shoot/src/models.py` and `src/tools.py` (NVIDIA Source Code License, research/evaluation only). See `LICENSE`.
