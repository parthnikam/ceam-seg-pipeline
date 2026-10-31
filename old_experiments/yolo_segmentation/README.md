# Road contour marking for F1 cockpit footage

Turn onboard F1 video into left/right track edges, then (later) a top-down map of the circuit.

## Goal

1. Use an existing semantic model to mask the road in cockpit frames.
2. Use those masks as labels for a small, fast CNN that only has to find track contours.
3. Project those contours into a 2D top-down map of the track.

Work so far is step 1 and a few failed/partial attempts at doing step 2 with classical OpenCV instead of a CNN.

## Dataset

Cockpit videos live in `public/dataset/videos/`. They are independent clips, not a train/val split. There is still no held-out validation set.

| video | resolution | fps | length | frames in file |
| --- | --- | --- | --- | --- |
| `f1_video1.mp4` | 1920×1080 | 50 | 103.3s | 5167 |
| `f1_video2.mp4` | 1920×1080 | 50 | 116.9s | 5843 |
| `f1_video3.mp4` | 1920×1080 | 50 | 115.9s | 5795 |
| `f1_video4.mp4` | 1280×720 | 60 | 141.5s | 8491 |

Extracted stills are in `public/dataset/images/<clip>/`. Current counts: **207 / 234 / 232** frames for videos 1–3. Video 4 has not been extracted. Earlier runs pulled every frame (thousands per clip); those were thinned later because neighboring frames are almost identical.

`public/dataset/data.yaml` lists the Cityscapes class names used by the YOLO checkpoint. Class `0` is `road`. It is **not** a training config: `f1_video1` and `f1_video2` are different circuits/sessions, not train and val.

## What exists

```
extract_frames.py              video → JPEG frames
yolo_segmenter.py              YOLO26-sem inference on a folder of frames
opencv_method/                 huent189 bird’s-eye + sliding-window pipeline
lift_splat/                    LSS frustum + splat of YOLO road masks into BEV
lift-splat-shoot/              upstream LSS repo (nuScenes trainer; not run as-is)
download_model.py              one-shot DeepLab tarball download
tensorflow_segmentation.py     DeepLab inference (loads the local tarball only)
public/models/yolo26m-sem.pt   packed YOLO26 medium semantic checkpoint
<images>/output/               YOLO outputs (semantic / road / overlays)
public/dataset/lines/          leftover outputs from the first OpenCV blob script
```

Install:

```bash
pip install -r requirements.txt
```

`tensorflow_segmentation.py` also needs TensorFlow 1.x-style APIs (`tf.Session`, frozen graph). That is separate from the Ultralytics stack.

This machine is CPU-only (Intel UHD, no CUDA). YOLO and DeepLab are slow here. A discrete NVIDIA GPU would change runtime, not the method.

## Work log

### 1. Frame extraction

`extract_frames.py` takes one video path and one output folder. It prints total frames vs. frames that will be written, creates the output folder, and waits for confirmation (`-y` skips that).

```bash
python extract_frames.py public/dataset/videos/f1_video1.mp4 public/dataset/images/f1_video1 --fps 2
```

`--every`, `--start` / `--end`, and `--max-frames` are also available. Full 50 fps extract is possible but not useful for labeling.

### 2. YOLO26 semantic segmentation

The downloaded `yolo26m-sem` folder was an unpacked torch `.pt`. `yolo_segmenter.py` zips it to `public/models/yolo26m-sem.pt` and runs **inference** (no training, no `coco8.yaml`).

```bash
python yolo_segmenter.py public/dataset/images/f1_video1 --max-images 10 -y
python yolo_segmenter.py public/gocart_extract/test4/images/left --max-images 10 -y
```

Each run writes `<images>/output/`:

- `semantic/` — PNG class-id maps (sidewalk remapped to road)
- `road/` — binary road mask (class 0, including former sidewalk)
- `overlays/` — color overlay for QA

Campus pavement is often split across Cityscapes `road` and `sidewalk` in the same frame. Both are folded into road. To apply that to masks that already exist, without re-running YOLO:

```bash
python yolo_segmenter.py public/gocart_extract/test5/images/right --rewrite -y
```

Smoke test on `f1_video1_000000`: tarmac ahead was `road` (~20% of the frame). The cockpit was labeled `car` (~41%), which is expected from a Cityscapes street model. Some runoff/walls leak into `road` / `building`.

Existing YOLO outputs on disk: **1000** frames for video 1, **4121** for video 2. Those numbers are from an earlier denser extract, not from the current ~200-frame folders.

### 3. First OpenCV attempt (dropped)

A homegrown script (`opencv_line_segmenter.py`) tried color threshold + Canny/Hough + asphalt-blob edges. It was bad on this footage: lines hugged the image border, locked onto barriers and the halo, and missed the Monaco tunnel. It was replaced by the method below. Residual outputs are still under `public/dataset/lines/`.

### 4. OpenCV method from huent189

`opencv_method/` is the [huent189/road_segmentation](https://github.com/huent189/road_segmentation) pipeline, kept in its own folder:

1. HLS + Sobel gradient threshold
2. Bird’s-eye warp
3. Sliding-window left/right fit (uses the previous frame when it can)
4. Unwarp the lane polygon back onto the cockpit view

The original repo was calibrated for a 1280×720 Digital Race camera. Here the ROI and trapezoid are fractions of the frame so they run on 1920×1080 F1 video. A couple of upstream bugs were fixed so it actually runs (`np.int`, a window that never slid).

```bash
python opencv_method/run.py public/dataset/images/f1_video1 public/dataset/opencv_method/f1_video1 --max-images 30 --debug -y
```

Better than the blob script on some outdoor frames (green lane sits on the tarmac). Still fails when the histogram latches onto a wall or the tunnel has no Cityscapes-like paint. That is the limitation the original author already noted: thresholding is brittle to lighting and scene change.

### 5. DeepLab (TensorFlow)

`tensorflow_segmentation.py` started as a notebook-style DeepLabv3 Cityscapes demo. It downloaded the checkpoint into a **new temp directory every run**.

That is now split:

```bash
python download_model.py                          # once → public/models/deeplab/*.tar.gz
python tensorflow_segmentation.py                 # load only, no download
```

`--force` re-downloads. `--model xception65_cityscapes_trainfine` fetches the larger variant. If the tarball is missing, segmentation tells you to run `download_model.py`. The DeepLab file is not on disk yet until that command is run.

### 6. Lift-Splat BEV from YOLO road masks

`lift_splat/` is the [Lift, Splat, Shoot](https://github.com/nv-tlabs/lift-splat-shoot) geometry only, in its own folder, the same way `opencv_method/` holds the huent189 pipeline. The upstream repo is a 6-camera nuScenes trainer; we do not load EfficientNet, train, or touch nuScenes.

Each YOLO road mask is lifted along camera rays and splatted onto a 0.5 m bird's-eye grid. Depth is peaked at the ground plane (tarmac sits on z=0). The onboard camera is an uncalibrated pinhole (80° HFOV, 1.15 m, 12° down) — tune `--hfov` / `--height` / `--pitch` if the top-down track looks scaled wrong.

```bash
python lift_splat/run.py public/dataset/images/f1_video1 --max-images 8 -y
```

Defaults to YOLO semantic classes `road+wall+terrain` because on this footage the tarmac is often labelled `wall`. Pass `--occupancy mask` to use only the binary `road/` PNG. This is still a per-frame BEV, not a stitched lap map.

## What we learned

F1 onboard is not highway lane detection. There is often no painted left/right line — the “line” is the edge of the asphalt against kerb, grass, gravel, or a barrier. The cockpit fills the bottom third of the frame. Shadows, advertising, the halo, and tunnels break color/Hough methods.

A semantic **road mask** is the useful intermediate: left and right contours are the two sides of that blob. YOLO already produces that mask. OpenCV is more reliable **on the mask** than on the raw RGB frame. A second CNN is only worth training if YOLO is too slow at 50 fps or the Cityscapes labels are too dirty for mapping.

No lightweight contour CNN has been trained. No bird’s-eye map of a full lap has been built. There is still no train/val split.

## next steps

1. QA the YOLO `road/` masks and throw out frames where the cockpit or runoff is labeled as track.
2. Trace left/right polylines from those cleaned masks (OpenCV `findContours` on the binary road image, not on RGB).
3. Only then decide whether to train a tiny CNN on those polylines, or use the mask contours directly for a homography / top-down stitch.
4. If you need a val set, hold out frames (or a later session). Do not treat `f1_video2` as validation for `f1_video1`.
