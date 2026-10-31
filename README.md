# road_segmentation

Go-kart perception from a single forward **ZED X**: drivable pavement, left/right track edges, a metric top-down map, and the cart’s pose on that map. LiDAR (Livox) is on the URDF and comes later.

**Implementation lives in [`zed_mapper/`](zed_mapper/).** That README is the system design (algorithms, curve fitting, mapping). Run it on the ZED extracts in `public/gocart_extract/test4` and `test5`.

```bash
python zed_mapper/run.py public/gocart_extract/test4 zed_mapper/out/test4 --max-frames 80 --debug -y
```

Older RGB-only experiments on F1 cockpit video (kept as baselines, not the product):

| folder | what it is |
| --- | --- |
| [`zed_mapper/`](zed_mapper/) | ZED X extract → mask, BEV spline, odom mosaic |
| [`opencv_method/`](opencv_method/) | white-paint ridges + homography mosaic |
| [`lift_splat/`](lift_splat/) | lift a YOLO mask along rays onto a BEV grid |
| [`yolo_segmentation/`](yolo_segmentation/) | YOLO26-sem Cityscapes inference |
| [`f1_threshold_method/`](f1_threshold_method/) | HLS + bird’s-eye sliding windows |

`public/` (extracts, models, F1 clips) and `repos/` are gitignored.
