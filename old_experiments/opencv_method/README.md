# White-line top-down map

1. Find white paint in the road band
2. Fill the area between left and right lines
3. Inverse-perspective onto a metric ground plane
4. Overlap frames into `map.png`

```
camera.py    pinhole ↔ ground
paint.py     ridges, line fit, fill
ground.py    BEV warp + mosaic
pipeline.py  one frame
run.py       CLI
```

```bash
python run.py ../public/dataset/videos/f1_video1.mp4 out/f1_video1 --max-images 80 --debug -y
```
