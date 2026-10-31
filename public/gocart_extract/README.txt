ZED extracts from test4_20260902_134059 and test5_20260902_134616
================================================================

Layout
  video_extract/
    test4/          ~4 min bag
    test5/          ~12 min bag
    scripts/        extract helpers (ignore unless re-running)
    README.txt

Each of test4/ and test5/ contains:

  left.mp4       left rectified color, 1920x1200
  right.mp4      right rectified color, 1920x1200
  stereo.mp4     left | right side-by-side, 3840x1200
  depth.mp4      depth turbo colormap, 0-15 m (black = invalid)
  rgbd.mp4       left color | colorized depth, 3840x1200
  imu.csv        ZED IMU (orientation, gyro, accel)
  pose_tum.txt   VIO pose odom -> zedx_camera_link
  calib.yaml     camera_info + tf_static extrinsics
  pointcloud/    colored RGB-D PLY, ~1 Hz, optical frame

Videos are ~4 fps H.264. These bags did not record confidence, disparity,
native point cloud, unrectified images, or object/body tracking.
