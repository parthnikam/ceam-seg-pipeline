# Road Segmentation to Point Cloud 


We're using lidar point cloud to self locate in 3d space.
PRoblem: We dont know what part is drivable out of that. and our lidar hasn't collected any data for the ground plane. 
We can't just map drivable segments from the image to the point clouds.


What I think should be done: 
1. Visual SLAM / Structure from Motion: Recover the camera's movement and reconstruct 3D landmarks from successive images.
2. LiDAR-based geometric mapping: Use the buildings and other vertical structures as geometric references.
3. Map registration: Align the camera-derived reconstruction with the LiDAR coordinate system.
4. Temporal synchronization: Associate each camera frame and its segmentation mask with the corresponding LiDAR scan.


use pycolmap for structure for motion; 
the left and right camera intrinsics are already known from the zed camera info;
and the extrinsics are also known;




