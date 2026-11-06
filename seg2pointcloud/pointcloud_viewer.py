
import open3d as o3d
import numpy as np
from tqdm import tqdm
import time

# 1. Load point cloud
print("\n[1/6] Loading point cloud...")
pcd = o3d.io.read_point_cloud(
    "../public/gocart_extract/allnew/campus/SurfMap.pcd"
)

print(f"Loaded {len(pcd.points):,} points")


# 2. Estimate normals
print("\n[2/6] Estimating normals...")
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1,
        max_nn=30
    )
)

print("Orienting normals...")
pcd.orient_normals_consistent_tangent_plane(k=15)


# 3. Poisson reconstruction
print("\n[3/6] Starting Poisson reconstruction...")
print("This can take a while depending on point cloud size.")

start = time.time()

# Open3D does not expose internal Poisson progress.
# This displays an activity indicator rather than fake percentage progress.
from tqdm import tqdm
import threading

done = threading.Event()

def show_progress():
    with tqdm(
        total=100,
        desc="Generating mesh",
        bar_format="{desc}: {bar} {elapsed} | {postfix}"
    ) as pbar:
        while not done.is_set():
            pbar.update(1)
            pbar.set_postfix_str("Reconstructing...")
            time.sleep(0.5)

        pbar.update(100 - pbar.n)
        pbar.set_postfix_str("Finished")

thread = threading.Thread(target=show_progress)
thread.start()

mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd,
    depth=9
)

done.set()
thread.join()

print(f"Reconstruction completed in {time.time() - start:.2f}s")


# 4. Remove low-density artifacts
print("\n[4/6] Removing low-density artifacts...")

densities = np.asarray(densities)

density_mesh_to_remove = densities < np.quantile(
    densities, 0.05
)

mesh.remove_vertices_by_mask(density_mesh_to_remove)

print(f"Vertices remaining: {len(mesh.vertices):,}")
print(f"Triangles remaining: {len(mesh.triangles):,}")


# 5. Compute mesh normals
print("\n[5/6] Computing mesh normals...")
mesh.compute_vertex_normals()


# 6. Visualize and save
print("\n[6/6] Saving mesh...")

o3d.io.write_triangle_mesh("poisson_mesh.ply", mesh)

print("Saved: poisson_mesh.ply")

o3d.visualization.draw_geometries([mesh],window_name="Poisson Mesh",width=1024,height=768)

print("\nAll operations completed!")

