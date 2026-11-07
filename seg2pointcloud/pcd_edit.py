#!/usr/bin/env python3
"""Interactive PCD editing helpers (crop, downsample, clean, transform, save).

Every function takes and returns an Open3D point cloud so they can be
chained. The CLI applies a sequence of edits to an input PCD and writes the
result to a new file.

Usage:
    python seg2pointcloud/pcd_edit.py --input map.pcd \
        --crop-box 0,-5,0,50,5,10 --voxel 0.05 --output campus.pcd

    # Open the point-cloud editing GUI:
    python seg2pointcloud/pcd_edit.py --gui

    In the GUI, open a point cloud and use "Crop box by picking two corners"
    to select opposite corners in the 3D window (Shift-click, then close it).
    The other edit operations are available as buttons.

    # Crop to a sphere around a point of interest:
    python seg2pointcloud/pcd_edit.py --input map.pcd --crop-sphere 10,2,1,15

    # Interactively pick the points to keep in a 3D window:
    python seg2pointcloud/pcd_edit.py --input map.pcd --pick
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def load_pcd(path) -> o3d.geometry.PointCloud:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Point cloud not found: {path}")
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise ValueError(f"Point cloud is empty: {path}")
    print(f"  loaded {len(pcd.points):,} points from {path}")
    return pcd


def save_pcd(pcd: o3d.geometry.PointCloud, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(path), pcd):
        raise RuntimeError(f"Could not write point cloud: {path}")
    print(f"  saved {len(pcd.points):,} points to {path}")
    return path


def crop_box(
    pcd: o3d.geometry.PointCloud,
    min_bound=(0.0, 0.0, 0.0),
    max_bound=(10.0, 10.0, 10.0),
) -> o3d.geometry.PointCloud:
    """Keep only points inside an axis-aligned box."""
    bbox = o3d.geometry.AxisAlignedBoundingBox(
        np.asarray(min_bound, dtype=np.float64),
        np.asarray(max_bound, dtype=np.float64),
    )
    return _report_crop(pcd, pcd.crop(bbox), f"box {min_bound} -> {max_bound}")


def crop_sphere(
    pcd: o3d.geometry.PointCloud,
    center=(0.0, 0.0, 0.0),
    radius: float = 10.0,
) -> o3d.geometry.PointCloud:
    """Keep only points within `radius` meters of `center`."""
    pts = np.asarray(pcd.points)
    dists = np.linalg.norm(pts - np.asarray(center, dtype=np.float64), axis=1)
    kept = pcd.select_by_index(np.flatnonzero(dists <= radius))
    return _report_crop(pcd, kept, f"sphere center={center} radius={radius}")


def crop_oriented_box(
    pcd: o3d.geometry.PointCloud,
    center=(0.0, 0.0, 0.0),
    extents=(10.0, 10.0, 10.0),
    yaw_deg: float = 0.0,
) -> o3d.geometry.PointCloud:
    """Keep only points inside a rotated (yaw about Z) box."""
    yaw = np.deg2rad(yaw_deg)
    R = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    obb = o3d.geometry.OrientedBoundingBox(
        np.asarray(center, dtype=np.float64), R, np.asarray(extents, dtype=np.float64)
    )
    return _report_crop(pcd, pcd.crop(obb), f"oriented box center={center} extents={extents} yaw={yaw_deg}deg")


def pick_points(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Open a 3D window, let the user pick points (shift-click), then keep those."""
    print("  shift-click to select points, then close the window")
    indices = _pick_point_indices(pcd, "Pick points to keep")
    return _report_crop(pcd, pcd.select_by_index(indices), "picked points")


def _pick_point_indices(pcd: o3d.geometry.PointCloud, title: str) -> list[int]:
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=title, width=1280, height=800)
    try:
        vis.add_geometry(pcd)
        vis.run()
        return vis.get_picked_points()
    finally:
        vis.destroy_window()


def downsample(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.05,
) -> o3d.geometry.PointCloud:
    """Voxel-grid downsample (also averages colors/normals per voxel)."""
    result = pcd.voxel_down_sample(voxel_size)
    print(f"  voxel {voxel_size} m: {len(pcd.points):,} -> {len(result.points):,} points")
    return result


def remove_outliers(
    pcd: o3d.geometry.PointCloud,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> o3d.geometry.PointCloud:
    """Remove statistical outliers."""
    result, _ = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
    print(f"  outliers: removed {len(pcd.points) - len(result.points):,} points")
    return result


def translate(
    pcd: o3d.geometry.PointCloud,
    xyz=(0.0, 0.0, 0.0),
) -> o3d.geometry.PointCloud:
    """Shift the whole cloud by the given offset."""
    pcd.translate(np.asarray(xyz, dtype=np.float64))
    print(f"  translated by {xyz}")
    return pcd


def preview(pcd: o3d.geometry.PointCloud) -> None:
    """Open an interactive 3D window showing the point cloud."""
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud", width=1024, height=768)


def _report_crop(original, kept, what: str) -> o3d.geometry.PointCloud:
    print(f"  crop {what}: {len(original.points):,} -> {len(kept.points):,} points")
    return kept


def _parse_floats(text: str) -> list[float]:
    return [float(v) for v in text.split(",") if v.strip()]


class PcdEditorApp:
    def __init__(
        self,
        root,
        initial_input: Path | None = None,
        initial_output: Path | None = None,
    ):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.tk = tk
        self.ttk = ttk
        self.pcd = None
        self.original = None
        self.input_path: Path | None = None
        self.undo_stack: list[o3d.geometry.PointCloud] = []
        self.output_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Open a point cloud to get started.")

        root.title("Point Cloud Editor")
        root.geometry("600x560")
        root.minsize(500, 450)

        files = ttk.LabelFrame(root, text="Files", padding=10)
        files.pack(fill="x", padx=10, pady=(10, 5))
        ttk.Button(files, text="Open point cloud...", command=self.open_file).pack(side="left")
        ttk.Button(files, text="Save output", command=self.save_file).pack(side="left", padx=(8, 0))
        ttk.Label(files, text="Output:").pack(side="left", padx=(16, 5))
        ttk.Entry(files, textvariable=self.output_var).pack(side="left", fill="x", expand=True)

        operations = ttk.LabelFrame(root, text="Edit operations", padding=10)
        operations.pack(fill="both", expand=True, padx=10, pady=5)
        buttons = [
            ("Crop box by picking two corners", self.crop_box_by_picking),
            ("Crop sphere (pick center)", self.crop_sphere_by_picking),
            ("Crop oriented box (pick center)", self.crop_oriented_box_by_picking),
            ("Keep picked points", self.keep_picked_points),
            ("Downsample", self.downsample_cloud),
            ("Remove outliers", self.remove_outliers),
            ("Translate", self.translate_cloud),
            ("Preview", self.preview_cloud),
        ]
        for index, (label, command) in enumerate(buttons):
            ttk.Button(operations, text=label, command=command).grid(
                row=index // 2, column=index % 2, sticky="ew", padx=5, pady=5
            )
        operations.columnconfigure(0, weight=1)
        operations.columnconfigure(1, weight=1)

        footer = ttk.Frame(root, padding=(10, 5, 10, 10))
        footer.pack(fill="x")
        ttk.Button(footer, text="Undo", command=self.undo).pack(side="left")
        ttk.Button(footer, text="Reset edits", command=self.reset).pack(side="left", padx=8)
        ttk.Label(footer, textvariable=self.status_var, wraplength=430).pack(
            side="left", padx=10, fill="x", expand=True
        )

        if initial_input is not None:
            self.load_file(initial_input)
            if initial_output is not None and self.pcd is not None:
                self.output_var.set(str(initial_output))

    def _require_cloud(self) -> bool:
        if self.pcd is None:
            from tkinter import messagebox

            messagebox.showwarning("No point cloud", "Open a point cloud first.", parent=self.root)
            return False
        return True

    def _update_status(self, message: str | None = None) -> None:
        if self.pcd is None:
            self.status_var.set(message or "Open a point cloud to get started.")
            return
        self.status_var.set(
            message or f"{len(self.pcd.points):,} points loaded from {self.input_path.name}"
        )

    def load_file(self, path: Path) -> None:
        from tkinter import messagebox

        try:
            pcd = load_pcd(path)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            messagebox.showerror("Could not open point cloud", str(exc), parent=self.root)
            return
        self.pcd = pcd
        self.original = o3d.geometry.PointCloud(pcd)
        self.input_path = path
        self.undo_stack.clear()
        self.output_var.set(str(path.with_name(f"{path.stem}_edited.pcd")))
        self._update_status()

    def open_file(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            parent=self.root,
            title="Open point cloud",
            filetypes=[("Point clouds", "*.pcd *.ply *.xyz *.xyzn *.xyzrgb *.pts"), ("All files", "*.*")],
        )
        if path:
            self.load_file(Path(path))

    def save_file(self) -> None:
        from tkinter import filedialog, messagebox

        if not self._require_cloud():
            return
        path = self.output_var.get().strip()
        if not path:
            path = filedialog.asksaveasfilename(
                parent=self.root,
                title="Save point cloud",
                defaultextension=".pcd",
                filetypes=[("PCD point cloud", "*.pcd"), ("PLY point cloud", "*.ply"), ("All files", "*.*")],
            )
            if not path:
                return
            self.output_var.set(path)
        try:
            save_pcd(self.pcd, path)
        except (OSError, RuntimeError, ValueError) as exc:
            messagebox.showerror("Could not save point cloud", str(exc), parent=self.root)
            return
        self._update_status(f"Saved {len(self.pcd.points):,} points to {path}")

    def _apply(self, operation) -> None:
        from tkinter import messagebox

        if not self._require_cloud():
            return
        working = o3d.geometry.PointCloud(self.pcd)
        try:
            result = operation(working)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            messagebox.showerror("Edit failed", str(exc), parent=self.root)
            return
        self.undo_stack.append(self.pcd)
        self.pcd = result
        self._update_status()

    def _pick_points(self, title: str, minimum: int = 1) -> list[int] | None:
        from tkinter import messagebox

        if not self._require_cloud():
            return None
        messagebox.showinfo(
            title,
            f"In the 3D window, Shift-click at least {minimum} point(s), then close the window.",
            parent=self.root,
        )
        indices = _pick_point_indices(self.pcd, title)
        if len(indices) < minimum:
            messagebox.showwarning(
                title,
                f"Pick at least {minimum} point(s). No changes were made.",
                parent=self.root,
            )
            return None
        return indices

    def crop_box_by_picking(self) -> None:
        indices = self._pick_points("Pick crop-box corners", minimum=2)
        if indices is None:
            return
        selected = np.asarray(self.pcd.points)[indices]
        if np.unique(selected, axis=0).shape[0] < 2:
            from tkinter import messagebox

            messagebox.showwarning(
                "Pick crop-box corners",
                "Pick two different points to define the crop box. No changes were made.",
                parent=self.root,
            )
            return
        low = selected.min(axis=0)
        high = selected.max(axis=0)
        self._apply(lambda pcd: crop_box(pcd, low, high))

    def crop_sphere_by_picking(self) -> None:
        from tkinter import simpledialog

        if not self._require_cloud():
            return
        radius = simpledialog.askfloat(
            "Crop sphere", "Radius:", parent=self.root, minvalue=0.000001
        )
        if radius is None:
            return
        indices = self._pick_points("Pick sphere center")
        if indices is None:
            return
        center = np.asarray(self.pcd.points)[indices[0]]
        self._apply(lambda pcd: crop_sphere(pcd, center, radius))

    def crop_oriented_box_by_picking(self) -> None:
        from tkinter import simpledialog

        if not self._require_cloud():
            return
        values = simpledialog.askstring(
            "Crop oriented box",
            "Enter box extents and yaw in degrees (width,height,depth,yaw):",
            initialvalue="10,10,10,0",
            parent=self.root,
        )
        if values is None:
            return
        try:
            parsed = _parse_floats(values)
            if len(parsed) != 4 or min(parsed[:3]) <= 0:
                raise ValueError("Enter three positive extents and one yaw value.")
        except ValueError as exc:
            from tkinter import messagebox

            messagebox.showerror("Invalid extents", str(exc), parent=self.root)
            return
        indices = self._pick_points("Pick oriented-box center")
        if indices is None:
            return
        center = np.asarray(self.pcd.points)[indices[0]]
        self._apply(lambda pcd: crop_oriented_box(pcd, center, parsed[:3], parsed[3]))

    def keep_picked_points(self) -> None:
        indices = self._pick_points("Pick points to keep")
        if indices is not None:
            self._apply(lambda pcd: _report_crop(
                pcd, pcd.select_by_index(indices), "picked points"
            ))

    def downsample_cloud(self) -> None:
        from tkinter import simpledialog

        if not self._require_cloud():
            return
        voxel = simpledialog.askfloat(
            "Downsample", "Voxel size in meters:", initialvalue=0.05,
            minvalue=0.000001, parent=self.root
        )
        if voxel is not None:
            self._apply(lambda pcd: downsample(pcd, voxel))

    def remove_outliers(self) -> None:
        from tkinter import simpledialog

        if not self._require_cloud():
            return
        neighbors = simpledialog.askinteger(
            "Remove outliers", "Number of neighbors:", initialvalue=20,
            minvalue=1, parent=self.root
        )
        if neighbors is None:
            return
        ratio = simpledialog.askfloat(
            "Remove outliers", "Standard-deviation ratio:", initialvalue=2.0,
            minvalue=0.000001, parent=self.root
        )
        if ratio is not None:
            self._apply(lambda pcd: remove_outliers(pcd, neighbors, ratio))

    def translate_cloud(self) -> None:
        from tkinter import simpledialog

        if not self._require_cloud():
            return
        values = simpledialog.askstring(
            "Translate", "Enter shift (dx,dy,dz):", initialvalue="0,0,0",
            parent=self.root
        )
        if values is None:
            return
        try:
            xyz = _parse_floats(values)
            if len(xyz) != 3:
                raise ValueError("Enter exactly three comma-separated values.")
        except ValueError as exc:
            from tkinter import messagebox

            messagebox.showerror("Invalid shift", str(exc), parent=self.root)
            return
        self._apply(lambda pcd: translate(pcd, xyz))

    def preview_cloud(self) -> None:
        if self._require_cloud():
            preview(self.pcd)

    def undo(self) -> None:
        if self.undo_stack:
            self.pcd = self.undo_stack.pop()
            self._update_status("Last edit undone.")

    def reset(self) -> None:
        if self.original is not None:
            self.undo_stack.append(self.pcd)
            self.pcd = o3d.geometry.PointCloud(self.original)
            self._update_status("Edits reset to the original point cloud.")


def run_gui(
    initial_input: Path | None = None,
    initial_output: Path | None = None,
) -> int:
    import tkinter as tk

    root = tk.Tk()
    PcdEditorApp(root, initial_input, initial_output)
    root.mainloop()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Edit a PCD point cloud file.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--gui", action="store_true", help="Open the point-cloud editing GUI.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PCD/PLY path (default: <input>_edited.pcd next to the input).",
    )
    parser.add_argument(
        "--crop-box",
        type=str,
        help="Axis-aligned crop: xmin,ymin,zmin,xmax,ymax,zmax.",
    )
    parser.add_argument(
        "--crop-sphere",
        type=str,
        help="Sphere crop: cx,cy,cz,radius.",
    )
    parser.add_argument(
        "--crop-oriented-box",
        type=str,
        help="Rotated crop: cx,cy,cz,ex,ey,ez,yaw_deg.",
    )
    parser.add_argument("--pick", action="store_true", help="Interactively pick points to keep.")
    parser.add_argument("--voxel", type=float, help="Voxel downsample size in meters.")
    parser.add_argument("--remove-outliers", action="store_true")
    parser.add_argument("--translate", type=str, help="Shift the cloud: dx,dy,dz.")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    if args.gui or args.input is None:
        try:
            return run_gui(args.input, args.output)
        except (ImportError, RuntimeError) as exc:
            print(f"error: could not start GUI: {exc}")
            return 1

    try:
        pcd = load_pcd(args.input)
        if args.crop_box:
            vals = _parse_floats(args.crop_box)
            if len(vals) != 6:
                raise ValueError("--crop-box needs xmin,ymin,zmin,xmax,ymax,zmax")
            pcd = crop_box(pcd, vals[:3], vals[3:])
        if args.crop_sphere:
            vals = _parse_floats(args.crop_sphere)
            if len(vals) != 4:
                raise ValueError("--crop-sphere needs cx,cy,cz,radius")
            pcd = crop_sphere(pcd, vals[:3], vals[3])
        if args.crop_oriented_box:
            vals = _parse_floats(args.crop_oriented_box)
            if len(vals) != 7:
                raise ValueError("--crop-oriented-box needs cx,cy,cz,ex,ey,ez,yaw_deg")
            pcd = crop_oriented_box(pcd, vals[:3], vals[3:6], vals[6])
        if args.pick:
            pcd = pick_points(pcd)
        if args.remove_outliers:
            pcd = remove_outliers(pcd)
        if args.voxel:
            pcd = downsample(pcd, args.voxel)
        if args.translate:
            vals = _parse_floats(args.translate)
            if len(vals) != 3:
                raise ValueError("--translate needs dx,dy,dz")
            pcd = translate(pcd, vals)

        if args.preview:
            preview(pcd)

        output = args.output or args.input.with_name(f"{args.input.stem}_edited.pcd")
        save_pcd(pcd, output)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
