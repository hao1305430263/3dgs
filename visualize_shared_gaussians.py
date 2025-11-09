#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#

"""
Visualize shared Gaussians between two views.
This script compares which Gaussians are used by different views.
"""

import os
import numpy as np
from argparse import ArgumentParser
from scene import Scene, GaussianModel
from arguments import ModelParams
from utils.general_utils import safe_state
from plyfile import PlyData, PlyElement
import json


def find_view_directory(export_dir, view_name):
    """
    Find the directory for a given view name.

    Args:
        export_dir: root export directory
        view_name: image name to search for

    Returns:
        path to the view directory
    """
    for dir_name in os.listdir(export_dir):
        if dir_name.endswith(view_name):
            return os.path.join(export_dir, dir_name)

    raise ValueError(f"View '{view_name}' not found in {export_dir}")


def load_visible_indices(view_dir):
    """Load visible indices from a view directory."""
    npy_path = os.path.join(view_dir, "visible_indices.npy")
    if os.path.exists(npy_path):
        return np.load(npy_path)

    # Fallback: try to load from metadata.json
    metadata_path = os.path.join(view_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            if "visible_indices" in metadata:
                return np.array(metadata["visible_indices"])

    raise ValueError(f"No visible indices found in {view_dir}")


def save_colored_point_cloud(positions, colors, output_path):
    """
    Save a simple point cloud PLY with positions and colors.

    Args:
        positions: Nx3 numpy array of XYZ positions
        colors: Nx3 numpy array of RGB colors (0-255)
        output_path: path to save PLY file
    """
    assert positions.shape[0] == colors.shape[0]

    # Construct PLY with positions and colors
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    elements = np.empty(positions.shape[0], dtype=dtype)
    elements["x"] = positions[:, 0]
    elements["y"] = positions[:, 1]
    elements["z"] = positions[:, 2]
    elements["red"] = colors[:, 0].astype(np.uint8)
    elements["green"] = colors[:, 1].astype(np.uint8)
    elements["blue"] = colors[:, 2].astype(np.uint8)

    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(output_path)


def visualize_shared_gaussians(
    model_path,
    iteration,
    export_dir,
    view1_name,
    view2_name,
    output_path,
    color_shared=(255, 0, 0),  # Red for shared
    color_view1=(0, 255, 0),  # Green for view1 only
    color_view2=(0, 0, 255),  # Blue for view2 only
):
    """
    Visualize shared Gaussians between two views.

    Args:
        model_path: path to trained model
        iteration: iteration to load
        export_dir: directory containing exported per-view data
        view1_name: name of first view
        view2_name: name of second view
        output_path: path to save output PLY
        color_shared: RGB color for shared Gaussians
        color_view1: RGB color for view1-only Gaussians
        color_view2: RGB color for view2-only Gaussians
    """
    print(f"Loading model from {model_path}, iteration {iteration}")

    # Load the model to get Gaussian positions
    from arguments import ModelParams
    import argparse

    # Create a dummy args object
    class Args:
        def __init__(self):
            self.sh_degree = 3
            self.source_path = None
            self.model_path = model_path
            self.images = "images"
            self.resolution = -1
            self.white_background = False
            self.data_device = "cuda"
            self.eval = False
            self.train_test_exp = False
            self.depths = ""

    args = Args()

    # Load saved config to get source_path
    cfg_args_path = os.path.join(model_path, "cfg_args")
    if os.path.exists(cfg_args_path):
        print(f"Loading config from {cfg_args_path}")
        with open(cfg_args_path, "r") as f:
            cfg_text = f.read()
            import re

            match = re.search(r"Namespace\((.*)\)", cfg_text, re.DOTALL)
            if match:
                args_str = match.group(1)
                saved_args = {}
                for item in re.findall(
                    r"(\w+)=([^,]+(?:\([^)]*\))?(?:\[[^\]]*\])?)", args_str
                ):
                    key, value = item
                    try:
                        saved_args[key] = eval(value)
                    except Exception:
                        saved_args[key] = value.strip().strip("'\"")

                for key, value in saved_args.items():
                    if hasattr(args, key):
                        setattr(args, key, value)

    # Load Gaussian model
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians, load_iteration=iteration, shuffle=False)

    # Get all Gaussian positions
    all_positions = gaussians.get_xyz.detach().cpu().numpy()
    total_gaussians = all_positions.shape[0]

    print(f"Total Gaussians in model: {total_gaussians}")

    # Load visible indices for both views
    print(f"\nLoading view 1: {view1_name}")
    view1_dir = find_view_directory(export_dir, view1_name)
    indices1 = load_visible_indices(view1_dir)
    print(
        f"  View 1 uses {len(indices1)} Gaussians ({len(indices1) / total_gaussians * 100:.1f}%)"
    )

    print(f"\nLoading view 2: {view2_name}")
    view2_dir = find_view_directory(export_dir, view2_name)
    indices2 = load_visible_indices(view2_dir)
    print(
        f"  View 2 uses {len(indices2)} Gaussians ({len(indices2) / total_gaussians * 100:.1f}%)"
    )

    # Convert to sets for intersection/difference
    set1 = set(indices1.tolist())
    set2 = set(indices2.tolist())

    # Calculate shared and unique
    shared = set1 & set2
    only1 = set1 - set2
    only2 = set2 - set1

    print(f"\nAnalysis:")
    print(
        f"  Shared by both views: {len(shared)} ({len(shared) / total_gaussians * 100:.1f}%)"
    )
    print(f"  Only in view 1: {len(only1)} ({len(only1) / total_gaussians * 100:.1f}%)")
    print(f"  Only in view 2: {len(only2)} ({len(only2) / total_gaussians * 100:.1f}%)")
    print(f"  Overlap ratio: {len(shared) / len(set1 | set2) * 100:.1f}%")

    # Collect positions and colors
    positions_list = []
    colors_list = []

    # Shared Gaussians
    for idx in shared:
        positions_list.append(all_positions[idx])
        colors_list.append(color_shared)

    # View 1 only
    for idx in only1:
        positions_list.append(all_positions[idx])
        colors_list.append(color_view1)

    # View 2 only
    for idx in only2:
        positions_list.append(all_positions[idx])
        colors_list.append(color_view2)

    positions = np.array(positions_list)
    colors = np.array(colors_list)

    # Save PLY
    print(f"\nSaving visualization to {output_path}")
    save_colored_point_cloud(positions, colors, output_path)

    print(f"\nDone! Point cloud saved with:")
    print(f"  - RED points: Shared by both views ({len(shared)} points)")
    print(f"  - GREEN points: Only in '{view1_name}' ({len(only1)} points)")
    print(f"  - BLUE points: Only in '{view2_name}' ({len(only2)} points)")
    print(f"\nTotal points in visualization: {len(positions)}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Visualize shared Gaussians between two views")
    parser.add_argument(
        "-m", "--model_path", required=True, type=str, help="Path to trained model"
    )
    parser.add_argument("--iteration", default=-1, type=int, help="Iteration to load")
    parser.add_argument(
        "--export_dir",
        required=True,
        type=str,
        help="Directory with exported per-view data",
    )
    parser.add_argument(
        "--view1", required=True, type=str, help="First view image name"
    )
    parser.add_argument(
        "--view2", required=True, type=str, help="Second view image name"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="./shared_gaussians.ply",
        type=str,
        help="Output PLY path",
    )
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    # Initialize system state
    safe_state(args.quiet)

    # Visualize
    visualize_shared_gaussians(
        args.model_path,
        args.iteration,
        args.export_dir,
        args.view1,
        args.view2,
        args.output,
    )

    print("\nAll done!")

# Usage example:
# python visualize_shared_gaussians.py -m ./output/xxx --iteration 30000 --export_dir ./per_view_exports --view1 image_001 --view2 image_002 -o shared.ply
