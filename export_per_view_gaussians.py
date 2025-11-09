#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#

"""
Export per-view Gaussian subsets and rendered images.
This script extracts which Gaussians are used for rendering each training image.
"""

import os
import torch
import json
import numpy as np
from argparse import ArgumentParser
from scene import Scene, GaussianModel
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams
from utils.general_utils import safe_state
from PIL import Image
from tqdm import tqdm
from plyfile import PlyData, PlyElement


def save_gaussian_subset(gaussians, indices, output_path):
    """
    Save a subset of Gaussians to PLY file.

    Args:
        gaussians: GaussianModel object
        indices: tensor of indices to save
        output_path: path to save the PLY file
    """
    # Convert indices to numpy
    if isinstance(indices, torch.Tensor):
        indices = indices.cpu().numpy().flatten()

    # Extract Gaussian parameters for selected indices
    xyz = gaussians._xyz.detach()[indices].cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = (
        gaussians._features_dc.detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()[indices]
        .cpu()
        .numpy()
    )
    f_rest = (
        gaussians._features_rest.detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()[indices]
        .cpu()
        .numpy()
    )
    opacities = gaussians._opacity.detach()[indices].cpu().numpy()
    scale = gaussians._scaling.detach()[indices].cpu().numpy()
    rotation = gaussians._rotation.detach()[indices].cpu().numpy()

    # Construct PLY attributes
    l = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(f_dc.shape[1]):
        l.append("f_dc_{}".format(i))
    for i in range(f_rest.shape[1]):
        l.append("f_rest_{}".format(i))
    l.append("opacity")
    for i in range(scale.shape[1]):
        l.append("scale_{}".format(i))
    for i in range(rotation.shape[1]):
        l.append("rot_{}".format(i))

    dtype_full = [(attribute, "f4") for attribute in l]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate(
        (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
    )
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(output_path)


def export_per_view_gaussians(dataset, iteration, pipeline, output_dir, save_ply=True, save_indices=True, max_views=None):
    """
    Export Gaussian subsets and rendered images for each training view.

    Args:
        dataset: ModelParams
        iteration: which iteration's model to load
        pipeline: PipelineParams
        output_dir: directory to save outputs
        save_ply: whether to save PLY files (slow for large models)
        save_indices: whether to save visible indices (uses .npy format)
        max_views: maximum number of views to export (None for all)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    # Get background color
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Get training cameras
    train_cameras = scene.getTrainCameras()

    # Limit number of views if specified
    if max_views is not None and max_views > 0:
        train_cameras = train_cameras[:max_views]
        print(f"Limiting to first {max_views} views")

    # Statistics
    total_gaussians = gaussians.get_xyz.shape[0]
    per_view_stats = []

    print(f"Total Gaussians in model: {total_gaussians}")
    print(f"Processing {len(train_cameras)} training views...")
    if not save_ply:
        print("  [FAST MODE] Skipping PLY file generation")
    if not save_indices:
        print("  [FAST MODE] Skipping index saving")

    # Process each training view
    for idx, camera in enumerate(tqdm(train_cameras, desc="Exporting views")):
        view_name = camera.image_name
        view_dir = os.path.join(output_dir, f"view_{idx:04d}_{view_name}")
        os.makedirs(view_dir, exist_ok=True)

        # Render the view
        with torch.no_grad():
            render_pkg = render(camera, gaussians, pipeline, background)
            rendered_image = render_pkg["render"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]

        # Get indices of visible Gaussians
        visible_indices = visibility_filter.flatten()
        num_visible = len(visible_indices)

        # Save rendered image
        rendered_img_array = rendered_image.clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        rendered_img_array = (rendered_img_array * 255).astype(np.uint8)
        Image.fromarray(rendered_img_array).save(os.path.join(view_dir, "rendered.png"))

        # Save ground truth image
        gt_image = camera.original_image.cpu().permute(1, 2, 0).numpy()
        gt_image = (gt_image * 255).astype(np.uint8)
        Image.fromarray(gt_image).save(os.path.join(view_dir, "ground_truth.png"))

        # Save visible indices as numpy file (much faster than JSON)
        if save_indices and num_visible > 0:
            np.save(os.path.join(view_dir, "visible_indices.npy"), visible_indices.cpu().numpy())

        # Save Gaussian subset PLY (optional, can be slow)
        if save_ply and num_visible > 0:
            ply_path = os.path.join(view_dir, "gaussians_subset.ply")
            save_gaussian_subset(gaussians, visible_indices, ply_path)

        # Save camera parameters
        camera_params = {
            "image_name": view_name,
            "width": camera.image_width,
            "height": camera.image_height,
            "FoVx": camera.FoVx,
            "FoVy": camera.FoVy,
            "camera_center": camera.camera_center.cpu().tolist(),
            "world_view_transform": camera.world_view_transform.cpu().tolist(),
            "full_proj_transform": camera.full_proj_transform.cpu().tolist(),
        }

        # Save metadata (without large index arrays to keep JSON fast)
        metadata = {
            "view_index": idx,
            "image_name": view_name,
            "total_gaussians": total_gaussians,
            "visible_gaussians": num_visible,
            "visibility_ratio": num_visible / total_gaussians,
            "camera_params": camera_params,
        }

        # Add note about where to find indices
        if save_indices:
            metadata["visible_indices_file"] = "visible_indices.npy"

        with open(os.path.join(view_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # Collect statistics
        per_view_stats.append(
            {
                "view_index": idx,
                "image_name": view_name,
                "visible_gaussians": num_visible,
                "visibility_ratio": num_visible / total_gaussians,
            }
        )

    # Save overall statistics
    overall_stats = {
        "total_gaussians": total_gaussians,
        "total_views": len(train_cameras),
        "iteration": iteration,
        "per_view_stats": per_view_stats,
        "average_visible_gaussians": np.mean(
            [s["visible_gaussians"] for s in per_view_stats]
        ),
        "average_visibility_ratio": np.mean(
            [s["visibility_ratio"] for s in per_view_stats]
        ),
    }

    with open(os.path.join(output_dir, "overall_statistics.json"), "w") as f:
        json.dump(overall_stats, f, indent=2)

    print(f"\nExport completed!")
    print(
        f"Average visible Gaussians per view: {overall_stats['average_visible_gaussians']:.0f} ({overall_stats['average_visibility_ratio'] * 100:.1f}%)"
    )
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Export per-view Gaussian subsets")
    model_params = ModelParams(parser, sentinel=True)
    pipeline_params = PipelineParams(parser)
    parser.add_argument(
        "--iteration", default=-1, type=int, help="Iteration to load (-1 for latest)"
    )
    parser.add_argument(
        "--output_dir", default="./per_view_exports", type=str, help="Output directory"
    )
    parser.add_argument(
        "--skip-ply", action="store_true", help="Skip saving PLY files (faster)"
    )
    parser.add_argument(
        "--skip-indices", action="store_true", help="Skip saving index arrays (faster)"
    )
    parser.add_argument(
        "--max-views", type=int, default=None, help="Maximum number of views to export"
    )
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Load saved config if available
    cfg_args_path = os.path.join(args.model_path, "cfg_args")
    if os.path.exists(cfg_args_path):
        print(f"Loading config from {cfg_args_path}")
        with open(cfg_args_path, "r") as f:
            cfg_text = f.read()
            # Parse the saved arguments using eval (safer than direct execution)
            # The file contains: Namespace(key1=value1, key2=value2, ...)
            import re

            # Extract the content inside Namespace(...)
            match = re.search(r"Namespace\((.*)\)", cfg_text, re.DOTALL)
            if match:
                args_str = match.group(1)
                # Parse key=value pairs
                saved_args = {}
                # Split by comma, but be careful with nested structures
                for item in re.findall(
                    r"(\w+)=([^,]+(?:\([^)]*\))?(?:\[[^\]]*\])?)", args_str
                ):
                    key, value = item
                    # Try to evaluate the value
                    try:
                        saved_args[key] = eval(value)
                    except Exception:
                        saved_args[key] = value.strip().strip("'\"")

                # Update args with saved values if not specified
                for key, value in saved_args.items():
                    if not hasattr(args, key) or getattr(args, key) is None:
                        setattr(args, key, value)

    # Extract parameters
    model_params = model_params.extract(args)
    pipeline_params = pipeline_params.extract(args)

    # Export
    export_per_view_gaussians(
        model_params,
        args.iteration,
        pipeline_params,
        args.output_dir,
        save_ply=not args.skip_ply,
        save_indices=not args.skip_indices,
        max_views=args.max_views,
    )

    print("\nAll done!")

# Usage examples:
# Basic: python export_per_view_gaussians.py -m <model_path> --iteration 30000
# Fast mode: python export_per_view_gaussians.py -m <model_path> --iteration 30000 --skip-ply --skip-indices
# Limited views: python export_per_view_gaussians.py -m <model_path> --iteration 30000 --max-views 10
