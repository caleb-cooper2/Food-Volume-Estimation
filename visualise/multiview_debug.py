"""
Render the multi-view pipeline for presentation use: the input photos, the VGGT reconstruction, the per-item masks,
and the Poisson mesh from the fused world cloud.

Usage:
    python visualise/multiview_debug.py \
        --images ./data/multiview/*.png \
        --output ./visualise/multiview_debug.png
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from PIL import Image
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

from logging_config import get_logger
from segmentation import segment_food
from volume_mesh import create_poisson_mesh, clean_food_points
from approaches.multiview import fit_table_plane, align_depth_to_vggt

logger = get_logger(__name__)


def build_debug_scene(image_paths: list[Path], output_path: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = (
        torch.bfloat16
        if device.type == "cpu" or (device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8)
        else torch.float16
    )

    pil_images = [Image.open(path).convert("RGB") for path in image_paths]
    primary_image = pil_images[0]

    food_mask, mask_scores, instance_masks = segment_food(primary_image)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_paths = []
        for i, image in enumerate(pil_images):
            tmp_path = os.path.join(tmp_dir, f"temp_image_{i}.png")
            image.save(tmp_path)
            tmp_paths.append(tmp_path)
        images_tensor = load_and_preprocess_images(tmp_paths).to(device)

    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            predictions = vggt_model(images_tensor)

    world_points = predictions["world_points"][0, 0].cpu().numpy()

    food_mask_resized = (
            align_depth_to_vggt(food_mask.astype(np.float32), primary_image.width, primary_image.height) > 0.5
    ).astype(np.uint8)
    table_plane = fit_table_plane(world_points, food_mask_resized)

    mesh_dir = output_path.parent / "multiview_mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    fused_food_pts = world_points[food_mask_resized > 0]
    cleaned_food_pts = clean_food_points(fused_food_pts, table_plane)
    mesh = create_poisson_mesh(cleaned_food_pts, table_plane, camera_location=np.zeros(3))
    mesh_path = mesh_dir / "food_mesh_debug.ply"
    mesh.export(mesh_path)

    fig = plt.figure(figsize=(18, 10), facecolor="#f5f5f3")
    gs = fig.add_gridspec(2, 3)

    input_titles = ["Input view 1", "Input view 2", "Input view 3"]
    for i, image in enumerate(pil_images):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(image)
        ax.set_title(input_titles[i], fontsize=12)
        ax.axis("off")

    ax_mask = fig.add_subplot(gs[1, 0])
    ax_mask.imshow(primary_image)
    rgba_mask = np.zeros((*food_mask.shape, 4), dtype=np.float32)
    rgba_mask[food_mask.astype(bool)] = [0.0, 0.85, 0.3, 0.55]
    ax_mask.imshow(rgba_mask)
    ax_mask.set_title("Pixel masks (SAM 3)", fontsize=12)
    ax_mask.axis("off")

    rgb_flat = images_tensor[0].permute(1, 2, 0).float().cpu().numpy().reshape(-1, 3)  # (H*W, 3), aligned with world_points
    conf = predictions["world_points_conf"][0, 0].float().cpu().numpy()

    pts_flat = world_points.reshape(-1, 3)
    food_flat = food_mask_resized.reshape(-1) > 0
    high_conf = conf.reshape(-1) > np.percentile(conf, 30)

    food_center = np.median(pts_flat[food_flat], axis=0)
    dist = np.linalg.norm(pts_flat - food_center, axis=1)
    scene_radius = 2.5 * np.percentile(dist[food_flat], 95)
    near_food = dist < scene_radius

    fg_mask = high_conf & food_flat
    bg_mask = high_conf & near_food & ~food_flat

    def to_plot(pts):  # OpenCV camera frame -> matplotlib (z up)
        pts = pts - food_center
        return pts[:, 0], pts[:, 2], -pts[:, 1]

    rng = np.random.default_rng(0)

    fg_idx = np.where(fg_mask)[0]
    fg_idx = rng.choice(fg_idx, min(2500, len(fg_idx)), replace=False)
    bg_idx = rng.choice(np.where(bg_mask)[0], min(12000, bg_mask.sum()), replace=False)

    ax_cloud = fig.add_subplot(gs[1, 1:3], projection="3d")
    ax_cloud.scatter(*to_plot(pts_flat[bg_idx]), c=rgb_flat[bg_idx], s=3, alpha=0.5,
                     depthshade=True, linewidths=0, rasterized=True)
    ax_cloud.scatter(*to_plot(pts_flat[fg_idx]), c=rgb_flat[fg_idx], s=3, alpha=0.9,
                     depthshade=True, linewidths=0, rasterized=True)

    lim = scene_radius
    ax_cloud.set_xlim(-lim, lim)
    ax_cloud.set_ylim(-lim, lim)
    ax_cloud.set_zlim(-lim, lim)
    ax_cloud.set_box_aspect((1, 1, 1))
    ax_cloud.view_init(elev=40, azim=-35)
    ax_cloud.set_axis_off()
    ax_cloud.set_title("VGGT point cloud (food highlighted)", fontsize=12)

    fig.suptitle("Multi-view reconstruction", fontsize=16, fontweight="bold", color="#111827")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"Saved multi-view debug render -> {output_path}")
    logger.info(f"Saved mesh -> {mesh_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render the multi-view pipeline as a debug figure")
    parser.add_argument("--images", type=Path, nargs="+", required=True, help="Input image paths for the multi-view reconstruction")
    parser.add_argument("--output", type=Path, default=Path("visualise/multiview_debug.png"), help="Output PNG")
    args = parser.parse_args()

    build_debug_scene(sorted(args.images), args.output)
