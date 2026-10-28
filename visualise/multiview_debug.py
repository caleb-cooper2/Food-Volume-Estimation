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

import cv2
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
from approaches.multiview import fit_table_plane

logger = get_logger(__name__)

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def build_debug_scene(image_paths: list[Path], output_path: Path) -> None:
    pil_images = [Image.open(path).convert("RGB") for path in image_paths]
    primary = pil_images[0]

    food_mask, mask_scores, instance_masks = segment_food(primary)

    with tempfile.TemporaryDirectory() as temp_directory:
        paths = []
        for i, image in enumerate(pil_images):
            path = os.path.join(temp_directory, f"temp_image_{i}.png")
            image.save(path)
            paths.append(path)
        images_tensor = load_and_preprocess_images(paths)

    vggt_model = VGGT.from_pretrained("facebook/VGGT-1B")
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=dtype):
            predictions = vggt_model(images_tensor.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")))

    world_points = predictions["world_points"][0, 0].cpu().numpy()
    vggt_depth = predictions["depth"][0, 0].cpu().numpy().squeeze(-1)

    target_h, target_w = vggt_depth.shape
    food_mask_resized = cv2.resize(food_mask.astype(np.uint8), (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    world_points_metric = world_points
    table_plane = fit_table_plane(world_points_metric, food_mask_resized)

    mesh_dir = output_path.parent / "multiview_mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    fused_points = world_points_metric[food_mask_resized > 0]
    cleaned = clean_food_points(fused_points, table_plane)
    mesh = create_poisson_mesh(cleaned, table_plane, camera_location=np.zeros(3))
    mesh_path = mesh_dir / "food_mesh_debug.ply"
    mesh.export(mesh_path)

    fig = plt.figure(figsize=(18, 10), facecolor="#f5f5f3")
    gs = fig.add_gridspec(2, 3)

    ax_input = fig.add_subplot(gs[0, 0])
    ax_input.imshow(primary)
    ax_input.set_title("Input photo", fontsize=12)
    ax_input.axis("off")

    ax_masks = fig.add_subplot(gs[0, 1])
    ax_masks.imshow(primary)
    rgba_mask = np.zeros((*food_mask.shape, 4), dtype=np.float32)
    rgba_mask[food_mask.astype(bool)] = [0.0, 0.85, 0.3, 0.55]
    ax_masks.imshow(rgba_mask)
    ax_masks.set_title("Pixel masks (SAM 3)", fontsize=12)
    ax_masks.axis("off")

    ax_cloud = fig.add_subplot(gs[0, 2], projection="3d")
    subsample = np.random.default_rng(0).choice(len(world_points_metric.reshape(-1, 3)), min(4000, len(world_points_metric.reshape(-1, 3))), replace=False)
    pts = world_points_metric.reshape(-1, 3)[subsample]
    scatter = ax_cloud.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap="viridis", s=5, alpha=0.7)
    ax_cloud.set_title("VGGT point cloud", fontsize=12)
    ax_cloud.set_xlabel("X")
    ax_cloud.set_ylabel("Y")
    ax_cloud.set_zlabel("Z")
    fig.colorbar(scatter, ax=ax_cloud, pad=0.08, shrink=0.7)

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
