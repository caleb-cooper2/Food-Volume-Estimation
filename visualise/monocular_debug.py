"""
Go through one monocular photo through the geometric pipeline and save a human-readable single figure summarising input, depth map, SAM 3 mask,
RANSAC support plane, height-field volume, and the final mass estimate.

Usage:
    python visualise/monocular_debug.py \
        --image ./data/your_photo.jpg \
        --output ./visualise/monocular_debug.png \
        --prompt "food or drink"
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from depth import estimate_depth
from geometry import compute_volume, extract_image_info, fit_support_plane
from logging_config import get_logger
from scale import resolve_scale_correction
from schemas import CameraInfo as CameraInfoType
from segmentation import segment_food

logger = get_logger(__name__)


def build_debug_frame(image_path: Path, output_path: Path, prompt: str = "food or drink", scale_ref: str = "size_prior") -> None:
    pillow_image = Image.open(image_path).convert("RGB")
    image_bytes = image_path.read_bytes()
    image_info = extract_image_info(image_bytes, pillow_image.width, pillow_image.height)

    depth_map, focal_length_px = estimate_depth(pillow_image)
    food_mask, scores, _ = segment_food(pillow_image, prompt=prompt)
    cam = CameraInfoType(
        fx=focal_length_px,
        fy=focal_length_px,
        cx=pillow_image.width / 2.0,
        cy=pillow_image.height / 2.0,
        image_width=pillow_image.width,
        image_height=pillow_image.height,
        source="depthpro_fov"
    )

    scale_correction, scale_source = resolve_scale_correction(pillow_image, depth_map, food_mask, cam, scale_ref)
    if scale_correction is not None:
        depth_map = depth_map * scale_correction

    plane_n, plane_p0, inlier_ratio = fit_support_plane(depth_map, food_mask, cam)
    volume_res = compute_volume(depth_map, food_mask, cam, plane_n, plane_p0)

    food_ys, food_xs = np.where(food_mask > 0)
    food_depth = depth_map[food_ys, food_xs].astype(np.float64)
    food_world_x = (food_xs - cam.cx) * food_depth / cam.fx
    food_world_y = (food_ys - cam.cy) * food_depth / cam.fy
    food_world_pts = np.stack([food_world_x, food_world_y, food_depth], axis=1)
    food_heights_m = np.clip((food_world_pts - plane_p0) @ plane_n, 0.0, None)
    height_map = np.zeros_like(depth_map, dtype=np.float32)
    height_map[food_ys, food_xs] = food_heights_m * 100.0

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="#f5f5f3")
    fig.subplots_adjust(wspace=0.28, hspace=0.22)

    axes[0, 0].imshow(pillow_image)
    axes[0, 0].set_title("Input image", fontsize=12, color="#1f2937")
    axes[0, 0].axis("off")

    depth_cmap = axes[0, 1].imshow(depth_map, cmap="plasma")
    axes[0, 1].set_title("Depth map (m)", fontsize=12, color="#1f2937")
    axes[0, 1].axis("off")
    depth_cbar = fig.colorbar(depth_cmap, ax=axes[0, 1], fraction=0.046, pad=0.04)
    depth_cbar.set_label("metres", color="#1f2937")

    rgba_mask = np.zeros((*food_mask.shape, 4), dtype=np.float32)
    rgba_mask[food_mask.astype(bool)] = [0.0, 0.85, 0.3, 0.45]
    axes[0, 2].imshow(pillow_image)
    axes[0, 2].imshow(rgba_mask)
    axes[0, 2].set_title("SAM 3 mask", fontsize=12, color="#1f2937")
    axes[0, 2].axis("off")

    plane_normal = plane_n / np.linalg.norm(plane_n)
    tangent_u = np.cross(plane_normal, [0.0, 1.0, 0.0]); tangent_u /= np.linalg.norm(tangent_u)
    tangent_v = np.cross(plane_normal, tangent_u)

    img_h, img_w = depth_map.shape
    row_grid, col_grid = np.mgrid[0:img_h, 0:img_w]
    depth_all = depth_map.astype(np.float64)
    world_pts_all = np.stack(
        [(col_grid - cam.cx) * depth_all / cam.fx,
         (row_grid - cam.cy) * depth_all / cam.fy,
         depth_all],
        axis=-1
    )

    pts_flat = world_pts_all.reshape(-1, 3)
    offset = pts_flat - plane_p0
    coord_u = offset @ tangent_u
    coord_v = offset @ tangent_v
    height_above_plane = offset @ plane_normal
    rgb_flat = np.asarray(pillow_image, dtype=np.float32).reshape(-1, 3) / 255.0
    food_flat = food_mask.reshape(-1) > 0

    center_u = np.median(coord_u[food_flat])
    center_v = np.median(coord_v[food_flat])
    scene_radius = 2.5 * max(np.ptp(coord_u[food_flat]), np.ptp(coord_v[food_flat]))
    near_food = (np.abs(coord_u - center_u) < scene_radius) & (np.abs(coord_v - center_v) < scene_radius)

    rng = np.random.default_rng(0)
    bg_idx = rng.choice(np.where(near_food & ~food_flat)[0], 8000, replace=False)
    fg_idx = rng.choice(np.where(food_flat)[0], 3000, replace=False)

    axes[1, 0].remove()
    plane_axis = fig.add_subplot(2, 3, 4, projection="3d")
    plane_axis.scatter(
        coord_u[bg_idx], coord_v[bg_idx], height_above_plane[bg_idx],
        c=rgb_flat[bg_idx], s=2, alpha=0.5, linewidths=0, rasterized=True
    )
    plane_axis.scatter(
        coord_u[fg_idx], coord_v[fg_idx], height_above_plane[fg_idx],
        c=rgb_flat[fg_idx], s=3, alpha=0.95, linewidths=0, rasterized=True
    )

    grid_u, grid_v = np.meshgrid(
        np.linspace(center_u - scene_radius, center_u + scene_radius, 2),
        np.linspace(center_v - scene_radius, center_v + scene_radius, 2)
    )
    plane_axis.plot_surface(grid_u, grid_v, np.zeros_like(grid_u), color="#3b82f6", alpha=0.18, shade=False)

    plane_axis.set_xlim(center_u - scene_radius, center_u + scene_radius)
    plane_axis.set_ylim(center_v - scene_radius, center_v + scene_radius)
    plane_axis.set_zlim(-scene_radius / 2, scene_radius)
    plane_axis.set_box_aspect((1, 1, 0.6))
    plane_axis.view_init(elev=22, azim=-55)
    plane_axis.set_title("Fitted support plane", fontsize=12, color="#1f2937")

    height_cmap = axes[1, 1].imshow(height_map, cmap="viridis")
    axes[1, 1].set_title("Height field above plane (cm)", fontsize=12, color="#1f2937")
    axes[1, 1].axis("off")
    height_cbar = fig.colorbar(height_cmap, ax=axes[1, 1], fraction=0.046, pad=0.04)
    height_cbar.set_label("cm", color="#1f2937")

    axes[1, 2].set_axis_off()
    scale_line = (
        f"Scale x{scale_correction:.3f} ({scale_source})"
        if scale_correction is not None
        else f"Scale: none ({scale_source})"
    )
    text_lines = [
        "Monocular geometric estimate",
        "",
        f"Volume: {volume_res.volume_cm3:.1f} cm³",
        f"Max height: {volume_res.max_food_height_cm:.1f} cm",
        f"Mean height: {volume_res.mean_food_height_cm:.1f} cm",
        f"Confidence: {volume_res.geometry_confidence:.2f}",
        "",
        scale_line,
        f"Plate depth: {plane_p0[2]:.3f}m",
        f"Food px: {int(food_mask.sum())}",
        f"Depth source: {image_info.source}"
    ]

    axes[1, 2].text(
        0.05, 0.98,
        "\n".join(text_lines),
        va="top",
        ha="left",
        fontsize=11,
        color="#111827",
        family="monospace",
        transform=axes[1, 2].transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#d1d5db", alpha=0.9)
    )

    fig.suptitle(
        f"Monocular geometric pipeline",
        fontsize=16,
        fontweight="bold",
        color="#111827",
        y=0.98
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"Saved debug render -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render the monocular geometric pipeline as a debug figure")
    parser.add_argument("--image", type=Path, required=True, help="Single RGB input image")
    parser.add_argument("--output", type=Path, default=Path("visualise/monocular_debug.png"), help="Output PNG")
    parser.add_argument("--prompt", type=str, default="food or drink", help="SAM 3 prompt for the food region")
    parser.add_argument("--scale-ref", type=str, default="size_prior", help="Scale anchor: utensil, size_prior, auto, or none")
    args = parser.parse_args()

    build_debug_frame(image_path=args.image, output_path=args.output, prompt=args.prompt, scale_ref=args.scale_ref)
