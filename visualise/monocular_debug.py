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
from schemas import CameraInfo as CameraInfoType
from segmentation import segment_food

logger = get_logger(__name__)


def build_debug_frame(image_path: Path, output_path: Path, prompt: str = "food or drink") -> None:
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
    plane_n, plane_p0, inlier_ratio = fit_support_plane(depth_map, food_mask, cam)
    volume_res = compute_volume(depth_map, food_mask, cam, plane_n, plane_p0)

    ys, xs = np.where(food_mask > 0)
    z = depth_map[ys, xs].astype(np.float64)
    x = (xs - cam.cx) * z / cam.fx
    y = (ys - cam.cy) * z / cam.fy
    pts = np.stack([x, y, z], axis=1)
    heights = np.clip((pts - plane_p0) @ plane_n, 0.0, None)
    height_map = np.zeros_like(depth_map, dtype=np.float32)
    height_map[ys, xs] = heights * 100.0

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="#f5f5f3")
    fig.subplots_adjust(wspace=0.28, hspace=0.22)

    axes[0, 0].imshow(pillow_image)
    axes[0, 0].set_title("Input image", fontsize=12, color="#1f2937")
    axes[0, 0].axis("off")

    depth_im = axes[0, 1].imshow(depth_map, cmap="plasma")
    axes[0, 1].set_title("Depth map (m)", fontsize=12, color="#1f2937")
    axes[0, 1].axis("off")
    cbar = fig.colorbar(depth_im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cbar.set_label("metres", color="#1f2937")

    rgba_mask = np.zeros((*food_mask.shape, 4), dtype=np.float32)
    rgba_mask[food_mask.astype(bool)] = [0.0, 0.85, 0.3, 0.45]
    axes[0, 2].imshow(pillow_image)
    axes[0, 2].imshow(rgba_mask)
    axes[0, 2].set_title("SAM 3 mask", fontsize=12, color="#1f2937")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(pillow_image)
    axes[1, 0].imshow(rgba_mask)
    axes[1, 0].text(
        12, 20,
        f"normal=[{plane_n[0]:.2f}, {plane_n[1]:.2f}, {plane_n[2]:.2f}]\n"
        f"plate_z={plane_p0[2]:.3f}m\n"
        f"inliers={inlier_ratio:.2f}",
        color="white",
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f172a", edgecolor="#94a3b8", alpha=0.7)
    )
    axes[1, 0].set_title("RANSAC plane fit", fontsize=12, color="#1f2937")
    axes[1, 0].axis("off")

    height_im = axes[1, 1].imshow(height_map, cmap="viridis")
    axes[1, 1].set_title("Height field above plane (cm)", fontsize=12, color="#1f2937")
    axes[1, 1].axis("off")
    cbar_h = fig.colorbar(height_im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar_h.set_label("cm", color="#1f2937")

    axes[1, 2].set_axis_off()
    text_lines = [
        "Monocular geometric estimate",
        "",
        f"Volume: {volume_res.volume_cm3:.1f} cm³",
        f"Max height: {volume_res.max_food_height_cm:.1f} cm",
        f"Mean height: {volume_res.mean_food_height_cm:.1f} cm",
        f"Confidence: {volume_res.geometry_confidence:.2f}",
        "",
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
    args = parser.parse_args()

    build_debug_frame(image_path=args.image, output_path=args.output, prompt=args.prompt)
