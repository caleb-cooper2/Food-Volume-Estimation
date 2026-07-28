"""Base64 PNG renders of the depth map and the segmentation overlay, for eyeballing what a request actually saw"""

import base64
import io
from typing import Optional

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


def depth_to_b64_png(depth_map: np.ndarray, pil_image: Optional[Image.Image] = None) -> str:
    """Render depth map and optionally input image side-by-side as base64 PNG"""
    n_panels = 2 if pil_image is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5), dpi=100)

    if n_panels == 1:
        axes = [axes]

    if pil_image is not None:
        axes[0].imshow(pil_image)
        axes[0].set_title("Input image", fontsize=11)
        axes[0].axis("off")

    im = axes[-1].imshow(depth_map, cmap="plasma", vmin=depth_map.min(), vmax=depth_map.max())
    axes[-1].set_title("Predicted depth (m)", fontsize=11)
    axes[-1].axis("off")

    cbar = fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    cbar.set_label("metres", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)  # prevent memory leak
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def overlay_food_mask_b64(pil_image: Image.Image, food_mask: np.ndarray, scores: list[float]) -> str:
    """Render input image with SAM food mask overlay and confidence label as base64 PNG"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)

    axes[0].imshow(pil_image)
    axes[0].set_title("Input image", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(pil_image)

    # Green mask overlay at 45% opacity
    rgba_mask = np.zeros((*food_mask.shape, 4), dtype=np.float32)
    rgba_mask[food_mask.astype(bool)] = [0.0, 0.85, 0.3, 0.45]
    axes[1].imshow(rgba_mask)

    # Confidence label derived from mean score of detected instances
    if scores:
        mean_score = float(np.mean(scores))
        confidence_label = (
            "High" if mean_score >= 0.75 else
            "Medium" if mean_score >= 0.50 else
            "Low"
        )
        color = "#22c55e" if mean_score >= 0.75 else "#f59e0b" if mean_score >= 0.50 else "#ef4444"
        n_instances = len(scores)
        label_text = f"Confidence: {confidence_label} ({mean_score:.2f})  |  Instances: {n_instances}"
    else:
        label_text = "Confidence: N/A  |  Instances: 0 (fallback mask)"
        color = "#6b7280"

    patch = mpatches.Patch(facecolor="#00d94f", edgecolor="white", linewidth=1.5, label="Food mask")
    axes[1].legend(handles=[patch], loc="lower left", fontsize=9, framealpha=0.75, facecolor="#1e1e1e", labelcolor="white")
    axes[1].set_title(label_text, fontsize=10, color=color, fontweight="bold")
    axes[1].axis("off")

    plt.suptitle("SAM 3 Food Segmentation", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def write_b64_png(b64: str, path: str) -> None:
    """Drop one of the renders above onto disk, for looking at after a request"""
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))
