"""Metric depth (DepthPro) and the table-relative relief map derived from it"""

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from geometry import MAX_FOOD_HEIGHT_M
from logging_config import get_logger
from model_manage import register_loader, get_model, torch_device

logger = get_logger(__name__)

RELIEF_MEAN_M = 0.010
RELIEF_STD_M = 0.020

processor = AutoImageProcessor.from_pretrained("apple/DepthPro-hf")


def _load_depth_model():
    m = AutoModelForDepthEstimation.from_pretrained("apple/DepthPro-hf", torch_dtype=torch.bfloat16)
    return m.eval()


register_loader("depthpro", _load_depth_model)


def estimate_depth(pillow_image: Image.Image) -> tuple[np.ndarray, float]:
    """
    Predict metric depth map using DepthPro, upsampled to original image resolution.
    :return: float32 array clipped to [0.1, 5.0] meters and the estimated focal length in pixels.
    """
    original_w, original_h = pillow_image.size

    inputs = processor(images=pillow_image, return_tensors="pt")  # preprocesses the image to ensure conformity to the model
    inputs = {k: v.to(torch_device) for k, v in inputs.items()}

    depth_model = get_model("depthpro", next_name="sam3")
    with torch.no_grad():
        outputs = depth_model(**inputs)

    post_processed = processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(original_h, original_w)]
    )

    depth_map = post_processed[0]["predicted_depth"].float().cpu().numpy().astype(np.float32)
    focal_length_px = float(post_processed[0]["focal_length"])  # model-estimated fx

    logger.info(
        f"  DepthPro | focal_length_px={focal_length_px:.1f}  "
        f"depth pre-clip: min={depth_map.min():.3f}m  "
        f"p50={np.percentile(depth_map, 50):.3f}m  "
        f"max={depth_map.max():.3f}m"
    )

    clipped_low = int(np.sum(depth_map < 0.1))
    clipped_high = int(np.sum(depth_map > 5.0))
    if clipped_low or clipped_high:
        logger.warning(
            f"  DepthPro | Clipping {clipped_low} px below 0.1m, "
            f"{clipped_high} px above 5.0m"
        )

    depth_map = np.clip(depth_map, 0.1, 5.0)
    return depth_map, focal_length_px


def depth_to_relief_channel(depth_m: np.ndarray) -> np.ndarray:
    """Metric depth -> table-relative relief (m above support). Mirrors train.py... the background median cancels the RealSense<->DepthPro offset."""
    valid = depth_m[depth_m > 0]
    plate_ref_m = float(np.median(valid)) if valid.size else 0.0
    return np.clip(plate_ref_m - depth_m, 0.0, MAX_FOOD_HEIGHT_M).astype(np.float32)
