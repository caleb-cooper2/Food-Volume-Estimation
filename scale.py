"""
Metric-scale anchors for the monocular route. DepthPro's depth is only up-to-scale in practice, so each anchor measures
something of known real size in the raw depth units and returns the factor that makes the map metric
"""

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from checkerboard import find_corners, adjacent_corner_pixel_pairs, CHECKERBOARD_SQUARE_M, CHECKERBOARD_SQUARE_CM
from geometry import backproject_pixel, measure_mask_endpoints, metric_footprint_diameter_cm
from logging_config import get_logger
from model_manage import register_loader, get_model, torch_device
from scale_prior import SizePriorHead, extract_clip_features, CLIP_MODEL_NAME
from schemas import CameraInfo
from segmentation import segment_reference_object

logger = get_logger(__name__)

SIZE_PRIOR_CHECKPOINT = "checkpoints/size_prior.pt"

# Known tip-to-tip lengths of common cutlery (metres). https://www.steelcitycutlery.com/shapesandsizes.html?srsltid=AfmBOoqN4Sg7zv4iAoLmDFJwXQP1FkXmRXZnqZTYWxcUiBf5r3rzJ14o, https://sabre-paris.com/en/pages/size-guide
# These could vary ~±10% by brand/style, and volume error grows with the CUBE of length error
REFERENCE_LENGTHS_M = {
    "fork": 0.210, # table fork ~20.5-22 cm
    "knife": 0.240, # table knife ~24 cm
    "spoon": 0.215, # tablespoon ~21.5 cm
}

# A >3x correction is almost always a bad mask/detection/depth rather than real scale, so every anchor rejects outside this
PLAUSIBLE_CORRECTION = (0.15, 3.0)

clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)


def _load_clip_model():
    m = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    return m.eval()


def load_size_prior_head(model_path: str) -> nn.Module:
    """Rebuild the trained CLIP size-prior head from its checkpoint"""
    checkpoint = torch.load(model_path, map_location="cpu")
    head = SizePriorHead(feature_dim=checkpoint.get("feature_dim", 512))
    head.load_state_dict(checkpoint["state_dict"])
    return head.eval()


register_loader("clip", _load_clip_model)
register_loader("size_prior", lambda: load_size_prior_head(SIZE_PRIOR_CHECKPOINT))


def is_plausible(correction: float) -> bool:
    low, high = PLAUSIBLE_CORRECTION
    return low <= correction <= high


def reference_scale_factor(pillow_image: Image.Image, depth_map: np.ndarray, cam: CameraInfo, utensil: str = "fork") -> Optional[float]:
    """
    Recover the absolute-scale correction from a utensil of known real length: measure its 3D length in the depth model's (up-to-scale) units and
    divide the known length by it. Returns the factor to multiply the depth map by, or None if no reliable reference was found (caller then falls back to
    the reference-free path)
    """
    known_m = REFERENCE_LENGTHS_M.get(utensil)
    if known_m is None:
        logger.warning(f"  Reference | no known length for '{utensil}'")
        return None

    mask = segment_reference_object(pillow_image, utensil)
    if mask is None or int(mask.sum()) < 200:
        return None

    p_a, p_b = measure_mask_endpoints(mask)
    measured_m = float(np.linalg.norm(
        backproject_pixel(*p_a, depth_map, cam) - backproject_pixel(*p_b, depth_map, cam)
    ))
    if measured_m <= 1e-4:
        logger.warning("  Reference | degenerate measured length -> skipping reference scaling")
        return None

    correction = known_m / measured_m
    logger.info(
        f"  Reference | '{utensil}' measured {measured_m*100:.1f}cm in raw depth units "
        f"vs known {known_m*100:.1f}cm -> depth scale x{correction:.3f}"
    )
    if not is_plausible(correction):
        logger.warning(f"  Reference | correction x{correction:.3f} implausible -> rejecting, using reference-free scale")
        return None
    return correction


def checkerboard_scale_factor(pillow_image: Image.Image, depth_map: np.ndarray, cam: CameraInfo) -> Optional[float]:
    """
    Absolute-scale correction from the SimpleFood45 checkerboard, measure adjacent inner-corner spacing in the depth model's raw units and divide the known 1.2 cm by it.
    Exact known geometry replaces the utensil's +-10% length guess. Returns the factor to multiply the depth map by, or None
    """
    found = find_corners(cv2.cvtColor(np.array(pillow_image), cv2.COLOR_RGB2BGR))
    if found is None:
        logger.warning("  Reference | no checkerboard found -> cannot anchor scale from the board")
        return None
    corners, cols, rows = found

    spacings = []
    for (row_a, col_a), (row_b, col_b) in adjacent_corner_pixel_pairs(corners, cols, rows):
        point_a = backproject_pixel(row_a, col_a, depth_map, cam)
        point_b = backproject_pixel(row_b, col_b, depth_map, cam)
        distance = float(np.linalg.norm(point_a - point_b))
        if distance > 1e-5:
            spacings.append(distance)
    if not spacings:
        return None

    measured_m = float(np.median(spacings))
    correction = CHECKERBOARD_SQUARE_M / measured_m
    logger.info(f"  Reference | checkerboard square measured {measured_m*100:.2f}cm in raw depth units vs known {CHECKERBOARD_SQUARE_CM}cm -> depth scale x{correction:.3f}")
    if not is_plausible(correction):
        logger.warning(f"  Reference | correction x{correction:.3f} implausible -> rejecting")
        return None
    return correction


def predict_scale_from_size_prior(pillow_image: Image.Image, depth_map: np.ndarray, food_mask: np.ndarray, cam: CameraInfo) -> Optional[float]:
    """
    Reference-free metric-scale anchor using a frozen-CLIP size-prior head.
    Predicts the food's real-world footprint diameter from appearance, dividing that by the diameter measured in the
    (up-to-scale) monocular depth gives the factor that makes the depth metric
    """
    measured_cm = metric_footprint_diameter_cm(depth_map, food_mask, cam)
    if measured_cm <= 1e-3:
        return None

    head = get_model("size_prior")
    clip_model = get_model("clip", next_name="sam3")
    features = extract_clip_features(pillow_image, clip_model, clip_processor, torch_device)

    head_device = next(head.parameters()).device
    features = features.to(head_device)
    predicted_log_cm = head(features)
    predicted_cm = float(torch.exp(predicted_log_cm).item())

    scale = predicted_cm / measured_cm
    logger.info(f"  SizePrior | predicted footprint {predicted_cm:.1f}cm vs measured {measured_cm:.1f}cm -> depth scale x{scale:.3f}")
    if not is_plausible(scale):
        logger.warning(f"  SizePrior | scale x{scale:.3f} implausible -> rejecting, keeping raw depth")
        return None
    return scale


def resolve_scale_correction(
        pillow_image: Image.Image,
        depth_map: np.ndarray,
        food_mask: np.ndarray,
        cam: CameraInfo,
        scale_ref: str
) -> tuple[Optional[float], str]:
    """
    Pick the scale anchor the request asked for. 'auto' tries the utensils first and falls back to the size prior.
    :return: (correction to multiply the depth map by, name of the anchor that produced it)
    """
    correction, source = None, "none"

    if scale_ref in ("auto", "utensil"):
        # Each utensil is its own SAM 3 pass; averaging the ones that fire smooths per-utensil length error
        utensil_corrections = [
            c for c in (
                reference_scale_factor(pillow_image, depth_map, cam, utensil=utensil)
                for utensil in REFERENCE_LENGTHS_M
            ) if c is not None
        ]
        if utensil_corrections:
            correction, source = float(np.mean(utensil_corrections)), "utensil"

    if scale_ref == "checkerboard":
        correction = checkerboard_scale_factor(pillow_image, depth_map, cam)
        source = "checkerboard" if correction is not None else "none"
    elif scale_ref == "size_prior" or (scale_ref == "auto" and correction is None):
        correction = predict_scale_from_size_prior(pillow_image, depth_map, food_mask, cam)
        source = "size_prior" if correction is not None else "none"

    return correction, source
