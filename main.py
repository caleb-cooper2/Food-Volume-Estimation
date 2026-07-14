import base64
import io
import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from pillow_heif import register_heif_opener

from model_manage import register_loader, get_model

register_heif_opener()

import piexif
import matplotlib
from rich.logging import RichHandler

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, Sam3Processor, Sam3Model

M3_TO_CM3 = 1_000_000.0  # 1 m^3 = 10^6 cm^3
MAX_FOOD_HEIGHT_M = 0.15
RELIEF_MEAN_M = 0.010
RELIEF_STD_M = 0.020
NOMINAL_FOOD_HEIGHT_CM = 2.0 # crude portion-height prior for the low-confidence fallback

# Known tip-to-tip lengths of common cutlery (metres). https://www.steelcitycutlery.com/shapesandsizes.html?srsltid=AfmBOoqN4Sg7zv4iAoLmDFJwXQP1FkXmRXZnqZTYWxcUiBf5r3rzJ14o, https://sabre-paris.com/en/pages/size-guide
# These could vary ~±10% by brand/style, and volume error grows with the CUBE of length error
REFERENCE_LENGTHS_M = {
    "fork": 0.210, # table fork ~20.5-22 cm
    "knife": 0.240, # table knife ~24 cm
    "spoon": 0.215, # tablespoon ~21.5 cm
}


@dataclass
class EstimationResponse:
    """
    volume_cm3 or mass_g gets filled depending on what the approach actually predicts (the geometric and multi-view routes give a volume, the trained model gives a mass), and anything
    approach-specific gets added in diagnostics so the top-level shape stays identical across all three.
    """
    approach: str # one of 'monocular-geometric' | 'deep-learning' | 'multi-view'
    volume_cm3: Optional[float]
    mass_g: Optional[float]
    confidence: Optional[str]
    diagnostics: dict # approach-specific extras (heights, coverage, scale, semantic fusion, debug overlay, ...)

@dataclass
class CameraInfo:
    """Pinhole camera model parameters derived from EXIF or fallback."""
    fx: float  # horizontal focal length in pixels
    fy: float  # vertical focal length in pixels
    cx: float  # principal point x (pixels)
    cy: float  # principal point y (pixels)
    image_width: int
    image_height: int
    source: str  # 'exif' | 'error'

@dataclass
class VolumeResult:
    """Computed volume and intermediate metric values."""
    volume_cm3: float
    plate_depth_m: float
    max_food_height_cm: float
    mean_food_height_cm: float
    clipped_high_pct: float = 0.0
    geometry_confidence: float = 1.0  # 0..1, drops on oblique views / heavy clipping where the height-field integral is unreliable


logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Volume Estimation API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load models at startup
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch_dtype = torch.bfloat16
torch_device = torch.device(device)
processor = AutoImageProcessor.from_pretrained("apple/DepthPro-hf")
sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")

def _load_depth_model():
    m = AutoModelForDepthEstimation.from_pretrained("apple/DepthPro-hf", torch_dtype=torch_dtype)
    return m.eval()

def _load_sam3_model():
    m = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch_dtype)
    return m.eval()

register_loader("depthpro", _load_depth_model)
register_loader("sam3", _load_sam3_model)

class VolumeAssistedRegressor(nn.Module):
    """
    Mirror of the training-time model: ConvNeXt features + a log-volume scalar concatenated before the head.
    Got to stay structurally identical to train.py or the state_dict won't load
    """
    def __init__(self, log_target: bool = True, depth_channel: bool = False):
        super().__init__()
        backbone = models.convnext_tiny(weights=None)
        if depth_channel:
            inflate_stem_to_4ch(backbone)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.norm = backbone.classifier[0]           # LayerNorm2d(768)
        self.log_target = log_target
        # Standardisation stats for the log-volume scalar, filled from the train cache and saved in the state_dict so inference normalises
        self.register_buffer("log_volume_mean", torch.zeros(1))
        self.register_buffer("log_volume_std", torch.ones(1))
        head_layers = [
            nn.Linear(768 + 1, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        ]
        if not log_target:
            head_layers.append(nn.Softplus())
        self.head = nn.Sequential(*head_layers)

    def forward(self, rgb: torch.Tensor, volume_cm3: torch.Tensor) -> torch.Tensor:
        feats = torch.flatten(self.norm(self.avgpool(self.features(rgb))), 1)
        v = torch.log(volume_cm3.clamp(min=1.0)).unsqueeze(1)
        v = (v - self.log_volume_mean) / self.log_volume_std   # whiten so the head sees the variation, not a constant offset
        return self.head(torch.cat([feats, v], dim=1)).squeeze(1)

def load_model(model_path: str) -> nn.Module:
    """
    Rebuild the trained model to match its checkpoint. Reads the saved args so it picks the right architecture: the plain ConvNeXt head,
    or the volume-assisted head (norm + head, with a 769-wide first Linear because the geometric volume scalar is concatenated in)
    """
    checkpoint = torch.load(model_path, map_location="cpu")
    saved_args = checkpoint.get("args", {})
    log_target = bool(saved_args.get("log_target", False))
    use_volume = bool(saved_args.get("use_volume", False))

    depth_channel = bool(saved_args.get("depth_channel", False))

    if use_volume:
        model = VolumeAssistedRegressor(log_target=log_target, depth_channel=depth_channel)
    else:
        head_layers = [
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        ]
        if not log_target:
            head_layers.append(nn.Softplus())
        model = models.convnext_tiny(weights=None)
        if depth_channel:
            inflate_stem_to_4ch(model)
        model.classifier[2] = nn.Sequential(*head_layers)

    model.load_state_dict(checkpoint["state_dict"])
    model._log_target = log_target
    model._use_volume = use_volume
    model._depth_channel = depth_channel
    return model.eval()



def segment_reference_object(pillow_image: Image.Image, utensil: str = "fork", threshold: float = 0.5) -> Optional[np.ndarray]:
    """Segment a reference utensil with SAM 3 and return the single highest-scoring instance mask, or None if nothing confident is found"""
    inputs = sam3_processor(images=pillow_image, text=utensil, return_tensors="pt").to(torch_device)
    sam3 = get_model("sam3")
    with torch.no_grad():
        outputs = sam3(**inputs)

    results = sam3_processor.post_process_instance_segmentation(
        outputs, threshold=threshold, mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    masks, scores = results["masks"], results["scores"]
    if masks.shape[0] == 0:
        logger.warning(f"  Reference | no '{utensil}' found -> cannot anchor scale from a reference object")
        return None

    best = int(torch.argmax(scores).item())
    logger.info(f"  Reference | '{utensil}' found: score={scores[best]:.3f}  ({masks.shape[0]} candidate(s))")
    return masks[best].cpu().numpy().astype(np.uint8)


def measure_mask_endpoints(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the two tip pixels of an elongated mask via PCA: project every mask pixel onto its principal
    axis and take the extremes. Returns two (row, col) pixel coordinates.
    """
    ys, xs = np.where(mask > 0)
    coords = np.stack([xs, ys], axis=1).astype(np.float64)  # (N, 2) in (x, y)
    centred = coords - coords.mean(axis=0)
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    projections = centred @ vh[0]  # vh[0] = principal (long) axis
    p_min = coords[np.argmin(projections)]
    p_max = coords[np.argmax(projections)]
    return p_min[::-1], p_max[::-1]  # (x, y) -> (row, col) for depth-map indexing


def backproject_pixel(row: float, col: float, depth_map: np.ndarray, cam: CameraInfo, window: int = 5
                      ) -> np.ndarray:
    """Back-project one pixel to a 3D camera-space point (metres), using the median depth in a small
    window for robustness against per-pixel depth noise right at the tip."""
    r, c = int(round(row)), int(round(col))
    r0, r1 = max(r - window, 0), min(r + window + 1, depth_map.shape[0])
    c0, c1 = max(c - window, 0), min(c + window + 1, depth_map.shape[1])
    patch = depth_map[r0:r1, c0:c1]
    z = float(np.median(patch[patch > 0])) if np.any(patch > 0) else float(depth_map[r, c])
    x = (c - cam.cx) * z / cam.fx
    y = (r - cam.cy) * z / cam.fy
    return np.array([x, y, z], dtype=np.float64)


def reference_scale_factor(pillow_image: Image.Image, depth_map: np.ndarray, cam: CameraInfo,
                           utensil: str = "fork") -> Optional[float]:
    """
    Recover the absolute-scale correction from a utensil of known real length: measure its 3D length
    in the depth model's (up-to-scale) units and divide the known length by it. Returns the factor to
    multiply the depth map by, or None if no reliable reference was found (caller then falls back to
    the reference-free path).
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
    if not (0.33 <= correction <= 3.0):  # a >3x correction is almost always a bad mask/depth, not real scale
        logger.warning(f"  Reference | correction x{correction:.3f} implausible -> rejecting, using reference-free scale")
        return None
    return correction


def depth_to_relief_channel(depth_m: np.ndarray) -> np.ndarray:
    """Metric depth -> table-relative relief (m above support). Mirrors train.py... the background median cancels the RealSense<->DepthPro offset."""
    valid = depth_m[depth_m > 0]
    plate_ref_m = float(np.median(valid)) if valid.size else 0.0
    return np.clip(plate_ref_m - depth_m, 0.0, MAX_FOOD_HEIGHT_M).astype(np.float32)


def inflate_stem_to_4ch(model: nn.Module) -> None:
    """Swap ConvNeXt's stem to 4 channels so a --depth_channel checkpoint loads"""
    old_stem = model.features[0][0]
    new_stem = nn.Conv2d(4, old_stem.out_channels, kernel_size=old_stem.kernel_size, stride=old_stem.stride)
    with torch.no_grad():
        new_stem.weight[:, :3] = old_stem.weight
        new_stem.weight[:, 3:4] = old_stem.weight.mean(dim=1, keepdim=True)
        new_stem.bias.copy_(old_stem.bias)
    model.features[0][0] = new_stem


def area_based_volume_proxy(depth_map: np.ndarray, food_mask: np.ndarray, cam: CameraInfo) -> float:
    """
    Portion-sensitive volume proxy for when the plane-fit geometry is untrusted: metric food footprint (sum of per-pixel areas) times
    a nominal food height
    """
    food_px = food_mask > 0
    if not np.any(food_px):
        return 0.0
    ys, xs = np.where(food_px)
    z = depth_map[ys, xs].astype(np.float64)
    pixel_area_m2 = (z / cam.fx) * (z / cam.fy)
    area_cm2 = float(np.sum(pixel_area_m2)) * 1e4  # m^2 -> cm^2
    return area_cm2 * NOMINAL_FOOD_HEIGHT_CM


def extract_image_info(image_bytes: bytes, actual_width: int, actual_height: int) -> CameraInfo:
    """
    Extract camera intrinsics from EXIF focal length or use smartphone default.
    Converts 35mm-equivalent focal length to pixel-space focal length.
    """
    focal_35mm = 0.0
    source = "exif"

    try:
        exif_dict = piexif.load(image_bytes)
        exif_ifd = exif_dict.get("Exif", {})

        # FocalLengthIn35mmFilm -> stored as unsigned short
        if piexif.ExifIFD.FocalLengthIn35mmFilm in exif_ifd:
            focal_35mm = float(exif_ifd[piexif.ExifIFD.FocalLengthIn35mmFilm])
    except Exception:
        pass

    # Fallback to typical smartphone wide-angle lens
    if focal_35mm <= 0.0:
        source = "fallback"
        focal_35mm = 26.0
        logger.warning("Invalid/missing EXIF focal length. Using smartphone fallback (26mm).")

    # Compute horizontal field of view
    full_frame_width_mm = 36.0
    half_fov_h = math.atan(full_frame_width_mm / (2.0 * focal_35mm))

    # Pixel focal lengths relative to image resolution
    fx = actual_width / (2.0 * math.tan(half_fov_h))
    fy = fx  # assume square sensor pixels

    cx = actual_width / 2.0
    cy = actual_height / 2.0

    return CameraInfo(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        image_width=actual_width,
        image_height=actual_height,
        source=source,
    )


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
        target_sizes=[(original_h, original_w)],
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


def segment_food(pillow_image: Image.Image, threshold: float = 0.5) -> tuple[np.ndarray, list[float], list[np.ndarray]]:
    """
    Segment food region using SAM 3 with text prompt "food".
    Falls back to full-image mask if no instances found.
    :return: union mask of all detected instances and per-instance scores.
    """
    img_w, img_h = pillow_image.size
    total_pixels = img_w * img_h

    inputs = sam3_processor(images=pillow_image, text="food", return_tensors="pt").to(torch_device)

    sam3 = get_model("sam3")
    with torch.no_grad():
        outputs = sam3(**inputs)

    results = sam3_processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]

    instance_masks = results["masks"]
    instance_scores = results["scores"]

    if instance_masks.shape[0] == 0:
        logger.warning("  SAM 3 | No food instances found -> using full-image fallback mask")
        fallback = np.ones((img_h, img_w), dtype=np.uint8)
        return fallback, [], [fallback]

    instance_scores_list = instance_scores.tolist()
    instance_masks_list = [m.cpu().numpy().astype(np.uint8) for m in instance_masks]

    # Log per-instance breakdown before union
    for i, s in enumerate(instance_scores_list):
        inst_px = int(instance_masks[i].sum().item())
        inst_pct = inst_px / total_pixels * 100
        logger.info(f"  SAM 3 | Instance {i}: score={s:.3f}  pixels={inst_px}  coverage={inst_pct:.1f}%")

    union_mask = instance_masks.any(dim=0).cpu().numpy().astype(np.uint8)
    union_px = int(union_mask.sum())
    union_pct = union_px / total_pixels * 100
    bg_pct = 100.0 - union_pct

    logger.info(
        f"  SAM 3 | Union mask: {union_px} px food ({union_pct:.1f}%)  |  "
        f"{total_pixels - union_px} px background ({bg_pct:.1f}%)"
    )

    if union_pct > 80.0:
        logger.warning(f"  SAM 3 | Food mask covers {union_pct:.1f}% of image -> background region too small for reliable plane fitting")
    if union_pct < 2.0:
        logger.warning(f"  SAM 3 | Food mask covers only {union_pct:.1f}% of image -> possible segmentation failure")

    return union_mask, instance_scores_list, instance_masks_list


def estimate_plate_depth(depth_map, food_mask) -> float:
    bg_mask = food_mask == 0
    if np.any(bg_mask):
        bg_depths = depth_map[bg_mask]
        plate_depth = float(np.median(bg_depths))
        logger.info(
            f"  PlateDepth | background pixels={int(bg_mask.sum())}  "
            f"median={plate_depth:.3f}m  "
            f"std={bg_depths.std():.4f}m"
        )
    else:
        plate_depth = float(np.median(depth_map))
        logger.warning(f"  PlateDepth | No background pixels... using full-image median ({plate_depth:.3f}m)")
    return plate_depth


def fit_support_plane(
        depth_map: np.ndarray,
        food_mask: np.ndarray,
        image_info: CameraInfo,
        inlier_thresh_m: float = 0.006,
        ransac_iters: int = 250,
        max_points: int = 8000
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Fits the supporting surface as a 3D plane, referenced to a local ring of background around the
    food rather than the whole scene.

    Sampling every non-food pixel lets RANSAC lock onto the far table/floor receding away in a casual photo -> deep, tilted, only ~half the points as inliers,
    and every food pixel floats several cm above it. A ring hugging the food references height to the surface the food actually sits on.

    Ring = (food dilated by ring_px) minus food, intersected with valid background. Falls back to full background if the ring comes out too thin (food fills the frame / runs off edges)
    :return: (unit normal pointing back toward the camera, a point on the plane, inlier_ratio)
    """
    fx, fy, cx, cy = image_info.fx, image_info.fy, image_info.cx, image_info.cy

    valid = (depth_map > 0.1) & (depth_map < 5.0)
    food = food_mask > 0

    # Ring width scales with the food's own size (a band ~10% of its extent), clamped for cost
    ring_px = int(np.clip(0.10 * np.sqrt(max(int(food.sum()), 1)), 25, 150))
    ring = cv2.dilate(food.astype(np.uint8), np.ones((ring_px, ring_px), np.uint8)).astype(bool)
    ring = ring & ~food & valid

    ys, xs = np.where(ring)
    if len(xs) < 200:
        ys, xs = np.where((~food) & valid)
        logger.warning(f"  Plane | ring too thin ({len(xs)} px), falling back to full background")
    else:
        logger.info(f"  Plane | using local ring of {len(xs)} px (ring_px={ring_px})")

    if len(xs) < 100:
        z = float(np.median(depth_map[~food])) if np.any(~food) else float(np.median(depth_map))
        logger.warning(f"  Plane | <100 usable bg px, falling back to flat plane at {z:.3f}m")
        return np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, z]), 0.0

    # Back-project the ring pixels to 3D (metres)
    z = depth_map[ys, xs].astype(np.float64)
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)

    if len(pts) > max_points:
        pts = pts[np.random.default_rng(0).choice(len(pts), max_points, replace=False)]

    rng = np.random.default_rng(0)
    best_inliers = None
    for _ in range(ransac_iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        dist = np.abs((pts - s[0]) @ n)
        inliers = dist < inlier_thresh_m
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    # Refit on the inlier set via SVD -> least-squares plane, steadier than the 3-point fit
    inlier_pts = pts[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_pts - centroid)
    n = vh[-1] / np.linalg.norm(vh[-1])

    # Orient the normal back toward the camera (origin) so food ABOVE the plate reads positive
    if n @ centroid > 0:
        n = -n

    inlier_ratio = float(best_inliers.mean())
    logger.info(
        f"  Plane | fitted on {len(inlier_pts)}/{len(pts)} ring pts (inliers={inlier_ratio:.2f})  "
        f"normal=[{n[0]:.2f}, {n[1]:.2f}, {n[2]:.2f}]  "
        f"tilt={np.degrees(np.arccos(min(abs(n[2]), 1.0))):.1f}deg off camera axis"
    )
    return n, centroid, inlier_ratio


def compute_volume(
        depth_map: np.ndarray,
        food_mask: np.ndarray,
        image_info: CameraInfo,
        plane_n: np.ndarray,
        plane_p0: np.ndarray,
) -> VolumeResult:
    """
    Integrate volume as a sum of per-pixel prisms, but measure height as the perpendicular distance from each food point to the fitted support plane
    (h = n . (P - p0)) rather than plate_depth - Z. Same prism idea as before, just referenced to a real tilted plane so the
    perspective slope is gone before integration
    """
    food_px = food_mask > 0
    if not np.any(food_px):
        logger.warning("  Volume | No food pixels :( returning zero volume")
        return VolumeResult(0.0, float(plane_p0[2]), 0.0, 0.0, 0.0, 0.0)

    ys, xs = np.where(food_px)
    fx, fy, cx, cy = image_info.fx, image_info.fy, image_info.cx, image_info.cy

    z = depth_map[ys, xs].astype(np.float64)
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)

    # Perpendicular height above the plane, clipped as before
    heights_m = (pts - plane_p0) @ plane_n
    n_below = int(np.sum(heights_m < 0))
    heights_m = np.clip(heights_m, 0.0, MAX_FOOD_HEIGHT_M)
    clipped_high = int(np.sum(heights_m >= MAX_FOOD_HEIGHT_M - 1e-9))
    clipped_high_pct = 100.0 * clipped_high / len(heights_m)  # relief pinned at the cap -> likely a scale failure

    # Per-pixel frontal footprint from the pinhole model. The cos term corrects foreshortening:
    # a ray hitting the surface at angle theta to the plane normal covers 1/cos(theta) more ground.
    pixel_area_m2 = (z / fx) * (z / fy)
    rays = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    cos_theta = np.abs(rays @ plane_n)
    cos_theta = np.clip(cos_theta, 0.7, 1.0) # floor avoids blow-up at grazing angles. Floor well above zero
    pixel_area_m2 = pixel_area_m2 / cos_theta

    total_volume_cm3 = float(np.sum(heights_m * pixel_area_m2)) * M3_TO_CM3

    # trust drops if
    # - the support plane tilts off the camera axis
    # - rays graze the surface -> the sidewalls where over-counting lives
    # - height pins at the cap (usually a depth-scale failure)
    view_cos = float(abs(plane_n[2])) # 1.0 overhead, cos(tilt) otherwise
    tilt_conf = float(np.clip((view_cos - 0.7) / 0.3, 0.0, 1.0)) # ~1 overhead, 0 by ~45deg tilt
    graze_frac = float(np.mean(cos_theta <= 0.7 + 1e-6))
    graze_conf = float(np.clip(1.0 - graze_frac / 0.3, 0.0, 1.0)) # 30%+ grazing food px -> untrusted
    clip_conf = float(np.clip(1.0 - clipped_high_pct / 10.0, 0.0, 1.0))
    geometry_confidence = round(tilt_conf * graze_conf * clip_conf, 3)

    nonzero = heights_m[heights_m > 0]
    if len(nonzero) == 0:
        logger.warning("  Volume | ALL heights <= 0 after clipping -> check plane orientation / segmentation")
    else:
        logger.info(
            f"  Volume | Height above plane (cm): min={nonzero.min()*100:.2f} "
            f"mean={nonzero.mean()*100:.2f} max={nonzero.max()*100:.2f} "
            f"| {n_below} px below plane ({n_below/len(heights_m)*100:.1f}%)"
        )
    logger.info(
        f"  Volume | {total_volume_cm3:.2f} cm^3  | food_px={len(z)}  fx={fx:.0f}  fy={fy:.0f}  "
        f"confidence={geometry_confidence:.2f} (tilt={tilt_conf:.2f} graze={graze_conf:.2f} clip={clip_conf:.2f})"
    )

    return VolumeResult(
        total_volume_cm3, float(plane_p0[2]),
        float(heights_m.max()) * 100.0, float(heights_m.mean()) * 100.0,
        clipped_high_pct, geometry_confidence
        )


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
    import matplotlib.patches as mpatches

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


# Approach A - Monocular geometric
# Single RGB image -> metric depth + food mask -> support plane -> height-field integral. Only geometry, no learned volume regression
@app.post("/api/v1/estimate-volume", response_model=EstimationResponse)
async def volume_estimation(file: UploadFile = File(...)) -> EstimationResponse:
    """
    End-to-end volume estimation pipeline:
    1. Extract camera intrinsics from EXIF
    2. Estimate metric depth map (DepthPro)
    3. Segment food region (SAM 3)
    4. Integrate volume from height field
    """
    t_start = time.perf_counter()

    if file.content_type not in ("image/jpeg", "image/jpg", "image/png", "image/heic"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Send JPEG or PNG only",
        )

    image_bytes = await file.read()

    if len(image_bytes) < 1024:
        raise HTTPException(status_code=400, detail="Image too small -> minimum 1 KB")
    if len(image_bytes) > 30 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large -> maximum 30 MB")

    # Decode PIL image ONCE early so we have true pixel geometry
    try:
        pillow_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}")

    actual_w, actual_h = pillow_image.size

    t_image_info_start = time.perf_counter()
    image_info = extract_image_info(image_bytes, actual_w, actual_h)
    logger.info(
        f"[A] Intrinsics ({image_info.source}): "
        f"fx={image_info.fx:.0f}, fy={image_info.fy:.0f}, "
        f"cx={image_info.cx:.0f}, cy={image_info.cy:.0f} "
        f"({time.perf_counter() - t_image_info_start:.3f}s)"
    )

    t_depth_start = time.perf_counter()
    depth_map, focal_length_px = estimate_depth(pillow_image)
    logger.info(
        f"[B] Depth inference done. "
        f"Range: [{depth_map.min():.3f}, {depth_map.max():.3f}] m "
        f"({time.perf_counter()-t_depth_start:.3f}s)"
    )

    t_reference_start = time.perf_counter()
    reference_correction = reference_scale_factor(pillow_image, depth_map, image_info, utensil="fork")
    if reference_correction is not None:
        depth_map = depth_map * reference_correction
        logger.info(f"[B*] Reference-anchored depth x{reference_correction:.3f} "
                    f"({time.perf_counter() - t_reference_start:.3f}s)")

    image_info = CameraInfo(
        fx=focal_length_px,
        fy=focal_length_px, # DepthPro predicts horizontal FOV; assume square pixels
        cx=actual_w / 2.0,
        cy=actual_h / 2.0,
        image_width=actual_w,
        image_height=actual_h,
        source="depthpro_fov",
    )

    t_segmentation_start = time.perf_counter()
    food_mask, mask_scores, _ = segment_food(pillow_image)  # (H, W) uint8 0/1
    food_pixel_count = int(food_mask.sum())
    food_coverage_pct = food_pixel_count / food_mask.size * 100
    logger.info(f"[C] Segmentation done. {food_pixel_count} food px ({food_coverage_pct:.1f}%) ({time.perf_counter() - t_segmentation_start:.3f}s)")

    t_plate_depth_start = time.perf_counter()
    plate_depth_m = estimate_plate_depth(depth_map, food_mask)
    logger.info(f"[D] Plate depth: {plate_depth_m:.3f}m ({time.perf_counter() - t_plate_depth_start:.3f}s) (deprecated now with plane fitting)")

    t_volume_start = time.perf_counter()
    plane_n, plane_p0, _ = fit_support_plane(depth_map, food_mask, image_info)
    volume_res = compute_volume(depth_map, food_mask, image_info, plane_n, plane_p0)
    logger.info(f"[E] Volume Estimation Result: {volume_res.volume_cm3:.2f} cm^3. ({time.perf_counter() - t_volume_start:.3f}s)")

    confidence = (
        "high" if volume_res.geometry_confidence >= 0.66 else
        "medium" if volume_res.geometry_confidence >= 0.33 else
        "low"
    )
    if confidence != "high":
        logger.warning(
            f"[E] Geometry confidence {volume_res.geometry_confidence:.2f} ({confidence}) -> "
            f"view likely too oblique for a reliable single-view volume"
        )

    # Render debug visualisations
    depth_b64 = depth_to_b64_png(depth_map, pil_image=pillow_image)
    depth_bytes = base64.b64decode(depth_b64)
    with open("/tmp/depth_debug.png", "wb") as f:
        f.write(depth_bytes)

    seg_b64 = overlay_food_mask_b64(pillow_image, food_mask, mask_scores)
    seg_bytes = base64.b64decode(seg_b64)
    with open("/tmp/seg_overlay.png", "wb") as f:
        f.write(seg_bytes)

    t_total = time.perf_counter() - t_start
    logger.info(f"Total pipeline time: {t_total:.3f}s")

    return EstimationResponse(
        approach="monocular-geometric",
        volume_cm3=volume_res.volume_cm3,
        mass_g=None,
        confidence=confidence,
        diagnostics={
            "food_pixel_count": food_pixel_count,
            "food_coverage_pct": food_coverage_pct,
            "max_food_height_cm": volume_res.max_food_height_cm,
            "mean_food_height_cm": volume_res.mean_food_height_cm,
            "plate_depth_m": volume_res.plate_depth_m,
            "intrinsics_source": image_info.source,
            "debug_overlay_b64": seg_b64
        },
    )


# Approach B - Deep learning
# Single RGB image -> trained ConvNeXt regressor -> mass (g). When the checkpoint was trained with --use_volume, the geometric approach above is reused to compute the
# volume scalar the head expects, mirroring how cache_volume_scalars.py builds it
class VolumeAssistedRegressor(nn.Module):
    """
    Mirror of the training-time model: ConvNeXt features + a log-volume scalar concatenated before the head.
    Got to stay structurally identical to train.py or the state_dict won't load
    """
    def __init__(self, log_target: bool = True):
        super().__init__()
        backbone = models.convnext_tiny(weights=None)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.norm = backbone.classifier[0]           # LayerNorm2d(768)
        self.log_target = log_target
        # Standardisation stats for the log-volume scalar, filled from the train cache and saved in the state_dict so inference normalises
        self.register_buffer("log_volume_mean", torch.zeros(1))
        self.register_buffer("log_volume_std", torch.ones(1))
        head_layers = [
            nn.Linear(768 + 1, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        ]
        if not log_target:
            head_layers.append(nn.Softplus())
        self.head = nn.Sequential(*head_layers)

    def forward(self, rgb: torch.Tensor, volume_cm3: torch.Tensor) -> torch.Tensor:
        feats = torch.flatten(self.norm(self.avgpool(self.features(rgb))), 1)
        v = torch.log(volume_cm3.clamp(min=1.0)).unsqueeze(1)
        v = (v - self.log_volume_mean) / self.log_volume_std   # whiten so the head sees the variation, not a constant offset
        return self.head(torch.cat([feats, v], dim=1)).squeeze(1)

def load_model(model_path: str) -> nn.Module:
    """
    Rebuild the trained model to match its checkpoint. Reads the saved args so it picks the right architecture: the plain ConvNeXt head,
    or the volume-assisted head (norm + head, with a 769-wide first Linear because the geometric volume scalar is concatenated in)
    """
    checkpoint = torch.load(model_path, map_location="cpu")
    saved_args = checkpoint.get("args", {})
    log_target = bool(saved_args.get("log_target", False))
    use_volume = bool(saved_args.get("use_volume", False))

    if use_volume:
        model = VolumeAssistedRegressor(log_target=log_target)
    else:
        head_layers = [
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        ]
        if not log_target:
            head_layers.append(nn.Softplus())
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Sequential(*head_layers)

    model.load_state_dict(checkpoint["state_dict"])
    model._log_target = log_target
    model._use_volume = use_volume  # stash so the endpoint knows whether to feed the scalar
    return model.eval()

register_loader("custom_volume", lambda: load_model("checkpoints/best_model.pt"))

_rgb_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


@torch.no_grad()
def run_custom_model(image: Image.Image) -> float:
    """
    Estimate mass (g) with the custom trained model. Builds whatever extra inputs the checkpoint was trained with...
    a metric relief channel (--depth_channel) and/or a geometric volume scalar (--use_volume)
    """
    tensor = _rgb_transform(image).unsqueeze(0)  # (1, 3, 224, 224)

    model = get_model("custom_volume")
    needs_depth = getattr(model, "_use_volume", False) or getattr(model, "_depth_channel", False)

    depth_map = focal_px = mask = None
    if needs_depth:
        depth_map, focal_px = estimate_depth(image)
        mask, _, _ = segment_food(image)

    if getattr(model, "_depth_channel", False):
        relief = depth_to_relief_channel(depth_map)
        relief_224 = cv2.resize(relief, (224, 224), interpolation=cv2.INTER_NEAREST)
        relief_tensor = (torch.from_numpy(relief_224)[None, None] - RELIEF_MEAN_M) / RELIEF_STD_M
        tensor = torch.cat([tensor, relief_tensor], dim=1)  # (1, 4, 224, 224): RGB + relief

    volume_t = None
    if getattr(model, "_use_volume", False):
        w, h = image.size
        cam = CameraInfo(fx=focal_px, fy=focal_px, cx=w / 2.0, cy=h / 2.0, image_width=w, image_height=h, source="depthpro_fov")
        plane_n, plane_p0, _ = fit_support_plane(depth_map, mask, cam)
        vr = compute_volume(depth_map, mask, cam, plane_n, plane_p0)
        if vr.geometry_confidence < 0.33:
            vol = area_based_volume_proxy(depth_map, mask, cam)
            logger.warning(f"  Custom | geometry confidence {vr.geometry_confidence:.2f} -> area proxy volume {vol:.1f} cm^3")
        else:
            vol = vr.volume_cm3
        volume_t = torch.tensor([vol], dtype=torch.float32)

    # Fetch the model LAST so it lands on the GPU and nothing evicts it before we run
    model = get_model("custom_volume")
    model_device = next(model.parameters()).device
    tensor = tensor.to(model_device)

    if volume_t is not None:
        out = model(tensor, volume_t.to(model_device)).squeeze().item()
    else:
        out = model(tensor).squeeze().item()

    if getattr(model, "_log_target", False):
        out = float(np.exp(out))
    return out


@app.post("/api/v1/estimate-volume-dl", response_model=EstimationResponse)
async def volume_estimation_dl(file: UploadFile = File(...)) -> EstimationResponse:
    """
    End-to-end volume estimation pipeline using deep learning model.
    """
    t_start = time.perf_counter()

    if file.content_type not in ("image/jpeg", "image/jpg", "image/png", "image/heic"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Send JPEG or PNG only",
        )

    image_bytes = await file.read()

    if len(image_bytes) < 1024:
        raise HTTPException(status_code=400, detail="Image too small -> minimum 1 KB")
    if len(image_bytes) > 30 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large -> maximum 30 MB")

    # Decode PIL image ONCE early so we have true pixel geometry
    try:
        pillow_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}")

    with torch.no_grad():
        mass = run_custom_model(pillow_image)

    logger.info(f"Model estimated mass: {mass:.2f} g")
    logger.info(f"Total pipeline time: {time.perf_counter() - t_start:.3f}s")
    return EstimationResponse(
        approach="deep-learning",
        volume_cm3=None,
        mass_g=round(mass, 2),
        confidence=None,
        diagnostics={}
    )
