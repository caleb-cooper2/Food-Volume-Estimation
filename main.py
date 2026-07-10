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
MAX_LONG_EDGE = 1280 # px
MAX_FOOD_HEIGHT_M = 0.15

@dataclass
class VolumeEstimateResponse:
    volume_cm3: float
    confidence: str
    food_pixel_count: int
    food_coverage_pct: float
    max_food_height_cm: float
    mean_food_height_cm: float
    plate_depth_m: float
    intrinsics_source: str
    debug_overlay_b64: Optional[str]

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
    def __init__(self, log_target: bool = True):
        super().__init__()
        backbone = models.convnext_tiny(weights=None)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.norm = backbone.classifier[0]           # LayerNorm2d(768)
        self.log_target = log_target
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


def fit_support_plane(
        depth_map: np.ndarray,
        food_mask: np.ndarray,
        image_info: CameraInfo,
        inlier_thresh_m: float = 0.006,
        ransac_iters: int = 250,
        max_points: int = 8000,
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
    best_inliers, best_n = None, None
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
            best_inliers, best_n = inliers, n

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

def depth_to_b64_png(depth_map: np.ndarray, pil_image: Optional[Image.Image] = None) -> str:
    """
    Render depth map and optionally input image side-by-side as base64 PNG.
    """
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
    """
    Render input image with SAM food mask overlay and confidence label as base64 PNG.
    """
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
        return VolumeResult(0.0, float(plane_p0[2]), 0.0, 0.0)

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
    cos_theta = np.clip(np.abs(rays @ plane_n), 0.3, 1.0)  # floor avoids blow-up at grazing angles
    pixel_area_m2 = pixel_area_m2 / cos_theta

    total_volume_cm3 = float(np.sum(heights_m * pixel_area_m2)) * M3_TO_CM3

    nonzero = heights_m[heights_m > 0]
    if len(nonzero) == 0:
        logger.warning("  Volume | ALL heights <= 0 after clipping -> check plane orientation / segmentation")
    else:
        logger.info(
            f"  Volume | Height above plane (cm): min={nonzero.min()*100:.2f} "
            f"mean={nonzero.mean()*100:.2f} max={nonzero.max()*100:.2f} "
            f"| {n_below} px below plane ({n_below/len(heights_m)*100:.1f}%)"
        )
    logger.info(f"  Volume | {total_volume_cm3:.2f} cm^3  | food_px={len(z)}  fx={fx:.0f}  fy={fy:.0f}")

    return VolumeResult(total_volume_cm3, float(plane_p0[2]), float(heights_m.max()) * 100.0, float(heights_m.mean()) * 100.0, clipped_high_pct)

@app.post("/api/v1/estimate-volume", response_model=VolumeEstimateResponse)
async def volume_estimation(file: UploadFile = File(...)) -> VolumeEstimateResponse:
    """
    End-to-end volume estimation pipeline:
    1. Extract camera intrinsics from EXIF
    2. Estimate metric depth map (DepthPro)
    3. Segment food region (SAM 3)
    4. Integrate volume from height field
    """
    t_start = time.perf_counter()
    confidence = "high"

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
    plane_n, plane_p0, plane_inliers = fit_support_plane(depth_map, food_mask, image_info)
    volume_res = compute_volume(depth_map, food_mask, image_info, plane_n, plane_p0)
    logger.info(f"[E] Volume Estimation Result: {volume_res.volume_cm3:.2f} cm^3. ({time.perf_counter() - t_volume_start:.3f}s)")

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

    return VolumeEstimateResponse(
        volume_cm3=volume_res.volume_cm3,
        confidence=confidence,
        food_pixel_count=food_pixel_count,
        food_coverage_pct=food_coverage_pct,
        max_food_height_cm=volume_res.max_food_height_cm,
        mean_food_height_cm=volume_res.mean_food_height_cm,
        plate_depth_m=volume_res.plate_depth_m,
        intrinsics_source=image_info.source,
        debug_overlay_b64=seg_b64
    )

@torch.no_grad()
def run_custom_model(image: Image.Image) -> float:
    """
    Estimate mass (g) using the custom trained model. If the model was trained with
    --use_volume, compute the geometric volume scalar the same way the cache did and feed it in
    """
    tensor = _rgb_transform(image).unsqueeze(0)

    # Work out whether we need the scalar. Peeking at the loader's flag would build it, so load once, read the flag, and compute the scalar BEFORE
    # the final fetch, since estimate_depth/segment_food pull other models onto cuda and would otherwise evict custom_volume back to cpu mid-call.
    volume_t = None
    if getattr(get_model("custom_volume"), "_use_volume", False):
        depth_map, focal_px = estimate_depth(image)
        w, h = image.size
        cam = CameraInfo(fx=focal_px, fy=focal_px, cx=w / 2.0, cy=h / 2.0,
                         image_width=w, image_height=h, source="depthpro_fov")
        mask, _, _ = segment_food(image)
        plane_n, plane_p0, _ = fit_support_plane(depth_map, mask, cam)
        vol = compute_volume(depth_map, mask, cam, plane_n, plane_p0).volume_cm3
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


@app.post("/api/v1/estimate-volume-dl")
async def volume_estimation_dl(file: UploadFile = File(...)):
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