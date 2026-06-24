import base64
import io
import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from PIL import Image

from pillow_heif import register_heif_opener
register_heif_opener()

import piexif
import matplotlib
from rich.logging import RichHandler

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import DepthProImageProcessor, DepthProForDepthEstimation, Sam3Processor, Sam3Model

M3_TO_CM3 = 1_000_000.0  # 1 m^3 = 10⁶ cm^3
MAX_LONG_EDGE = 1280 # px

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

torch_device = torch.device(device)
processor = DepthProImageProcessor.from_pretrained("apple/DepthPro-hf")
model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(torch_device)
model.eval()
logger.info("Depth model loaded successfully.")

sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")
sam3_model = Sam3Model.from_pretrained("facebook/sam3").to(torch_device)
sam3_model.eval()
logger.info("SAM 3 loaded.")


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

    with torch.no_grad():
        outputs = model(**inputs)

    post_processed = processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(original_h, original_w)],
    )

    depth_map = post_processed[0]["predicted_depth"].cpu().numpy().astype(np.float32)
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


def segment_food(pillow_image: Image.Image, threshold: float = 0.5) -> tuple[np.ndarray, list[float]]:
    """
    Segment food region using SAM 3 with text prompt "food".
    Falls back to full-image mask if no instances found.
    :return: union mask of all detected instances and per-instance scores.
    """
    img_w, img_h = pillow_image.size
    total_pixels = img_w * img_h

    inputs = sam3_processor(images=pillow_image, text="food", return_tensors="pt").to(torch_device)

    with torch.no_grad():
        outputs = sam3_model(**inputs)

    results = sam3_processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]

    instance_masks = results["masks"]
    instance_scores = results["scores"]

    if instance_masks.shape[0] == 0:
        logger.warning("  SAM 3 | No food instances found — using full-image fallback mask")
        logger.warning("  SAM 3 | Scale correction will be unreliable (no background pixels)")
        return np.ones((img_h, img_w), dtype=np.uint8), []

    instance_scores_list = instance_scores.tolist()

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
        logger.warning(
            f"  SAM 3 | Food mask covers {union_pct:.1f}% of image — "
            f"background region too small for reliable plane fitting"
        )
    if union_pct < 2.0:
        logger.warning(
            f"  SAM 3 | Food mask covers only {union_pct:.1f}% of image — "
            f"possible segmentation failure"
        )

    return union_mask, instance_scores_list


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
        plate_depth_m: float,
) -> VolumeResult:
    """
    Integrate volume from height field: sum(h * pixel_area).
    Heights computed as h = plate_depth - Z_food, using DepthPro's metric depth directly.
    """
    food_pixels = food_mask > 0

    if not np.any(food_pixels):
        logger.warning("  Volume | No food pixels :( returning zero volume")
        return VolumeResult(0.0, plate_depth_m, 0.0, 0.0)

    food_depths = depth_map[food_pixels]

    logger.info(
        f"  Volume | Food pixel depth: "
        f"min={food_depths.min():.3f}m  median={np.median(food_depths):.3f}m  max={food_depths.max():.3f}m"
    )
    logger.info(f"  Volume | Plate surface={plate_depth_m:.3f}m")

    # Height above plate surface (negative values clipped to 0)
    heights_m = plate_depth_m - food_depths
    n_clipped = int(np.sum(heights_m < 0))
    heights_m = np.clip(heights_m, 0.0, None)

    nonzero_heights = heights_m[heights_m > 0]
    if len(nonzero_heights) == 0:
        logger.warning(
            "  Volume | ALL height values are <= 0 after clipping. "
            "Food pixels appear deeper than plate surface. Segmentation or depth may be wrong."
        )
    else:
        logger.info(
            f"  Volume | Height field (cm): "
            f"min={nonzero_heights.min()*100:.2f}  "
            f"mean={nonzero_heights.mean()*100:.2f}  "
            f"max={nonzero_heights.max()*100:.2f}  "
            f"| {n_clipped} px clipped to 0 ({n_clipped/len(heights_m)*100:.1f}%)"
        )

    # Pixel area in world coordinates at each food pixel's depth
    pixel_area_m2 = (food_depths / image_info.fx) * (food_depths / image_info.fy)
    logger.info(
        f"  Volume | Pixel area (cm²): "
        f"min={pixel_area_m2.min()*1e4:.4f}  "
        f"mean={pixel_area_m2.mean()*1e4:.4f}  "
        f"max={pixel_area_m2.max()*1e4:.4f}"
    )

    # Volume integration: sum(height * pixel_area)
    dV = heights_m * pixel_area_m2
    total_volume_m3 = float(np.sum(dV))
    total_volume_cm3 = total_volume_m3 * M3_TO_CM3

    logger.info(
        f"  Volume | Integration: sum(dV)={total_volume_m3:.8f} m^3 = {total_volume_cm3:.2f} cm^3  "
        f"| food_pixels={int(food_pixels.sum())}  fx={image_info.fx:.0f}  fy={image_info.fy:.0f}"
    )

    max_h_cm = float(heights_m.max()) * 100.0 if len(heights_m) > 0 else 0.0
    mean_h_cm = float(heights_m.mean()) * 100.0 if len(heights_m) > 0 else 0.0

    return VolumeResult(
        volume_cm3=total_volume_cm3,
        plate_depth_m=plate_depth_m,
        max_food_height_cm=max_h_cm,
        mean_food_height_cm=mean_h_cm,
    )


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
    food_mask, mask_scores = segment_food(pillow_image)  # (H, W) uint8 0/1
    food_pixel_count = int(food_mask.sum())
    food_coverage_pct = food_pixel_count / food_mask.size * 100
    logger.info(f"[C] Segmentation done. {food_pixel_count} food px ({food_coverage_pct:.1f}%) ({time.perf_counter() - t_segmentation_start:.3f}s)")

    t_plate_depth_start = time.perf_counter()
    plate_depth_m = estimate_plate_depth(depth_map, food_mask)
    logger.info(f"[D] Plate depth: {plate_depth_m:.3f}m ({time.perf_counter() - t_plate_depth_start:.3f}s)")

    t_volume_start = time.perf_counter()
    volume_res = compute_volume(depth_map, food_mask, image_info, plate_depth_m)
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
