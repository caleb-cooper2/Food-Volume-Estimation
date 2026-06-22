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
from sklearn.linear_model import RANSACRegressor

import piexif
import matplotlib
from rich.logging import RichHandler

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import DepthProImageProcessor, DepthProForDepthEstimation, Sam3Processor, Sam3Model

ASSUMED_CAMERA_HEIGHT_M = 0.30  # 30 cm baseline prior
SCALE_CORRECTION_TOLERANCE = 0.02  # Skip correction if drift is < 2 cm
M3_TO_CM3 = 1_000_000.0  # 1 m³ = 10⁶ cm³

@dataclass
class VolumeEstimateResponse:
    volume_cm3: float
    confidence: str
    food_pixel_count: int
    food_coverage_pct: float
    max_food_height_cm: float
    mean_food_height_cm: float
    plate_depth_m: float
    scale_correction_factor: float
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
    scale_correction_factor: float
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


def estimate_depth(pillow_image: Image.Image) -> np.ndarray:
    """
    Predict metric depth map using DepthPro, upsampled to original image resolution.
    :return: float32 array clipped to [0.1, 5.0] meters.
    """
    original_w, original_h = pillow_image.size

    inputs = processor(images=pillow_image, return_tensors="pt")  # preprocesses the image to ensure conformity to the model
    inputs = {k: v.to(torch_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    depth_upsampled = torch.nn.functional.interpolate(
        outputs.predicted_depth.unsqueeze(1), # (1, 1, H_model, W_model)
        size=(original_h, original_w), # torch wants (H, W)
        mode="bilinear",
        align_corners=False,
    ).squeeze() # converts to -> (H_orig, W_orig)

    depth_map = depth_upsampled.cpu().numpy().astype(np.float32)

    logger.info(
        f"  Depth pre-clip  | "
        f"min={depth_map.min():.3f}m  "
        f"p5={np.percentile(depth_map, 5):.3f}m  "
        f"p25={np.percentile(depth_map, 25):.3f}m  "
        f"p50={np.percentile(depth_map, 50):.3f}m  "
        f"p75={np.percentile(depth_map, 75):.3f}m  "
        f"p95={np.percentile(depth_map, 95):.3f}m  "
        f"max={depth_map.max():.3f}m"
    )

    clipped_low = int(np.sum(depth_map < 0.1))
    clipped_high = int(np.sum(depth_map > 5.0))
    if clipped_low > 0 or clipped_high > 0:
        logger.warning(
            f"  Depth clipping  | {clipped_low} px below 0.1m, {clipped_high} px above 5.0m "
            f"({(clipped_low + clipped_high) / depth_map.size * 100:.1f}% of image)"
        )

    depth_map = np.clip(depth_map, 0.1, 5.0)
    return depth_map


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


def apply_scale_correction(
        depth_map: np.ndarray,
        food_mask: np.ndarray,
        image_info: CameraInfo,
        assumed_height_m: float = ASSUMED_CAMERA_HEIGHT_M
) -> tuple[np.ndarray, float, float, np.ndarray | None]:
    """
    Estimate plate depth via RANSAC plane fitting on background pixels.
    Compute scale correction factor α = assumed_height / estimated_plate_depth.
    Clamps α to [0.5, 2.0] if out of bounds.
    :return: corrected depth map, plate depth, α, and plane normal
    """
    bg_mask = food_mask == 0
    has_background = np.any(bg_mask)
    plane_normal = None

    if has_background:
        ys, xs = np.where(bg_mask)
        bg_depths = depth_map[bg_mask]

        logger.info(
            f"  ScaleCorr | Background depth distribution: "
            f"min={bg_depths.min():.3f}m  "
            f"p25={np.percentile(bg_depths, 25):.3f}m  "
            f"median={np.median(bg_depths):.3f}m  "
            f"p75={np.percentile(bg_depths, 75):.3f}m  "
            f"max={bg_depths.max():.3f}m  "
            f"std={bg_depths.std():.4f}m  "
            f"n={len(bg_depths)}"
        )

        if bg_depths.std() > 0.05:
            logger.warning(
                f"  ScaleCorr | Background depth std={bg_depths.std():.4f}m > 0.05m — "
                f"possible mask leakage or non-flat surface. RANSAC may be unreliable."
            )

        # Backproject background pixels to 3D using camera model
        X_bg = (xs - image_info.cx) * bg_depths / image_info.fx
        Y_bg = (ys - image_info.cy) * bg_depths / image_info.fy
        points_3d = np.stack([X_bg, Y_bg, bg_depths], axis=1)

        if len(points_3d) >= 10:
            try:
                ransac = RANSACRegressor(
                    residual_threshold=0.015,
                    max_trials=200,
                    min_samples=0.1,
                    random_state=42,
                )
                ransac.fit(points_3d[:, :2], points_3d[:, 2])
                inlier_mask = ransac.inlier_mask_

                inlier_count = int(inlier_mask.sum())
                inlier_ratio = inlier_count / len(points_3d)
                inlier_depths = bg_depths[inlier_mask]

                logger.info(
                    f"  ScaleCorr | RANSAC inliers: {inlier_count}/{len(points_3d)} "
                    f"({inlier_ratio*100:.1f}%)"
                )
                logger.info(
                    f"  ScaleCorr | Inlier depth: "
                    f"min={inlier_depths.min():.3f}m  "
                    f"median={np.median(inlier_depths):.3f}m  "
                    f"max={inlier_depths.max():.3f}m  "
                    f"std={inlier_depths.std():.4f}m"
                )

                if inlier_ratio < 0.3:
                    logger.warning(
                        f"  ScaleCorr | Low inlier ratio ({inlier_ratio*100:.1f}%) — "
                        f"plane fit is weak. Background may be cluttered or mask may be wrong."
                    )

                estimated_plate_depth = float(np.median(inlier_depths))

                # Fit plane equation: Z = a*X + b*Y + c
                a, b = ransac.estimator_.coef_
                normal_unnorm = np.array([-a, -b, 1.0])
                plane_normal = normal_unnorm / np.linalg.norm(normal_unnorm)

                tilt_deg = float(np.degrees(np.arccos(np.clip(plane_normal[2], -1.0, 1.0))))
                logger.info(
                    f"  ScaleCorr | Plane normal=[{plane_normal[0]:.3f}, {plane_normal[1]:.3f}, {plane_normal[2]:.3f}]  "
                    f"tilt={tilt_deg:.1f}° from nadir"
                )
                if tilt_deg > 20.0:
                    logger.warning(
                        f"  ScaleCorr | Camera tilt {tilt_deg:.1f}° > 20° — "
                        f"volume will be overestimated without tilt correction"
                    )

            except Exception as e:
                logger.warning(f"  ScaleCorr | RANSAC failed: {e} — falling back to median")
                estimated_plate_depth = float(np.median(bg_depths))
        else:
            logger.warning(f"  ScaleCorr | Only {len(points_3d)} background points — too few for RANSAC, using median")
            estimated_plate_depth = float(np.median(bg_depths))
    else:
        logger.warning("  ScaleCorr | No background pixels — scale correction unreliable")
        estimated_plate_depth = float(np.median(depth_map))

    naive_median = float(np.median(depth_map[bg_mask])) if has_background else float(np.median(depth_map))
    logger.info(
        f"  ScaleCorr | Plate depth: RANSAC={estimated_plate_depth*100:.1f}cm  "
        f"naive_median={naive_median*100:.1f}cm  "
        f"prior={assumed_height_m*100:.1f}cm"
    )

    scale_correction_factor = assumed_height_m / estimated_plate_depth
    logger.info(f"  ScaleCorr | α = {assumed_height_m:.3f} / {estimated_plate_depth:.3f} = {scale_correction_factor:.4f}")

    # Clamp α to reasonable range; extreme values indicate depth estimation failure
    if not (0.5 <= scale_correction_factor <= 2.0):
        logger.warning(
            f"  ScaleCorr | α={scale_correction_factor:.4f} outside [0.5, 2.0] — clamping to 1.0. "
            f"Plate depth ({estimated_plate_depth*100:.1f}cm) far from prior ({assumed_height_m*100:.1f}cm)."
        )
        scale_correction_factor = 1.0

    return depth_map.copy(), estimated_plate_depth, scale_correction_factor, plane_normal


def compute_volume(
        depth_map: np.ndarray,
        food_mask: np.ndarray,
        image_info: CameraInfo,
        estimated_plate_depth_m: float,
        scale_correction_factor: float,
) -> VolumeResult:
    """
    Integrate volume from height field: sum(h * pixel_area).
    Heights computed as h = plate_depth_corrected - Z_food_corrected.
    """
    food_pixels = food_mask > 0

    if not np.any(food_pixels):
        logger.warning("  Volume | No food pixels — returning zero volume")
        return VolumeResult(0.0, estimated_plate_depth_m, scale_correction_factor, 0.0, 0.0)

    food_depths = depth_map[food_pixels]
    food_depths_corrected = food_depths * scale_correction_factor
    plate_depth_corrected = estimated_plate_depth_m * scale_correction_factor

    logger.info(
        f"  Volume | Food pixel depth (raw): "
        f"min={food_depths.min():.3f}m  median={np.median(food_depths):.3f}m  max={food_depths.max():.3f}m"
    )
    logger.info(
        f"  Volume | Corrected food depth: "
        f"min={food_depths_corrected.min():.3f}m  median={np.median(food_depths_corrected):.3f}m  max={food_depths_corrected.max():.3f}m"
    )
    logger.info(
        f"  Volume | Plate surface (corrected)={plate_depth_corrected:.3f}m  "
        f"Expected food above this — if food depth > plate depth, heights will clip to 0"
    )

    # Height above plate surface (negative values clipped to 0)
    heights_m = plate_depth_corrected - food_depths_corrected
    n_clipped_negative = int(np.sum(heights_m < 0))
    heights_m = np.clip(heights_m, 0.0, None)

    nonzero_heights = heights_m[heights_m > 0]
    if len(nonzero_heights) == 0:
        logger.warning(
            f"  Volume | ALL height values are <= 0 after clipping. "
            f"Food pixels appear deeper than plate surface. "
            f"Scale correction or segmentation is wrong."
        )
    else:
        logger.info(
            f"  Volume | Height field (cm): "
            f"min={nonzero_heights.min()*100:.2f}  "
            f"mean={nonzero_heights.mean()*100:.2f}  "
            f"max={nonzero_heights.max()*100:.2f}  "
            f"| {n_clipped_negative} px clipped to 0 "
            f"({n_clipped_negative/len(heights_m)*100:.1f}%)"
        )

    # Pixel area in world coordinates
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
        f"  Volume | Integration: sum(dV)={total_volume_m3:.8f} m³ = {total_volume_cm3:.2f} cm³  "
        f"| food_pixels={int(food_pixels.sum())}  fx={image_info.fx:.0f}  fy={image_info.fy:.0f}"
    )

    max_h_cm = float(heights_m.max()) * 100.0 if len(heights_m) > 0 else 0.0
    mean_h_cm = float(heights_m.mean()) * 100.0 if len(heights_m) > 0 else 0.0

    return VolumeResult(
        volume_cm3=total_volume_cm3,
        plate_depth_m=estimated_plate_depth_m,
        scale_correction_factor=scale_correction_factor,
        max_food_height_cm=max_h_cm,
        mean_food_height_cm=mean_h_cm
    )


@app.post("/api/v1/estimate-volume", response_model=VolumeEstimateResponse)
async def volume_estimation(file: UploadFile = File(...)) -> VolumeEstimateResponse:
    """
    End-to-end volume estimation pipeline:
    1. Extract camera intrinsics from EXIF
    2. Estimate metric depth map (DepthPro)
    3. Segment food region (SAM 3)
    4. Correct scale via plate plane fitting (RANSAC)
    5. Integrate volume from height field
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
    depth_map = estimate_depth(pillow_image)
    logger.info(
        f"[B] Depth inference done. "
        f"Range: [{depth_map.min():.3f}, {depth_map.max():.3f}] m "
        f"({time.perf_counter()-t_depth_start:.3f}s)"
    )

    t_segmentation_start = time.perf_counter()
    food_mask, mask_scores = segment_food(pillow_image)  # (H, W) uint8 0/1
    food_pixel_count = int(food_mask.sum())
    food_coverage_pct = food_pixel_count / food_mask.size * 100
    logger.info(f"[C] Segmentation done. {food_pixel_count} food px ({food_coverage_pct:.1f}%) ({time.perf_counter() - t_segmentation_start:.3f}s)")

    t_scale_start = time.perf_counter()
    depth_map_corrected, raw_plate_depth, alpha, plane_normal = apply_scale_correction(depth_map, food_mask, image_info)
    logger.info(f"[D] Scale correction done. ({time.perf_counter() - t_scale_start:.3f}s)")

    t_volume_start = time.perf_counter()
    volume_res = compute_volume(
        depth_map=depth_map,
        food_mask=food_mask,
        image_info=image_info,
        estimated_plate_depth_m=raw_plate_depth,
        scale_correction_factor=alpha
    )
    logger.info(f"[E] Volume Estimation Result: {volume_res.volume_cm3:.2f} cm³. ({time.perf_counter() - t_volume_start:.3f}s)")

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
        scale_correction_factor=volume_res.scale_correction_factor,
        intrinsics_source=image_info.source,
        debug_overlay_b64=seg_b64
    )
