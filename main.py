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
import piexif
import matplotlib
from rich.logging import RichHandler

matplotlib.use("Agg")  # headless — must be before pyplot import
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForDepthEstimation, AutoImageProcessor, Sam3Processor, Sam3Model


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
    fx: float # horizontal focal length in pixels
    fy: float # vertical focal length in pixels
    cx: float # principal point x (pixels)
    cy: float # principal point y (pixels)
    image_width: int
    image_height: int
    source: str # 'exif' | 'error'


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
processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf").to(torch_device)
model.eval()
logger.info("Depth model loaded successfully.")

sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")
sam3_model = Sam3Model.from_pretrained("facebook/sam3").to(torch_device)
sam3_model.eval()
logger.info("SAM 3 loaded.")

def extract_image_info(image_bytes: bytes) -> CameraInfo:
    focal_35mm = 0.0
    img_w, img_h = 0, 0

    try:
        exif_dict = piexif.load(image_bytes)
        exif_ifd = exif_dict.get("Exif", {})

        # FocalLengthIn35mmFilm -> stored as unsigned short
        if piexif.ExifIFD.FocalLengthIn35mmFilm in exif_ifd:
            focal_35mm = float(exif_ifd[piexif.ExifIFD.FocalLengthIn35mmFilm])

        # Image dimensions -> PixelXDimension / PixelYDimension
        if piexif.ExifIFD.PixelXDimension in exif_ifd:
            img_w = int(exif_ifd[piexif.ExifIFD.PixelXDimension])
        if piexif.ExifIFD.PixelYDimension in exif_ifd:
            img_h = int(exif_ifd[piexif.ExifIFD.PixelYDimension])

    except Exception:
        pass

    source = "exif"
    if focal_35mm == 0.0 or focal_35mm <= 0:
        source = "error"

    if img_w == 0 or img_h == 0:
        source = "error"

    # Horizontal FoV calculation based on 35mm equivalent focal length and sensor width
    full_frame_width_mm = 36.0
    half_fov_h = math.atan(full_frame_width_mm / (2.0 * focal_35mm))

    # Pixel-space focal lengths
    fx = img_w / (2.0 * math.tan(half_fov_h))
    fy = fx  # assumption that camera sensor from phone is square/squarish

    cx = img_w / 2.0
    cy = img_h / 2.0

    return CameraInfo(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        image_width=img_w,
        image_height=img_h,
        source=source,
    )


def estimate_depth(pillow_image: Image.Image) -> np.ndarray:
    original_w, original_h = pillow_image.size

    inputs = processor(images=pillow_image, return_tensors="pt") # preprocesses the image to ensure conformaty to the model
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
    depth_map = np.clip(depth_map, 0.1, 5.0)
    return depth_map

def depth_to_b64_png(depth_map: np.ndarray, pil_image: Optional[Image.Image] = None) -> str:
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
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)

    # Left panel — original
    axes[0].imshow(pil_image)
    axes[0].set_title("Input image", fontsize=11)
    axes[0].axis("off")

    # Right panel — overlay
    axes[1].imshow(pil_image)

    # Green mask overlay at 45% opacity
    rgba_mask = np.zeros((*food_mask.shape, 4), dtype=np.float32)
    rgba_mask[food_mask.astype(bool)] = [0.0, 0.85, 0.3, 0.45]
    axes[1].imshow(rgba_mask)

    # Confidence label derived from mean score of detected instances
    if scores:
        mean_score = float(np.mean(scores))
        confidence_label = (
            "High"   if mean_score >= 0.75 else
            "Medium" if mean_score >= 0.50 else
            "Low"
        )
        color = "#22c55e" if mean_score >= 0.75 else "#f59e0b" if mean_score >= 0.50 else "#ef4444"
        n_instances = len(scores)
        label_text = f"Confidence: {confidence_label} ({mean_score:.2f})  |  Instances: {n_instances}"
    else:
        label_text = "Confidence: N/A  |  Instances: 0 (fallback mask)"
        color = "#6b7280"

    patch = mpatches.Patch(facecolor="#00d94f", edgecolor="white", linewidth=1.5,label="Food mask")
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
    img_w, img_h = pillow_image.size

    inputs = sam3_processor(images=pillow_image, text="food or drink", return_tensors="pt").to(torch_device)

    with torch.no_grad():
        outputs = sam3_model(**inputs)

    results = sam3_processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]

    masks = results["masks"]
    scores = results["scores"]

    if masks.shape[0] == 0:
        logger.warning("SAM 3 found no food instances... returning full-image fallback mask")
        return np.ones((img_h, img_w), dtype=np.uint8), []

    list_of_scores = scores.tolist()
    logger.info(f"SAM 3 found {masks.shape[0]} food instances, scores: {list_of_scores}")

    union_mask = masks.any(dim=0).cpu().numpy().astype(np.uint8) # multiple foods or split up foods, so need to union them
    return union_mask, list_of_scores

@app.post("/api/v1/estimate-volume", response_model=VolumeEstimateResponse)
async def volume_estimation(file: UploadFile = File(...)) -> VolumeEstimateResponse:
    t_start = time.perf_counter()

    if file.content_type not in ("image/jpeg", "image/jpg", "image/png"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Send JPEG or PNG only",
        )

    image_bytes = await file.read()

    if len(image_bytes) < 1024:
        raise HTTPException(status_code=400, detail="Image too small -> minimum 1 KB")
    if len(image_bytes) > 30 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large -> maximum 30 MB")


    t_image_info_start = time.perf_counter()
    image_info = extract_image_info(image_bytes)
    logger.info(
        f"[A] Intrinsics ({image_info.source}): "
        f"fx={image_info.fx:.0f}, fy={image_info.fy:.0f}, "
        f"cx={image_info.cx:.0f}, cy={image_info.cy:.0f} "
        f"({time.perf_counter()-t_image_info_start:.3f}s)"
    )

    try:
        pillow_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}")


    t_depth_start = time.perf_counter()
    depth_map = estimate_depth(pillow_image)   # (H, W) float32, metres
    logger.info(
        f"[B] Depth inference done. "
        f"Range: [{depth_map.min():.3f}, {depth_map.max():.3f}] m "
        f"({time.perf_counter()-t_depth_start:.3f}s)"
    )

    # Render and save the depth map
    depth_b64 = depth_to_b64_png(depth_map, pil_image=pillow_image)
    depth_bytes = base64.b64decode(depth_b64)
    with open("/tmp/depth_debug.png", "wb") as f:
        f.write(depth_bytes)

    t_segmentation_start = time.perf_counter()
    food_mask, mask_scores = segment_food(pillow_image) # (H, W) uint8 0/1
    food_pixel_count = int(food_mask.sum())
    food_coverage_pct = food_pixel_count / food_mask.size * 100
    logger.info(f"[C] Segmentation done. {food_pixel_count} food px ({food_coverage_pct:.1f}%) ({time.perf_counter()-t_segmentation_start:.3f}s)")

    # Render and save the segmentation mask
    seg_b64 = overlay_food_mask_b64(pillow_image, food_mask, mask_scores)
    seg_bytes = base64.b64decode(seg_b64)
    with open("/tmp/seg_overlay.png", "wb") as f:
        f.write(seg_bytes)

    t_total = time.perf_counter() - t_start
    logger.info(f"Total pipeline time: {t_total:.3f}s")

    return VolumeEstimateResponse(
        volume_cm3=1,
        confidence="high",
        food_pixel_count=1,
        food_coverage_pct=1,
        max_food_height_cm=1,
        mean_food_height_cm=1,
        plate_depth_m=1,
        scale_correction_factor=1,
        intrinsics_source=image_info.source,
        debug_overlay_b64="1"
    )
