import io
import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

from PIL import Image
import piexif
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Volume Estimation API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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

    pillow_image.show()


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
