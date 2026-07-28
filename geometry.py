"""
Camera intrinsics and the metric geometry every approach leans on: back-projection, the support-plane fit
and the height-field integral that turns a depth map plus a mask into a volume
"""

import math

import cv2
import numpy as np
import piexif

from logging_config import get_logger
from schemas import CameraInfo, VolumeResult

logger = get_logger(__name__)

M3_TO_CM3 = 1_000_000.0  # 1 m^3 = 10^6 cm^3
MAX_FOOD_HEIGHT_M = 0.15
NOMINAL_FOOD_HEIGHT_CM = 2.0 # crude portion-height prior for the low-confidence fallback


# Camera intrinsics

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
        source=source
    )


# Back-projection and mask measurement

def backproject_pixel(row: float, col: float, depth_map: np.ndarray, cam: CameraInfo, window: int = 5) -> np.ndarray:
    """
    Back-project one pixel to a 3D camera-space point (metres), using the median depth in a small window for robustness
    against per-pixel depth noise right at the tip
    """
    pixel_row, pixel_col = int(round(row)), int(round(col))
    row_start, row_end = max(pixel_row - window, 0), min(pixel_row + window + 1, depth_map.shape[0])
    col_start, col_end = max(pixel_col - window, 0), min(pixel_col + window + 1, depth_map.shape[1])
    depth_patch = depth_map[row_start:row_end, col_start:col_end]
    valid = depth_patch[depth_patch > 0]
    depth = float(np.median(valid)) if valid.size else float(depth_map[pixel_row, pixel_col])
    x_cam = (pixel_col - cam.cx) * depth / cam.fx
    y_cam = (pixel_row - cam.cy) * depth / cam.fy
    return np.array([x_cam, y_cam, depth], dtype=np.float64)


def measure_mask_endpoints(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find the two tip pixels of an elongated mask via PCA: project every mask pixel onto its principal
    axis and take the extremes. Returns two (row, col) pixel coordinates
    """
    rows, cols = np.where(mask > 0)
    coords_xy = np.stack([cols, rows], axis=1).astype(np.float64)  # (N, 2) in (x, y)
    centred_xy = coords_xy - coords_xy.mean(axis=0)
    _, _, principal_axes = np.linalg.svd(centred_xy, full_matrices=False)
    long_axis = principal_axes[0]  # first singular vector = principal (long) axis
    projections = centred_xy @ long_axis
    min_endpoint_xy = coords_xy[np.argmin(projections)]
    max_endpoint_xy = coords_xy[np.argmax(projections)]
    return min_endpoint_xy[::-1], max_endpoint_xy[::-1]  # (x, y) -> (row, col) for depth-map indexing


def metric_footprint_diameter_cm(depth_map: np.ndarray, food_mask: np.ndarray, cam: CameraInfo) -> float:
    """
    Equivalent circular diameter (cm) of the food's real-world footprint achieved by
    - sum the per-pixel metric areas over the mask
    - then D = 2*sqrt(A/pi)
    """
    food_px = food_mask > 0
    if not np.any(food_px):
        return 0.0
    ys, xs = np.where(food_px)
    z = depth_map[ys, xs].astype(np.float64)
    pixel_area_m2 = (z / cam.fx) * (z / cam.fy)
    area_cm2 = float(np.sum(pixel_area_m2)) * 10000 # m^2 -> cm^2
    return 2.0 * math.sqrt(area_cm2 / math.pi)


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


# Support plane and volume integration

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
        plane_p0: np.ndarray
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
