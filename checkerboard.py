from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

CHECKERBOARD_INNER = (4, 3)  # (cols, rows) of inner corners
CHECKERBOARD_SQUARE_CM = 1.2
CHECKERBOARD_SQUARE_M = CHECKERBOARD_SQUARE_CM / 100.0
DETECT_MAX_WIDTH = 1280  # locate corners on a downscaled copy
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def find_corners(image_bgr: np.ndarray):
    """Locate board's inner corners, returns (corners (N,1,2) in full-res px, cols, rows) or None"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = min(1.0, DETECT_MAX_WIDTH / w)
    small = cv2.resize(gray, (round(w * scale), round(h * scale))) if scale < 1.0 else gray
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    for cols, rows in (CHECKERBOARD_INNER, CHECKERBOARD_INNER[::-1]): # board may sit either orientation
        found, corners = cv2.findChessboardCorners(small, (cols, rows), flags)
        if found:
            break
    if not found:
        return None

    corners = (corners / scale).astype(np.float32)  # back to full-res pixel coords
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)
    return corners, cols, rows


def adjacent_corner_pixel_pairs(corners: np.ndarray, cols: int, rows: int) -> list:
    """
    Neighbouring inner-corner pairs as ((row, col), (row, col)) with their true separation being CHECKERBOARD_SQUARE_M. OpenCV gives
    corners as (x, y)... returned here as (row, col) to match backproject / world-sample helpers
    """
    grid = corners.reshape(rows, cols, 2)
    pairs = []
    for r in range(rows):
        for c in range(cols):
            x, y = grid[r, c]
            if c + 1 < cols:
                x2, y2 = grid[r, c + 1]
                pairs.append(((float(y), float(x)), (float(y2), float(x2))))
            if r + 1 < rows:
                x2, y2 = grid[r + 1, c]
                pairs.append(((float(y), float(x)), (float(y2), float(x2))))
    return pairs


@dataclass
class CheckerboardPose:
    found: bool
    scale_cm_per_px: Optional[float] = None
    tilt_deg: Optional[float] = None  # board-plane angle off the optical axis -> 0 = straight-on / overhead
    n_corners: int = 0


def detect_pose(image_path: Path) -> CheckerboardPose:
    """Scale + camera-tilt reference from one image, for the benchmark's manifest and tilt stratification"""
    img = cv2.imread(str(image_path))
    if img is None:
        return CheckerboardPose(False)
    found = find_corners(img)
    if found is None:
        return CheckerboardPose(False)
    corners, cols, rows = found
    scale = corner_spacing_scale(corners, cols, rows)
    tilt = checkerboard_tilt(corners, cols, rows, img.shape)
    return CheckerboardPose(True, round(scale, 5), round(tilt, 1), len(corners))


def corner_spacing_scale(corners: np.ndarray, cols: int, rows: int) -> float:
    """cm-per-pixel from the median neighbour spacing. Mild tilt foreshortens this slightly but good enough as a reference"""
    grid = corners.reshape(rows, cols, 2)
    dx = np.linalg.norm(np.diff(grid, axis=1), axis=2)
    dy = np.linalg.norm(np.diff(grid, axis=0), axis=2)
    spacing_px = float(np.median(np.concatenate([dx.ravel(), dy.ravel()])))
    return CHECKERBOARD_SQUARE_CM / spacing_px


def checkerboard_tilt(corners: np.ndarray, cols: int, rows: int, img_shape) -> float:
    image_height, image_width = img_shape[:2]
    focal_px = 1.2 * max(image_width, image_height)
    camera_matrix = np.array([[focal_px, 0, image_width / 2.0], [0, focal_px, image_height / 2.0], [0, 0, 1]], dtype=np.float64)

    board_corners_cm = np.zeros((cols * rows, 3), np.float32)
    board_corners_cm[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * CHECKERBOARD_SQUARE_CM
    solved, rotation_vector, translation_vector = cv2.solvePnP(board_corners_cm, corners, camera_matrix, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not solved:
        return float("nan")

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    board_normal = rotation_matrix[:, 2]  # board +z axis expressed in the camera frame
    view_direction = translation_vector.ravel() / np.linalg.norm(translation_vector) # camera -> board centre
    normal_view_cos = abs(float(board_normal @ view_direction))
    return float(np.degrees(np.arccos(np.clip(normal_view_cos, 0.0, 1.0))))
