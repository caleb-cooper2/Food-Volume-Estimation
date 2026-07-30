from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EstimationResponse:
    """
    volume_cm3 or mass_g gets filled depending on what the approach actually predicts (the geometric and multi-view routes give a volume, the trained model gives a mass), and anything
    approach-specific gets added in diagnostics so the top-level shape stays identical across all three
    """
    approach: str # one of 'monocular-geometric' | 'deep-learning' | 'multi-view'
    volume_cm3: Optional[float]
    mass_g: Optional[float]
    confidence: Optional[str]
    diagnostics: dict # approach-specific extras (heights, coverage, scale, semantic fusion, debug overlay, ...)


@dataclass
class CameraInfo:
    """Pinhole camera model parameters derived from EXIF or fallback"""
    fx: float  # horizontal focal length in pixels
    fy: float  # vertical focal length in pixels
    cx: float  # principal point x (pixels)
    cy: float  # principal point y (pixels)
    image_width: int
    image_height: int
    source: str  # 'exif' | 'fallback' | 'depthpro_fov'


@dataclass
class VolumeResult:
    """Computed volume and intermediate metric values"""
    volume_cm3: float
    plate_depth_m: float
    max_food_height_cm: float
    mean_food_height_cm: float
    clipped_high_pct: float = 0.0
    geometry_confidence: float = 1.0  # 0..1, drops on oblique views / heavy clipping where the height-field integral is unreliable


@dataclass
class FoodItemResult:
    """One named food: its mask and everything derived from it"""
    prompt: str
    mask: np.ndarray
    score: float # mean SAM 3 instance score
    entity: Optional[dict] = None # the NLP entity it came from
    volume_cm3: float = 0.0
    geometry_confidence: float = 0.0
    mass: Optional[dict] = None # volume_to_mass output, None when no density was available
    nutrients: Optional[dict] = None # entity's NLP nutrients rescaled to mass_g, None when no mass was available
