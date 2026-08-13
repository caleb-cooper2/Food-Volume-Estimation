"""
Approach A - Monocular geometric

Single RGB image -> metric depth + food mask -> support plane -> height-field integral
"""

import time

import numpy as np
from fastapi import APIRouter, File, UploadFile, Form

from approaches import read_upload, validate_participant_code
from reference_images import depth_to_b64_png, overlay_food_mask_b64, write_b64_png
from depth import estimate_depth
from geometry import extract_image_info, fit_support_plane, compute_volume
from logging_config import get_logger
from nlp_client import extract_entities
from participant_data import record_request
from scale import resolve_scale_correction
from schemas import CameraInfo, EstimationResponse, FoodItemResult
from segmentation import segment_food, segment_food_items, make_masks_disjoint
from volume_to_mass import scale_nutrients, volume_to_mass

logger = get_logger(__name__)

router = APIRouter()

DEPTH_DEBUG_PATH = "/tmp/depth_debug.png"
SEGMENTATION_DEBUG_PATH = "/tmp/seg_overlay.png"


@router.post("/api/v1/estimate-volume", response_model=EstimationResponse)
async def volume_estimation(
        file: UploadFile = File(...),
        participant_code: str = Form(...),
        scale_ref: str = Form("utensil"),
        text: str = Form("")
) -> EstimationResponse:
    """
    End-to-end volume estimation pipeline:
    1. Extract camera intrinsics from EXIF
    2. Estimate metric depth map (DepthPro)
    3. Segment food region (SAM 3)
    4. Integrate volume from height field
    """
    t_start = time.perf_counter()

    participant_code = validate_participant_code(participant_code)
    logger.info(f"[start] participant={participant_code}")

    image_bytes, pillow_image = await read_upload(file)

    entities = await extract_entities(text)
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
        source="depthpro_fov"
    )

    t_segmentation_start = time.perf_counter()
    food_items = segment_food_items(pillow_image, entities)
    if not food_items:
        # No text, or nothing matched a named prompt -> one generic pass so the image path still works
        generic_mask, generic_scores, _ = segment_food(pillow_image, prompt="food or drink")
        food_items = [FoodItemResult(prompt="food", mask=generic_mask, score=float(np.mean(generic_scores)) if generic_scores else 0.0)]

    make_masks_disjoint(food_items)
    food_items = [item for item in food_items if int(item.mask.sum()) > 0]

    # Union drives the support-plane ring and the scale anchor, both of which are scene-wide properties
    food_mask = np.zeros(food_items[0].mask.shape, dtype=np.uint8)
    for item in food_items:
        food_mask |= item.mask
    mask_scores = [item.score for item in food_items]

    food_pixel_count = int(food_mask.sum())
    food_coverage_pct = food_pixel_count / food_mask.size * 100
    logger.info(f"[C] Segmentation done. {len(food_items)} item(s), {food_pixel_count} food px ({food_coverage_pct:.1f}%) ({time.perf_counter() - t_segmentation_start:.3f}s)")

    t_reference_start = time.perf_counter()
    scale_correction, scale_source = resolve_scale_correction(pillow_image, depth_map, food_mask, image_info, scale_ref)

    if scale_correction is not None:
        depth_map = depth_map * scale_correction
        logger.info(f"[B*] Scale anchor '{scale_ref}' -> {scale_source} x{scale_correction:.3f} ({time.perf_counter() - t_reference_start:.3f}s)")
    else:
        logger.info(f"[B*] Scale anchor '{scale_ref}' found nothing -> using depth as-is ({time.perf_counter() - t_reference_start:.3f}s)")

    t_volume_start = time.perf_counter()
    # One plane for the whole scene: every item sits on the same surface, and fitting per item would reference each food to a slightly different plate
    plane_n, plane_p0, _ = fit_support_plane(depth_map, food_mask, image_info)

    for item in food_items:
        item_volume = compute_volume(depth_map, item.mask, image_info, plane_n, plane_p0)
        item.volume_cm3 = item_volume.volume_cm3
        item.geometry_confidence = item_volume.geometry_confidence
        density_block = item.entity["match"]["density"] if item.entity and item.entity.get("match") else None
        item.mass = volume_to_mass(item.volume_cm3, density_block)
        if item.mass and item.entity and item.entity.get("match"):
            item.nutrients = scale_nutrients(
                item.entity["match"].get("nutrients"),
                item.entity.get("grams"),
                item.mass["mass_g"]
            )
        logger.info(
            f"  Items | '{item.prompt}' -> {item.volume_cm3:.1f} cm^3"
            + (f", {item.mass['mass_g']:.1f} g ({item.mass['density_source']})" if item.mass else ", no density")
        )

    total_volume_cm3 = sum(item.volume_cm3 for item in food_items)
    priced_items = [item for item in food_items if item.mass]
    total_mass_g = round(sum(item.mass["mass_g"] for item in priced_items), 1) if priced_items else None

    nutrient_items = [item for item in food_items if item.nutrients]
    total_nutrients = None
    if nutrient_items:
        total_nutrients = {}
        for item in nutrient_items:
            for nutrient, value in item.nutrients.items():
                if value is not None:
                    total_nutrients[nutrient] = round(total_nutrients.get(nutrient, 0.0) + value, 2)

    # Volume-weighted so one tiny low-confidence item does not drag down an otherwise ok total
    geometry_confidence = float(np.average(
        [item.geometry_confidence for item in food_items],
        weights=[max(item.volume_cm3, 1e-6) for item in food_items],
    ))

    logger.info(
        f"[E] Volume Estimation Result: {total_volume_cm3:.2f} cm^3 across {len(food_items)} item(s). "
        f"({time.perf_counter() - t_volume_start:.3f}s)"
    )
    if len(priced_items) < len(food_items):
        logger.warning(f"[E] Mass total covers {len(priced_items)}/{len(food_items)} items -> the rest had no density")

    confidence = (
        "high" if geometry_confidence >= 0.66 else
        "medium" if geometry_confidence >= 0.33 else
        "low"
    )
    if confidence != "high":
        logger.warning(
            f"[E] Geometry confidence {geometry_confidence:.2f} ({confidence}) -> "
            f"view likely too oblique for a reliable single-view volume"
        )

    # Render debug visualisations
    write_b64_png(depth_to_b64_png(depth_map, pil_image=pillow_image), DEPTH_DEBUG_PATH)
    write_b64_png(overlay_food_mask_b64(pillow_image, food_mask, mask_scores), SEGMENTATION_DEBUG_PATH)

    logger.info(f"Total pipeline time: {time.perf_counter() - t_start:.3f}s")

    response = EstimationResponse(
        approach="monocular-geometric",
        volume_cm3=round(total_volume_cm3, 2),
        mass_g=total_mass_g,
        confidence=confidence,
        diagnostics={
            "participant_code": participant_code,
            "food_pixel_count": food_pixel_count,
            "food_coverage_pct": food_coverage_pct,
            "plate_depth_m": float(plane_p0[2]),
            "intrinsics_source": image_info.source,
            "scale_ref": scale_ref,
            "scale_source": scale_source,
            "scale_factor": round(scale_correction, 4) if scale_correction is not None else None,
            "items": [
                {
                    "prompt": item.prompt,
                    "matched_food": item.entity["match"]["name"] if item.entity and item.entity.get("match") else None,
                    "volume_cm3": round(item.volume_cm3, 1),
                    "mass_g": item.mass["mass_g"] if item.mass else None,
                    "mass_interval_g": [item.mass["mass_low_g"], item.mass["mass_high_g"]] if item.mass else None,
                    "density_source": item.mass["density_source"] if item.mass else None,
                    "presentation": item.mass["presentation"] if item.mass else None,
                    "nutrients": item.nutrients,
                    "coverage_pct": round(int(item.mask.sum()) / item.mask.size * 100, 2),
                    "segmentation_score": round(item.score, 3),
                    "geometry_confidence": item.geometry_confidence
                }
                for item in food_items
            ],
            "items_with_masses": len(priced_items),
            "items_with_nutrients": len(nutrient_items),
            "total_nutrients": total_nutrients,
            "nlp_available": bool(entities)
        }
    )
    record_request(participant_code, image_bytes, file.content_type, {"text": text, "scale_ref": scale_ref, "entities": entities}, response)
    return response
