import io
import json
import logging
import os
import tempfile

import cv2
import numpy as np
import open3d as o3d
import torch
import trimesh
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

from main import estimate_depth, segment_food
from model_manage import register_loader, get_model, preload_all

logger = logging.getLogger(__name__)

app = FastAPI(title="Volume Estimation API - Multi-Image")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

SEMANTIC_CV_FLOOR = 0.15  # rough uncertainty we give the geometric estimate when fusing with the NLP prior

register_loader("vggt", lambda: VGGT.from_pretrained("facebook/VGGT-1B"))

def load_and_preprocess_images_from_pil(pil_images: list[Image.Image]) -> torch.Tensor:
    with tempfile.TemporaryDirectory() as temp_directory:
        paths = []
        for i, image in enumerate(pil_images):
            path = os.path.join(temp_directory, f"temp_image_{i}.png")
            image.save(path)
            paths.append(path)
        return load_and_preprocess_images(paths)

def compute_volume_from_mesh(food_points_metric: np.ndarray) -> float:
    """
    Fit a Poisson surface mesh to the food point cloud and compute its watertight volume.
    Requires the point cloud to have normals
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(food_points_metric)
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(10)

    # Poisson reconstruction -> produces a watertight mesh
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd)

    o3d.io.write_triangle_mesh("/tmp/debug_food_mesh.ply", mesh)
    #image_names = ["data/camera_Aframe001.jpeg", "data/camera_Bframe027.jpeg", "data/camera_Aframe029.jpeg"]

    # Which can be converted to a trimesh and volume can be derived
    trimesh_result = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles)
    )
    volume_m3 = abs(trimesh_result.volume) # it's in m3 as that's what depth map + vggt provides
    return volume_m3 * 1_000_000.0  # computer to cm^3


def compute_volume_per_instance(instance_masks: list[np.ndarray], world_points_metric: np.ndarray) -> float:
    """
    Rather than computing volume from one blob all together, attempt to be more precise and compute each segmented
    instance by separating each mask. Seems to improve on the compute by mesh implementation?
    """
    total_volume_cm3 = 0.0
    target_h, target_w = world_points_metric.shape[0], world_points_metric.shape[1]

    for i, mask in enumerate(instance_masks):
        mask_resized = cv2.resize(
            mask,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST
        )
        pts = world_points_metric[mask_resized > 0]

        if len(pts) < 50:
            logger.info(f"Skipping instance {i} with less than 50 points")
            continue

        try:
            vol_cm3 = compute_volume_from_mesh(pts)
        except Exception as e:
            logger.error(f"Error computing volume for instance {i}: {e}")
            continue

        logger.info(f"Instance {i} volume: {vol_cm3:.2f} cm^3 with {len(pts)} points")
        total_volume_cm3 += vol_cm3

    return total_volume_cm3


def compute_vggt_crop_geometry(orig_width: int, orig_height: int, target_size: int = 518) -> dict:
    """Mirrors load_and_preprocess_images' crop-mode geometry exactly, for any aspect ratio."""
    new_width = target_size
    new_height = round(orig_height * (new_width / orig_width) / 14) * 14

    crop_top = 0
    if new_height > target_size:
        crop_top = (new_height - target_size) // 2

    return {
        "resize_width": new_width,
        "resize_height": new_height,
        "crop_top": crop_top,
        "final_height": min(new_height, target_size),
    }


def align_depth_to_vggt(depth_map: np.ndarray, orig_width: int, orig_height: int, target_size: int = 518) -> np.ndarray:
    """Resizes and center-crops depth_map to match VGGT's actual working-resolution output"""
    geo = compute_vggt_crop_geometry(orig_width, orig_height, target_size)

    resized = cv2.resize(
        depth_map,
        (geo["resize_width"], geo["resize_height"]),
        interpolation=cv2.INTER_LINEAR,
    )

    if geo["resize_height"] > target_size:
        top = geo["crop_top"]
        resized = resized[top: top + target_size, :]

    return resized


def compute_mask_overlap_coefficient(instance_masks):
    """
    Computes the overlap coefficient for a set of binary instance masks. (ratio of area of intersection to area of smaller mask)
    The coefficient can then be used to perform non-maximum suppression (NMS) to remove duplicate masks that overlap too much
    """
    n_instances = len(instance_masks)
    areas = np.array([m.sum() for m in instance_masks], dtype=np.float64)
    iou = np.zeros((n_instances, n_instances))
    overlap_coefficient = np.zeros((n_instances, n_instances))

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            intersection = np.logical_and(instance_masks[i], instance_masks[j]).sum()
            union = areas[i] + areas[j] - intersection
            iou[i, j] = iou[j, i] = intersection / union if union > 0 else 0.0 # interception over union
            min_area = min(areas[i], areas[j])
            overlap_coefficient[i, j] = overlap_coefficient[j, i] = intersection / min_area if min_area > 0 else 0.0

    return overlap_coefficient


def mask_non_maximum_suppression(instance_masks, scores, overlap_coefficient, containment_threshold=0.6):
    """Perform NMS over the instance masks based on their scores and overlap coefficient. Masks with high overlap are suppressed"""
    order = list(np.argsort(scores)[::-1])  # highest score first
    keep, suppressed = [], set()

    for index in order:
        if index in suppressed:
            continue
        keep.append(index)
        for j in order:
            if j == index or j in suppressed:
                continue
            if overlap_coefficient[index, j] > containment_threshold:
                suppressed.add(j)

    return [instance_masks[i] for i in keep]


def fuse_lognormal(v_geo: float, cv_geo: float, v_prior: float, cv_prior: float) -> dict:
    """Blend the geometric volume with the NLP portion prior in log space (volumes are positive
    with multiplicative error, so log-normal is the natural fit). Inverse-variance weighting, so
    whichever estimate is more confident pulls harder."""
    var_g = np.log1p(cv_geo ** 2)
    var_p = np.log1p(cv_prior ** 2)
    wg, wp = 1.0 / var_g, 1.0 / var_p
    mu = (wg * np.log(v_geo) + wp * np.log(v_prior)) / (wg + wp)
    var = 1.0 / (wg + wp)
    return {"volume_cm3": round(float(np.exp(mu)), 2), "cv": round(float(np.sqrt(np.expm1(var))), 3)}


def parse_semantic_context(context: str | None) -> dict | None:
    """Optional JSON prior from the NLP pipeline, e.g. {"total_prior_cm3": 240, "total_prior_cv": 0.3}.
    Only total_prior_cm3 is needed. Returns None if it's missing/unparseable so fusion just gets skipped."""
    if not context:
        return None
    try:
        data = json.loads(context)
        if data.get("total_prior_cm3"):
            return data
    except Exception as e:
        logger.warning(f"Could not parse semantic context: {e}")
    return None


@app.on_event("startup")
def startup_event():
    preload_all()


@app.post("/api/v1/estimate-volume-multiview")
async def volume_estimation_multiview(
        files: list[UploadFile] = File(...),
        context: str | None = Form(None),  # optional NLP semantic prior, JSON string
):
    # no minimum image amount as vggt outlines that single image performance is still acceptable?
    if len(files) > 10:
        raise HTTPException(400, "Maximum 10 images supported")

    pil_images = []
    for f in files:
        raw = await f.read()
        pil_images.append(Image.open(io.BytesIO(raw)).convert("RGB"))

    # Primary image for SAM 3 segmentation
    primary = pil_images[0]
    food_mask, mask_scores, instance_masks = segment_food(primary)

    overlap_coef = compute_mask_overlap_coefficient(instance_masks)
    instance_masks_refined = mask_non_maximum_suppression(instance_masks, mask_scores, overlap_coef)
    logger.info(
        f"Prior to NMS: {len(instance_masks)} instances, after NMS: {len(instance_masks_refined)} instances "
        f"The ratio between summed instance area to union mask area is now {sum(m.sum() for m in instance_masks_refined) / food_mask.sum():.3f}"
    )

    # VGGT reconstruction
    images_tensor = load_and_preprocess_images_from_pil(pil_images)
    vggt_model = get_model("vggt")
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=dtype):
            predictions = vggt_model(images_tensor.to(torch.device(device)))

    world_points = predictions["world_points"][0, 0].cpu().numpy()  # (H, W, 3)
    vggt_depth = predictions["depth"][0, 0].cpu().numpy().squeeze(-1)  # (H, W)

    print(f"world_points shape: {world_points.shape}")
    print(f"X range: {world_points[..., 0].min():.3f} to {world_points[..., 0].max():.3f}")
    print(f"Y range: {world_points[..., 1].min():.3f} to {world_points[..., 1].max():.3f}")
    print(f"Z range: {world_points[..., 2].min():.3f} to {world_points[..., 2].max():.3f}")

    # Resize food mask to VGGT's working resolution
    food_mask_resized = cv2.resize(
        food_mask.astype(np.uint8),
        (vggt_depth.shape[1], vggt_depth.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    bg_mask = food_mask_resized == 0

    # Have to get a scale anchor using DepthPro on primary image, resized to match VGGT
    depth_metric, focal_px = estimate_depth(primary)
    depth_metric_resized = align_depth_to_vggt(depth_metric, primary.width, primary.height)

    logger.info(f"Depth metric after resize: {depth_metric_resized.shape}, VGGT: {vggt_depth.shape}. In theory, these should match")

    # Anchor VGGT's up-to-scale depth to DepthPro's metric depth over the background. Median of the per-pixel ratios is a better estimate of the single multiplicative factor than the ratio of medians
    bg_ratio = depth_metric_resized[bg_mask] / np.clip(vggt_depth[bg_mask], 1e-6, None)
    bg_ratio = bg_ratio[np.isfinite(bg_ratio) & (bg_ratio > 0)]
    if bg_ratio.size < 200:
        logger.warning(f"Only {bg_ratio.size} valid background px for scale anchoring -> scale may be unreliable")
    scale = float(np.median(bg_ratio))
    print(f"VGGT scale factor from DepthPro: {scale:.4f}")

    world_points_metric = world_points * scale

    food_points_metric = world_points_metric[food_mask_resized > 0]
    volume_blob_cm3 = compute_volume_from_mesh(food_points_metric)
    print(volume_blob_cm3)

    print("-----------------------------------------")

    # per-instance beats the single blob, and use the NMS-refined masks (we already computed them above, no point double-counting the raw overlaps)
    volume_instance_cm3 = compute_volume_per_instance(instance_masks_refined, world_points_metric)
    print(volume_instance_cm3)

    volume_cm3 = volume_instance_cm3
    response = {
        "volume_cm3": round(volume_cm3, 2),
        "volume_instance_cm3": round(volume_instance_cm3, 2),
        "volume_blob_cm3": round(volume_blob_cm3, 2),
        "scale": round(scale, 4),
        "semantic_fusion": None,
    }

    # optional: fuse with the NLP portion prior if one was passed. inert until the NLP side actually sends context
    sem = parse_semantic_context(context)
    if sem is not None:
        fused = fuse_lognormal(
            volume_cm3, SEMANTIC_CV_FLOOR,
            float(sem["total_prior_cm3"]), float(sem.get("total_prior_cv", 0.3)),
        )
        response["volume_cm3"] = fused["volume_cm3"]
        response["semantic_fusion"] = {
            "prior_cm3": float(sem["total_prior_cm3"]),
            "fused_cm3": fused["volume_cm3"],
            "fused_cv": fused["cv"],
        }
        logger.info(f"Fused geometric {volume_cm3:.1f} with prior {sem['total_prior_cm3']:.1f} -> {fused['volume_cm3']:.1f} cm^3")


    logger.info(response)