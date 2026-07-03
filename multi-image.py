import io
import logging
import os
import tempfile

import cv2
import numpy as np
import open3d as o3d
import torch
import trimesh
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
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



@app.on_event("startup")
def startup_event():
    preload_all()


@app.post("/api/v1/estimate-volume-multiview")
async def volume_estimation_multiview(files: list[UploadFile] = File(...)):
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
    depth_metric_resized = cv2.resize(
        depth_metric,
        (vggt_depth.shape[1], vggt_depth.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    scale = float(np.median(depth_metric_resized[bg_mask]) / np.median(vggt_depth[bg_mask]))
    print(f"VGGT scale factor from DepthPro: {scale:.4f}")

    world_points_metric = world_points * scale

    food_points_metric = world_points_metric[food_mask_resized > 0]
    volume_cm3 = compute_volume_from_mesh(food_points_metric)
    print(volume_cm3)

    print("-----------------------------------------")

    volume_cm3 = compute_volume_per_instance(instance_masks, world_points_metric)
    print(volume_cm3)