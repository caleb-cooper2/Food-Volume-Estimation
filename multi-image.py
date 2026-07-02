import io
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
from model_manage import register_loader, release_model, get_model

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

    # Which can be converted to a trimesh and volume can be derived
    trimesh_result = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles)
    )
    volume_m3 = abs(trimesh_result.volume) # it is in m3 as that is what depth map + vggt provides
    return volume_m3 * 1_000_000.0  # computer to cm^3


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
    food_mask, mask_scores = segment_food(primary)

    # VGGT reconstruction
    images_tensor = load_and_preprocess_images_from_pil(pil_images)
    vggt_model = get_model("vggt")
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=dtype):
            predictions = vggt_model(images_tensor.to(torch.device(device)))
    release_model("vggt")

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
    print("After scale correction")
    print(f"X range: {world_points_metric[..., 0].min():.3f} to {world_points_metric[..., 0].max():.3f}")
    print(f"Y range: {world_points_metric[..., 1].min():.3f} to {world_points_metric[..., 1].max():.3f}")
    print(f"Z range: {world_points_metric[..., 2].min():.3f} to {world_points_metric[..., 2].max():.3f}")

    food_points_metric = world_points_metric[food_mask_resized > 0]
    volume_cm3 = compute_volume_from_mesh(food_points_metric)

    print(volume_cm3)