import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

image_names = ["data/camera_Aframe001.jpeg", "data/camera_Bframe027.jpeg", "data/camera_Aframe029.jpeg"]
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.amp.autocast('cuda', dtype=dtype):
        predictions = model(images)

world_points = predictions["world_points"][0, 0].cpu().numpy() # (H, W, 3)
conf = predictions["world_points_conf"][0, 0].cpu().numpy() # (H, W)

print(f"world_points shape: {world_points.shape}")
print(f"X range: {world_points[..., 0].min():.3f} to {world_points[..., 0].max():.3f}")
print(f"Y range: {world_points[..., 1].min():.3f} to {world_points[..., 1].max():.3f}")
print(f"Z range: {world_points[..., 2].min():.3f} to {world_points[..., 2].max():.3f}")
