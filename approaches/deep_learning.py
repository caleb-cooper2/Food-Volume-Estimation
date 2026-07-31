"""
Approach B - Deep learning

Single RGB image -> trained ConvNeXt regressor -> mass (g). When the checkpoint was trained with --use_volume, the
geometric approach is reused to compute the volume scalar the head expects, mirroring how cache_volume_scalars.py
builds it
"""

import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from fastapi import APIRouter, File, Form, UploadFile
from torchvision import transforms

from approaches import read_upload, validate_participant_code
from depth import estimate_depth, depth_to_relief_channel, RELIEF_MEAN_M, RELIEF_STD_M
from geometry import fit_support_plane, compute_volume, area_based_volume_proxy
from logging_config import get_logger
from model_manage import register_loader, get_model
from schemas import CameraInfo, EstimationResponse
from segmentation import segment_food

logger = get_logger(__name__)

router = APIRouter()

DL_CHECKPOINT = "models/convnext-tiny-scalar-depth.pt"

_rgb_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def inflate_stem_to_4ch(model: nn.Module) -> None:
    """Swap ConvNeXt's stem to 4 channels so a --depth_channel checkpoint loads"""
    old_stem = model.features[0][0]
    new_stem = nn.Conv2d(4, old_stem.out_channels, kernel_size=old_stem.kernel_size, stride=old_stem.stride)
    with torch.no_grad():
        new_stem.weight[:, :3] = old_stem.weight
        new_stem.weight[:, 3:4] = old_stem.weight.mean(dim=1, keepdim=True)
        new_stem.bias.copy_(old_stem.bias)
    model.features[0][0] = new_stem


class VolumeAssistedRegressor(nn.Module):
    """
    Mirror of the training-time model: ConvNeXt features + a log-volume scalar concatenated before the head.
    Got to stay structurally identical to train.py or the state_dict won't load
    """
    def __init__(self, log_target: bool = True, depth_channel: bool = False):
        super().__init__()
        backbone = models.convnext_tiny(weights=None)
        if depth_channel:
            inflate_stem_to_4ch(backbone)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.norm = backbone.classifier[0]           # LayerNorm2d(768)
        self.log_target = log_target
        # Standardisation stats for the log-volume scalar, filled from the train cache and saved in the state_dict so inference normalises
        self.register_buffer("log_volume_mean", torch.zeros(1))
        self.register_buffer("log_volume_std", torch.ones(1))
        head_layers = [
            nn.Linear(768 + 1, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        ]
        if not log_target:
            head_layers.append(nn.Softplus())
        self.head = nn.Sequential(*head_layers)

    def forward(self, rgb: torch.Tensor, volume_cm3: torch.Tensor) -> torch.Tensor:
        feats = torch.flatten(self.norm(self.avgpool(self.features(rgb))), 1)
        v = torch.log(volume_cm3.clamp(min=1.0)).unsqueeze(1)
        v = (v - self.log_volume_mean) / self.log_volume_std   # whiten so the head sees the variation, not a constant offset
        return self.head(torch.cat([feats, v], dim=1)).squeeze(1)


def load_model(model_path: str) -> nn.Module:
    """
    Rebuild the trained model to match its checkpoint. Reads the saved args so it picks the right architecture: the plain ConvNeXt head,
    or the volume-assisted head (norm + head, with a 769-wide first Linear because the geometric volume scalar is concatenated in)
    """
    checkpoint = torch.load(model_path, map_location="cpu")
    saved_args = checkpoint.get("args", {})
    log_target = bool(saved_args.get("log_target", False))
    use_volume = bool(saved_args.get("use_volume", False))

    depth_channel = bool(saved_args.get("depth_channel", False))

    if use_volume:
        model = VolumeAssistedRegressor(log_target=log_target, depth_channel=depth_channel)
    else:
        head_layers = [
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        ]
        if not log_target:
            head_layers.append(nn.Softplus())
        model = models.convnext_tiny(weights=None)
        if depth_channel:
            inflate_stem_to_4ch(model)
        model.classifier[2] = nn.Sequential(*head_layers)

    model.load_state_dict(checkpoint["state_dict"])
    model._log_target = log_target
    model._use_volume = use_volume
    model._depth_channel = depth_channel
    return model.eval()


register_loader("custom_volume", lambda: load_model(DL_CHECKPOINT))


@torch.no_grad()
def run_custom_model(image: Image.Image) -> float:
    """
    Estimate mass (g) with the custom trained model. Builds whatever extra inputs the checkpoint was trained with...
    a metric relief channel (--depth_channel) and/or a geometric volume scalar (--use_volume)
    """
    tensor = _rgb_transform(image).unsqueeze(0)  # (1, 3, 224, 224)

    model = get_model("custom_volume")
    needs_depth = getattr(model, "_use_volume", False) or getattr(model, "_depth_channel", False)

    depth_map = focal_px = mask = None
    if needs_depth:
        depth_map, focal_px = estimate_depth(image)
        mask, _, _ = segment_food(image)

    if getattr(model, "_depth_channel", False):
        relief = depth_to_relief_channel(depth_map)
        relief_224 = cv2.resize(relief, (224, 224), interpolation=cv2.INTER_NEAREST)
        relief_tensor = (torch.from_numpy(relief_224)[None, None] - RELIEF_MEAN_M) / RELIEF_STD_M
        tensor = torch.cat([tensor, relief_tensor], dim=1)  # (1, 4, 224, 224): RGB + relief

    volume_t = None
    if getattr(model, "_use_volume", False):
        w, h = image.size
        cam = CameraInfo(fx=focal_px, fy=focal_px, cx=w / 2.0, cy=h / 2.0, image_width=w, image_height=h, source="depthpro_fov")
        plane_n, plane_p0, _ = fit_support_plane(depth_map, mask, cam)
        vr = compute_volume(depth_map, mask, cam, plane_n, plane_p0)
        if vr.geometry_confidence < 0.33:
            vol = area_based_volume_proxy(depth_map, mask, cam)
            logger.warning(f"  Custom | geometry confidence {vr.geometry_confidence:.2f} -> area proxy volume {vol:.1f} cm^3")
        else:
            vol = vr.volume_cm3
        volume_t = torch.tensor([vol], dtype=torch.float32)

    # Fetch the model LAST so it lands on the GPU and nothing evicts it before we run
    model = get_model("custom_volume")
    model_device = next(model.parameters()).device
    tensor = tensor.to(model_device)

    if volume_t is not None:
        out = model(tensor, volume_t.to(model_device)).squeeze().item()
    else:
        out = model(tensor).squeeze().item()

    if getattr(model, "_log_target", False):
        out = float(np.exp(out))
    return out


@router.post("/api/v1/estimate-volume-dl", response_model=EstimationResponse)
async def volume_estimation_dl(
        file: UploadFile = File(...),
        participant_code: str = Form(...)
) -> EstimationResponse:
    """
    End-to-end volume estimation pipeline using deep learning model.
    """
    t_start = time.perf_counter()

    participant_code = validate_participant_code(participant_code)
    logger.info(f"[start] participant={participant_code}")

    _, pillow_image = await read_upload(file)

    with torch.no_grad():
        mass = run_custom_model(pillow_image)

    logger.info(f"Model estimated mass: {mass:.2f} g")
    logger.info(f"Total pipeline time: {time.perf_counter() - t_start:.3f}s")
    return EstimationResponse(
        approach="deep-learning",
        volume_cm3=None,
        mass_g=round(mass, 2),
        confidence=None,
        diagnostics={"participant_code": participant_code}
    )
