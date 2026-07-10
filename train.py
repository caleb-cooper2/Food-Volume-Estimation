"""
ConvNeXt-Tiny volume/mass regression trainer on Nutrition5k dataset

Using the following tutorials as a starting point:
- https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- https://medium.com/exemplifyml-ai/image-classification-with-resnet-convnext-using-pytorch-f051d0d7e098

Phase 1 (epochs 1–warmup_epochs): Freeze backbone, train head only.
- using ImageNet pretraining as a fixed feature extractor
Phase 2 (epochs warmup_epochs+1–total): Unfreeze backbone with 10x lower LR than head.
- prevents the newly initialised head from corrupting the pretrained features
"""

import argparse
import logging
import random
import re
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SIDE_ANGLE_PATTERN = re.compile(r"^camera_([ABCD])frame(\d{3})\.jpeg$")


def apply_random_tilt(rgb: np.ndarray) -> np.ndarray:
    """
    Simulate a more realistic handheld photo by rotating the virtual camera forward about the X-axis (tilting toward the far edge of the plate).

    Each corner of the source image is projected into 3D space, rotated, then projected back to 2D via perspective division.
    The resulting 4-point correspondence is used to compute the warp homography.
    """
    tilt_deg = float(np.random.choice([0, 15, 25, 35, 45], p=[0.1, 0.2, 0.3, 0.3, 0.1]))
    if tilt_deg == 0:
        return rgb

    height, width = rgb.shape[:2]

    # Project each source corner through 3D rotation and back to 2D
    source_corners = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height],
    ], dtype=np.float32)

    principal_x = width / 2.0
    principal_y = height / 2.0
    focal_px = float(width)
    camera_z = focal_px # virtual camera sits one focal length above the plate

    cos_t = np.cos(np.radians(tilt_deg))
    sin_t = np.sin(np.radians(tilt_deg))

    destination_corners = []
    for x, y in source_corners:
        # Shift origin to image centre before rotating
        x_origin = x - principal_x
        y_origin = y - principal_y

        # X-axis rotation: far edge (positive Y) rotates away from camera
        x_rotation = x_origin
        y_rotation = y_origin * cos_t
        z_rotation = y_origin * sin_t

        # Perspective projection back to pixel space
        x_projection = focal_px * (x_rotation / (camera_z - z_rotation)) + principal_x
        y_projection = focal_px * (y_rotation / (camera_z - z_rotation)) + principal_y
        destination_corners.append([x_projection, y_projection])

    destination_corners = np.array(destination_corners, dtype=np.float32)
    homography = cv2.getPerspectiveTransform(source_corners, destination_corners)

    # Fill exposed border pixels with the mean edge colour of the original image so the model doesn't learn fill colour as a tilt cue
    border_pixels = np.concatenate([
        rgb[:10, :].reshape(-1, 3),
        rgb[-10:, :].reshape(-1, 3),
        rgb[:, :10].reshape(-1, 3),
        rgb[:, -10:].reshape(-1, 3),
    ])
    border_fill = tuple(int(mean) for mean in border_pixels.mean(axis=0))

    return cv2.warpPerspective(
        rgb, homography, (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_fill
    )


def is_valid_file(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0

class Nutrition5KDataset(Dataset):
    """
    Loads RGB images and ground-truth targets from Nutrition5K dataset.
    """

    def __init__(
            self,
            data_root: str,
            dish_ids: list[str],
            target_lookup: dict[str, float],
            transform,
            mode: str,
            spatial_aug_overhead: A.Compose | None = None,
            spatial_aug_side: A.Compose | None = None,
            photometric_aug: A.Compose | None = None,
            augment_tilt: bool = True,
            volume_lookup: dict[str, float] | None = None,
    ):
        self.root = Path(data_root)
        self.transform = transform
        self.mode = mode
        self.augment_tilt = augment_tilt and mode == "train"
        self.target_lookup = target_lookup
        self.volume_lookup = volume_lookup or {}  # dish-level geometric scalar, only used with --use_volume
        self.spatial_aug_overhead = spatial_aug_overhead
        self.spatial_aug_side = spatial_aug_side
        self.photometric_aug = photometric_aug

        self.views: dict[str, dict] = {}
        n_with_side = 0
        n_overhead_only = 0
        n_side_only = 0

        for dish_id in dish_ids:
            if dish_id not in self.target_lookup:
                logger.warning(f"Skipping dish {dish_id} as it's not in the target lookup")
                continue
            manifest = self.discover_views(dish_id)
            if manifest["overhead"] is None and not manifest["side_by_camera"]:
                logger.warning(f"Skipping dish {dish_id} as it has no usable views")
                continue # no usable image data, skip
            self.views[dish_id] = manifest
            if manifest["overhead"] is None:
                n_side_only += 1
            elif manifest["side_by_camera"]:
                n_with_side += 1
            else:
                n_overhead_only += 1

        self.samples = list(self.views.keys())
        logger.info(
            f"Dataset[{mode}]: {len(self.samples)}/{len(dish_ids)} dishes usable | "
            f"overhead+side={n_with_side} | overhead_only={n_overhead_only} | side_only={n_side_only}"
        )

    def __len__(self):
        return len(self.samples)

    def discover_views(self, dish_id: str) -> dict:
        overhead_path = self.root / "imagery/realsense_overhead" / dish_id / "rgb.png"
        overhead = overhead_path if is_valid_file(overhead_path) else None

        side_dir = self.root / "imagery/side_angles" / dish_id
        side_by_camera: dict[str, list[Path]] = {}
        if side_dir.is_dir():
            for candidate in side_dir.glob("camera_*frame*.jpeg"):
                match = SIDE_ANGLE_PATTERN.match(candidate.name)
                if match is None or not is_valid_file(candidate):
                    logger.warning(f"Skipping invalid side angle candidate: {candidate}. no match or not valid file")
                    continue
                camera = match.group(1)
                side_by_camera.setdefault(camera, []).append(candidate)
        return {"overhead": overhead, "side_by_camera": side_by_camera}

    def select_view(self, dish_id: str) -> tuple[Path, str]:
        manifest = self.views[dish_id]

        if self.mode != "train":
            # Deterministic pick so val/test metrics are stable and comparable across epochs
            if manifest["overhead"] is not None:
                return manifest["overhead"], "overhead"
            first_camera = sorted(manifest["side_by_camera"])[0]
            return sorted(manifest["side_by_camera"][first_camera])[0], "side"

        # For training, pick uniformly from available viewpoint "pools"
        pools = []
        if manifest["overhead"] is not None:
            pools.append(("overhead", None))
        for camera, frames in manifest["side_by_camera"].items():
            if frames:
                pools.append(("side", camera))

        view_kind, camera = pools[np.random.randint(len(pools))]
        if view_kind == "overhead":
            return manifest["overhead"], "overhead"

        frames = manifest["side_by_camera"][camera]
        return frames[np.random.randint(len(frames))], "side"


    def __getitem__(self, idx: int) -> dict:
        dish_id = self.samples[idx]
        image_path, view_type = self.select_view(dish_id)

        rgb = np.array(Image.open(image_path).convert("RGB"))

        spatial_aug = self.spatial_aug_overhead if view_type == "overhead" else self.spatial_aug_side
        if spatial_aug is not None:
            rgb = spatial_aug(image=rgb)["image"]

        if self.photometric_aug is not None:
            rgb = self.photometric_aug(image=rgb)["image"]

        if self.augment_tilt and view_type == "overhead":
            rgb = apply_random_tilt(rgb)

        target = self.target_lookup[dish_id]
        rgb_tensor = self.transform(Image.fromarray(rgb))

        return {
            "rgb": rgb_tensor,
            "target": torch.tensor(target, dtype=torch.float32),
            "volume_cm3": torch.tensor(self.volume_lookup.get(dish_id, 1.0), dtype=torch.float32), # dish-level scalar, view-independent. 1.0 sentinel when no cache -> log() = 0, and it's only ever read under --use_volume anyway
            "dish_id": dish_id,
            "view_type": view_type
        }


class VolumeAssistedRegressor(nn.Module):
    """
    Nutrition5k's volume-assisted head (Thames et al., CVPR 2021): ConvNeXt features from the RGB image are concatenated with
    a single geometry-derived volume scalar (log cm^3) before the final FC. The scalar injects the metric scale the RGB image
    can't carry on its own, which is the exact trick that took the paper's mass error from 18.7% to 13.7%

    Mirrors torchvision's ConvNeXt layout (features -> avgpool -> LayerNorm2d -> flatten) so the ImageNet pretraining,
    the freeze/unfreeze phasing and the log-space read-out all still hold
    """
    def __init__(self, freeze_backbone: bool = False, log_target: bool = True):
        super().__init__()
        backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.norm = backbone.classifier[0] # LayerNorm2d(768), runs after avgpool
        self.log_target = log_target

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        head_layers = [
            nn.Linear(768 + 1, 256),  # +1 for the volume scalar
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        ]
        if not log_target:
            head_layers.append(nn.Softplus())
        self.head = nn.Sequential(*head_layers)

    def forward(self, rgb: torch.Tensor, volume_cm3: torch.Tensor) -> torch.Tensor:
        feats = torch.flatten(self.norm(self.avgpool(self.features(rgb))), 1)  # (B, 768)
        v = torch.log(volume_cm3.clamp(min=1.0)).unsqueeze(1)                  # (B, 1), log cm^3
        return self.head(torch.cat([feats, v], dim=1)).squeeze(1)      # (B,)


def build_model(
        freeze_backbone: bool = False,
        log_target: bool = True,
        head_bias_init: float | None = None,
        use_volume: bool = False,
) -> nn.Module:
    """
    Load pretrained ConvNeXt-Tiny and replace classification head with regression head.
    If freeze_backbone=True, keep backbone frozen and only enable classifier gradients.

    In log-space mode the head outputs an unbounded real value (interpreted as
    log target); positivity is guaranteed by exp() at read-out, so Softplus is
    dropped. In raw mode Softplus is kept.

    head_bias_init warm-starts the final bias to the mean target (mean log-target
    in log mode). Zeroing the final weight makes the initial prediction exactly
    that mean, which removes the ~12,000 opening loss and the slow first epochs.

    use_volume swaps in VolumeAssistedRegressor, which concatenates a geometry-derived volume
    scalar into the head (Nutrition5k's volume-assisted trick: 18.7% -> 13.7% mass).
    """
    if use_volume:
        model = VolumeAssistedRegressor(freeze_backbone=freeze_backbone, log_target=log_target)
        final_linear = model.head[3]
    else:
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        head_layers = [
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        ]
        if not log_target:
            head_layers.append(nn.Softplus())
        model.classifier[2] = nn.Sequential(*head_layers)
        final_linear = model.classifier[2][3]

        for param in model.classifier.parameters():
            param.requires_grad = True

    # Final Linear is index 3 in both cases (Softplus, when present, sits after it)
    if head_bias_init is not None:
        nn.init.zeros_(final_linear.weight)
        final_linear.bias.data.fill_(float(head_bias_init))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model: {'VolumeAssisted ' if use_volume else ''}ConvNeXt-Tiny | "
        f"total={total_params:,} | trainable={trainable_params:,}"
    )

    return model


def split_backbone_head_params(model: nn.Module) -> tuple[list, list]:
    """
    (backbone, head) param split that works for both the plain ConvNeXt and the VolumeAssistedRegressor. Backbone = the
    pretrained feature extractor (gets the 10x lower LR) while head = everything else (norm/pool/regression head). Both expose .features, so key off that
    instead of .classifier which the wrapper doesn't have
    """
    backbone = list(model.features.parameters())
    backbone_ids = {id(p) for p in backbone}
    head = [p for p in model.parameters() if id(p) not in backbone_ids]
    return backbone, head


def get_transforms(img_size: int = 224):
    """
    Val/test: deterministic resize + normalise only.
    Train: spatial and photometric augmentation is applied in __getitem__
    so it can operate on both RGB and depth simultaneously.
    """
    train_rgb_finalise = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_rgb_finalise, val_transform

def build_train_augmentation(img_size: int = 224) -> tuple[A.Compose, A.Compose, A.Compose]:
    """
    Spatial augmentation: applied to overhead and side views, side view doesn't get vertical flip
    Photometric augmentation: applied to both overhead and side views
    """
    spatial_overhead = A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.6, 1.0), ratio=(0.9, 1.1), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Affine(
            scale=(0.35, 0.85),
            border_mode=cv2.BORDER_REPLICATE,
            p=0.5,
        )
    ])

    spatial_side = A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.6, 1.0), ratio=(0.9, 1.1), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.35, 0.85),
            border_mode=cv2.BORDER_REPLICATE,
            p=0.5,
        )
    ])

    photometric = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(std_range=(5.0 / 255.0, 255.0 / 255.0) , p=0.2)
    ])

    return spatial_overhead, spatial_side, photometric


def get_mean_absolute_percentage_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """MAPE with a 1-unit epsilon to avoid division by very small targets."""
    eps = 1.0
    mape = torch.mean(torch.abs(pred-target) / torch.clamp(torch.abs(target), min=eps))
    return mape.item() * 100.0


def get_mean_absolute_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """MAE in the target's units (g for mass, cm^3 for volume_density)."""
    return torch.mean(torch.abs(pred-target)).item()


def _to_real(model_out: torch.Tensor, log_target: bool) -> torch.Tensor:
    """Map the model output back to real target units (grams or cm^3)."""
    if log_target:
        return torch.exp(model_out)
    return model_out


def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        loss_fn: nn.Module,
        optimiser: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        log_target: bool,
        use_volume: bool,
        scheduler=None
) -> dict:
    """Train for one epoch. Accumulate predictions and targets for epoch-level metrics."""
    model.train()

    total_loss = 0.0
    all_preds = []
    all_targets = []
    t_start = time.perf_counter()

    for batch_idx, batch in enumerate(loader):
        rgb = batch["rgb"].to(device)
        target = batch["target"].to(device)
        volume = batch["volume_cm3"].to(device) if use_volume else None

        out = model(rgb, volume) if use_volume else model(rgb).squeeze(1)
        target_model = torch.log(target) if log_target else target # match the space the loss is computed in
        loss = loss_fn(out, target_model)

        #backwards pass
        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        all_preds.append(_to_real(out, log_target).detach().cpu()) # metrics always in real units
        all_targets.append(target.detach().cpu())

        if (batch_idx + 1) % 20 == 0:
            logger.info(
                f"  Epoch {epoch} | batch {batch_idx+1}/{len(loader)} | "
                f"loss={loss.item():.4f}"
            )

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    return {
        "loss": total_loss / len(loader),
        "mape": get_mean_absolute_percentage_error(all_preds, all_targets),
        "mae": get_mean_absolute_error(all_preds, all_targets),
        "time": time.perf_counter() - t_start
    }

@torch.no_grad()
def validate(
        model: nn.Module,
        loader: DataLoader,
        loss_fn: nn.Module,
        device: torch.device,
        log_target: bool,
        use_volume: bool = False,
        tta: bool = False,
) -> dict:
    """
    Evaluate on validation/test set without gradient updates.
    """
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in loader:
        rgb = batch["rgb"].to(device)
        target = batch["target"].to(device)
        volume = batch["volume_cm3"].to(device) if use_volume else None

        out = model(rgb, volume) if use_volume else model(rgb).squeeze(1)
        target_model = torch.log(target) if log_target else target
        total_loss += loss_fn(out, target_model).item()

        pred_real = _to_real(out, log_target)
        if tta:
            # horizontal-flip TTA: average the prediction over the image and its mirror
            rgb_flip = torch.flip(rgb, dims=[3])
            out_flip = model(rgb_flip, volume) if use_volume else model(rgb_flip).squeeze(1)
            pred_real = 0.5 * (pred_real + _to_real(out_flip, log_target))

        all_preds.append(pred_real.cpu())
        all_targets.append(target.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    return {
        "loss": total_loss / len(loader),
        "mape": get_mean_absolute_percentage_error(all_preds, all_targets),
        "mae": get_mean_absolute_error(all_preds, all_targets)
    }


def worker_init_fn(worker_id: int) -> None:
    "Reseed numpy per DataLoader worker to ensure the random view/frame selection is actually independent"
    worker_seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(worker_seed)


def build_target_lookup(args, dish_ids, mass_g) -> tuple[dict[str, float], str]:
    """Return (dish_id -> target, unit_label) according to --target."""
    mass_min_g, mass_max_g = 20.0, 1500.0

    if args.target == "mass":
        values = mass_g
        unit = "g"
    else:  # volume_density: convert mass to a volume proxy via average food density
        values = mass_g / args.density
        unit = "cm3"

    lookup = {
        d: float(v)
        for d, m, v in zip(dish_ids, mass_g, values)
        if pd.notna(m) and mass_min_g <= m <= mass_max_g
    }
    n_filtered = len(dish_ids) - len(lookup)
    logger.info(
        f"Target={args.target} [{unit}]: {len(lookup)} dishes after filtering "
        f"({n_filtered} removed outside {mass_min_g}–{mass_max_g}g range)"
    )
    return lookup, unit


def load_volume_cache(path: str) -> dict[str, float]:
    """dish_id -> geometric volume scalar (cm^3), from cache_volume_scalars.py output"""
    df = pd.read_csv(path)
    lookup = {
        str(r.dish_id): float(r.volume_cm3)
        for r in df.itertuples(index=False)
        if pd.notna(r.volume_cm3) and float(r.volume_cm3) > 0
    }
    logger.info(f"Volume cache: {len(lookup)} dishes with a usable scalar from {path}")
    return lookup


def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # Load metadata and extract ground-truth targets
    meta = pd.read_csv(
        args.metadata,
        header=None,
        engine="python",
        on_bad_lines="skip",
    )

    # Col layout: dish_id, calories, total_mass_g, fat_g, carb_g, protein_g, ingredients...
    dish_ids = meta.iloc[:, 0].astype(str).tolist()
    mass_g = pd.to_numeric(meta.iloc[:, 2], errors="coerce")

    logger.info(f"Mass col raw: min={mass_g.min():.1f}  mean={mass_g.mean():.1f}  max={mass_g.max():.1f}")

    target_lookup, unit = build_target_lookup(args, dish_ids, mass_g)

    values = list(target_lookup.values())
    logger.info(
        f"Target stats [{unit}]: min={min(values):.1f}  mean={sum(values)/len(values):.1f}  max={max(values):.1f}"
    )

    train_transform, val_transform = get_transforms(img_size=args.img_size)
    spatial_overhead_aug, spatial_side_aug, photometric_aug = build_train_augmentation(img_size=args.img_size)

    valid_dish_ids = [d for d in dish_ids if d in target_lookup]

    # --use_volume: restrict to dishes that also have a cached geometric scalar. This shrinks the set to the overhead/depth subset,
    # matching how the paper's depth experiments were run
    volume_lookup = None
    if args.use_volume:
        volume_lookup = load_volume_cache(args.volume_cache)
        before = len(valid_dish_ids)
        valid_dish_ids = [d for d in valid_dish_ids if d in volume_lookup]
        logger.info(
            f"--use_volume: {len(valid_dish_ids)}/{before} dishes have both a mass target "
            f"and a cached volume scalar"
        )

    rng = random.Random(42)
    rng.shuffle(valid_dish_ids)

    n_total = len(valid_dish_ids)
    n_val = max(1, int(n_total * 0.15))
    n_test = max(1, int(n_total * 0.05))
    n_train = n_total - n_val - n_test

    train_ids = valid_dish_ids[:n_train]
    val_ids = valid_dish_ids[n_train:n_train + n_val]
    test_ids = valid_dish_ids[n_train + n_val:]

    logger.info(f"Split (by dish): train={len(train_ids)} | val={len(val_ids)} | test={len(test_ids)}")

    # Warm-start bias: mean log-target (log mode) or mean target (raw mode)
    train_values = np.array([target_lookup[d] for d in train_ids], dtype=np.float64)
    head_bias_init = float(np.mean(np.log(train_values))) if args.log_target else float(np.mean(train_values))
    logger.info(f"Head bias init = {head_bias_init:.4f} ({'log-space' if args.log_target else 'raw'})")

    train_set = Nutrition5KDataset(
        data_root=args.data_root,
        dish_ids=train_ids,
        target_lookup=target_lookup,
        transform=train_transform,
        mode="train",
        spatial_aug_overhead=spatial_overhead_aug,
        spatial_aug_side=spatial_side_aug,
        photometric_aug=photometric_aug,
        augment_tilt=True,
        volume_lookup=volume_lookup,
    )
    val_set = Nutrition5KDataset(
        data_root=args.data_root,
        dish_ids=val_ids,
        target_lookup=target_lookup,
        transform=val_transform,
        mode="val",
        volume_lookup=volume_lookup,
    )
    test_set = Nutrition5KDataset(
        data_root=args.data_root,
        dish_ids=test_ids,
        target_lookup=target_lookup,
        transform=val_transform,
        mode="test",
        volume_lookup=volume_lookup,
    )


    # --- DataLoaders ---
    train_loader = DataLoader(
        train_set,
        batch_size = args.batch,
        shuffle = True,
        num_workers = args.workers,
        pin_memory = device.type != "mps", # pin_memory unsupported on MPS
        worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_set,
        batch_size = args.batch * 2,
        shuffle = False,
        num_workers = args.workers,
        pin_memory = device.type != "mps"
    )

    # --- Phase 1: Only training head ---
    model = build_model(
        freeze_backbone=True,
        log_target=args.log_target,
        head_bias_init=head_bias_init,
        use_volume=args.use_volume,
    ).to(device)

    # log-space MSE optimises relative error and is robust to the skewed target.
    loss_fn = nn.MSELoss() if args.log_target else nn.HuberLoss(delta=50.0)

    _, head_params = split_backbone_head_params(model)
    optimiser = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs * len(train_loader))

    best_val_mape = float("inf")
    best_ckpt = Path(args.output) / "best_model.pt"
    best_ckpt.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Transition to Phase 2: unfreeze backbone with differential LR
        if epoch == args.warmup_epochs + 1:
            logger.info(f"--- Phase 2: unfreezing backbone at epoch {epoch} ---")
            for param in model.parameters():
                param.requires_grad = True

            backbone_params, head_params = split_backbone_head_params(model)
            optimiser = torch.optim.AdamW([
                {"params": backbone_params, "lr": args.lr * 0.1},
                {"params": head_params, "lr": args.lr},
            ], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser,
                T_max=(args.epochs - args.warmup_epochs) * len(train_loader),
            )

        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimiser, device, epoch, args.log_target, args.use_volume, scheduler,
        )
        val_metrics = validate(model, val_loader, loss_fn, device, args.log_target, use_volume=args.use_volume, tta=args.tta)

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss={train_metrics['loss']:.4f}  mape={train_metrics['mape']:.1f}%  mae={train_metrics['mae']:.1f}{unit} | "
            f"val   loss={val_metrics['loss']:.4f}  mape={val_metrics['mape']:.1f}%  mae={val_metrics['mae']:.1f}{unit} | "
            f"lr={optimiser.param_groups[-1]['lr']:.2e} | "
            f"time={train_metrics['time']:.1f}s"
        )

        # Checkpoint on best validation MAPE
        if val_metrics["mape"] < best_val_mape:
            best_val_mape = val_metrics["mape"]
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_mape": best_val_mape,
                "val_mae": val_metrics["mae"],
                "args": vars(args),
                "unit": unit,
            }, best_ckpt)
            logger.info(f"  ✓ New best val MAPE={best_val_mape:.1f}% - saved to {best_ckpt}")

    logger.info("--- Test set evaluation ---")
    test_loader = DataLoader(
        test_set,
        batch_size = args.batch * 2,
        shuffle = False,
        num_workers = args.workers
    )
    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics = validate(model, test_loader, loss_fn, device, args.log_target, use_volume=args.use_volume, tta=args.tta)
    logger.info(f"Test | mape={test_metrics['mape']:.1f}%  mae={test_metrics['mae']:.1f}{unit}  loss={test_metrics['loss']:.4f}")
    logger.info(f"Best checkpoint: {best_ckpt}")

if __name__ == "__main__":
    #/uc/usersc/cco139/Home/Downloads/datasets/gillesokhin/nutrition5k-dataset
    # /uc/usersc/cco139/Home/Downloads/datasets/gillesokhin/nutrition5k-dataset/versions/6
    parser = argparse.ArgumentParser(description="ConvNeXt-Tiny mass/volume regression - Nutrition5K")
    parser.add_argument("--data_root", type=str, default="./data/nutrition5k_dataset")
    parser.add_argument("--metadata", type=str, default="./data/dish_metadata_cafe1.csv")
    parser.add_argument("--output", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--target", choices=["mass", "volume_density"], default="mass",
                        help="Regression target. mass is honest and matches Nutrition5k baselines.")
    parser.add_argument("--density", type=float, default=0.8,
                        help="Only used by --target volume_density (mass/density).")
    parser.add_argument("--log_target", action=argparse.BooleanOptionalAction, default=True,
                        help="Regress in log space (recommended for the skewed target).")
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True,
                        help="Horizontal-flip test-time augmentation at val/test.")
    parser.add_argument("--use_volume", action=argparse.BooleanOptionalAction, default=False) # Volume-assisted regression: concat a cached geometric volume scalar into the head (Nutrition5k, Thames et al. 2021)
    parser.add_argument("--volume_cache", type=str, default="./data/volume_scalars.csv") # needed if we do --use_volume

    args = parser.parse_args()
    main(args)