"""
ConvNeXt-Tiny volume regression trainer on Nutrition5k dataset

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
    Loads RGB images and ground-truth volumes from Nutrition5K dataset.
    """

    def __init__(
            self,
            data_root: str,
            dish_ids: list[str],
            volume_lookup: dict[str, float],
            transform,
            mode: str,
            spatial_aug_overhead: A.Compose | None = None,
            spatial_aug_side: A.Compose | None = None,
            photometric_aug: A.Compose | None = None,
            augment_tilt: bool = True
    ):
        self.root = Path(data_root)
        self.transform = transform
        self.mode = mode
        self.augment_tilt = augment_tilt and mode == "train"
        self.volume_lookup = volume_lookup
        self.spatial_aug_overhead = spatial_aug_overhead
        self.spatial_aug_side = spatial_aug_side
        self.photometric_aug = photometric_aug

        self.views: dict[str, dict] = {}
        n_with_side = 0
        n_overhead_only = 0
        n_side_only = 0

        for dish_id in dish_ids:
            if dish_id not in self.volume_lookup:
                logger.warning(f"Skipping dish {dish_id} as it's not in the volume lookup")
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

        volume_cm3 = self.volume_lookup[dish_id]
        rgb_tensor = self.transform(Image.fromarray(rgb))

        return {
            "rgb": rgb_tensor,
            "volume_cm3": torch.tensor(volume_cm3, dtype=torch.float32),
            "dish_id": dish_id,
            "view_type": view_type
        }


def build_model(freeze_backbone: bool = False) -> nn.Module:
    """
    Load pretrained ConvNeXt-Tiny and replace classification head with regression head.
    If freeze_backbone=True, keep backbone frozen and only enable classifier gradients.
    """
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.classifier[2] = nn.Sequential(
        nn.Linear(768, 256),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Softplus()
    )

    for param in model.classifier.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: ConvNeXt-Tiny | total={total_params:,} | trainable={trainable_params:,}")

    return model


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
    """MAPE with 1 cm^3 epsilon to avoid division by very small targets."""
    eps = 1.0
    mape = torch.mean(torch.abs(pred-target) / torch.clamp(torch.abs(target), min=eps))
    return mape.item() * 100.0


def get_mean_absolute_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    """MAE in cm^3"""
    return torch.mean(torch.abs(pred-target)).item()


def train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        loss_fn: nn.Module,
        optimiser: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
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
        target = batch["volume_cm3"].to(device)

        pred = model(rgb).squeeze(1)
        loss = loss_fn(pred, target)

        #backwards pass
        optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        all_preds.append(pred.detach().cpu())
        all_targets.append(target.detach().cpu())

        if (batch_idx + 1) % 20 == 0:
            logger.info(
                f"  Epoch {epoch} | batch {batch_idx+1}/{len(loader)} | "
                f"loss={loss.item():.2f}"
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
        device: torch.device
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
        target = batch["volume_cm3"].to(device)

        pred = model(rgb).squeeze(1)
        loss = loss_fn(pred, target)

        total_loss += loss.item()
        all_preds.append(pred.cpu())
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


def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Device: {device}")

    # Load metadata and extract ground-truth volumes
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

    # Convert mass to volume via average food density
    mass_min_g = 20.0
    mass_max_g = 1500.0
    density_g_cm3 = 0.8 # g/cm^3, mean food density

    volume_cm3 = mass_g / density_g_cm3

    volume_lookup: dict[str, float] = {
        dish_id: vol
        for dish_id, mass, vol in zip(dish_ids, mass_g, volume_cm3)
        if pd.notna(mass) and mass_min_g <= mass <= mass_max_g
    }

    n_filtered = len(dish_ids) - len(volume_lookup)
    logger.info(
        f"Volume GT: {len(volume_lookup)} dishes after filtering "
        f"({n_filtered} removed outside {mass_min_g}–{mass_max_g}g range)"
    )

    volumes = list(volume_lookup.values())
    logger.info(
        f"Volume GT stats: min={min(volumes):.1f}  mean={sum(volumes)/len(volumes):.1f}  "
        f"max={max(volumes):.1f} cm³"
    )

    train_transform, val_transform = get_transforms(img_size=args.img_size)
    spatial_overhead_aug, spatial_side_aug, photometric_aug = build_train_augmentation(img_size=args.img_size)

    valid_dish_ids = [d for d in dish_ids if d in volume_lookup]
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

    train_set = Nutrition5KDataset(
        data_root=args.data_root,
        dish_ids=train_ids,
        volume_lookup=volume_lookup,
        transform=train_transform,
        mode="train",
        spatial_aug_overhead=spatial_overhead_aug,
        spatial_aug_side=spatial_side_aug,
        photometric_aug=photometric_aug,
        augment_tilt=True
    )
    val_set = Nutrition5KDataset(
        data_root=args.data_root,
        dish_ids=val_ids,
        volume_lookup=volume_lookup,
        transform=val_transform,
        mode="val"
    )
    test_set = Nutrition5KDataset(
        data_root=args.data_root,
        dish_ids=test_ids,
        volume_lookup=volume_lookup,
        transform=val_transform,
        mode="test"
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
    model = build_model(freeze_backbone=True).to(device)
    loss_fn = nn.HuberLoss(delta=50.0)

    head_params = list(model.classifier.parameters())
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

            optimiser = torch.optim.AdamW([
                {"params": model.features.parameters(), "lr": args.lr * 0.1},
                {"params": model.classifier.parameters(), "lr": args.lr},
            ], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser,
                T_max=(args.epochs - args.warmup_epochs) * len(train_loader),
            )

        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimiser, device, epoch, scheduler)
        val_metrics = validate(model, val_loader, loss_fn, device)

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train loss={train_metrics['loss']:.2f}  mape={train_metrics['mape']:.1f}%  mae={train_metrics['mae']:.1f}cm³ | "
            f"val   loss={val_metrics['loss']:.2f}  mape={val_metrics['mape']:.1f}%  mae={val_metrics['mae']:.1f}cm³ | "
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
                "args": vars(args)
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
    test_metrics = validate(model, test_loader, loss_fn, device)
    logger.info(f"Test | mape={test_metrics['mape']:.1f}%  mae={test_metrics['mae']:.1f}cm³  loss={test_metrics['loss']:.2f}")
    logger.info(f"Best checkpoint: {best_ckpt}")

if __name__ == "__main__":
    #/uc/usersc/cco139/Home/Downloads/datasets/gillesokhin/nutrition5k-dataset
    # /uc/usersc/cco139/Home/Downloads/datasets/gillesokhin/nutrition5k-dataset/versions/6
    parser = argparse.ArgumentParser(description="ConvNeXt-Tiny volume regression - Nutrition5K")
    parser.add_argument("--data_root", type=str, default="./data/nutrition5k_dataset")
    parser.add_argument("--metadata", type=str, default="./data/dish_metadata_cafe1.csv")
    parser.add_argument("--output", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()
    main(args)