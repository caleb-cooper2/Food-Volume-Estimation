"""
ConvNeXt-Tiny volume regression trainer on Nutrition5k dataset

Using the following tutorials:
- https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- https://medium.com/exemplifyml-ai/image-classification-with-resnet-convnext-using-pytorch-f051d0d7e098

Phase 1 (epochs 1–warmup_epochs): Freeze backbone, train head only.
- using ImageNet pretraining as a fixed feature extractor
Phase 2 (epochs warmup_epochs+1–total): Unfreeze backbone with 10x lower LR than head.
- prevents the newly initialised head from corrupting the pretrained features
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torchvision.models as models

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


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
            augment_tilt: bool = True
    ):
        self.root = Path(data_root)
        self.transform = transform
        self.augment_tilt = augment_tilt
        self.volume_lookup = volume_lookup

        self.samples = [
            d for d in dish_ids
            if (self.root / "imagery/realsense_overhead" / d / "rgb.png").exists()
               and d in self.volume_lookup
        ]
        logger.info(f"Dataset: {len(self.samples)}/{len(dish_ids)} dishes (on disk + labelled)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        dish_id = self.samples[idx]
        dish_dir = self.root / "imagery/realsense_overhead" / dish_id

        rgb = np.array(Image.open(dish_dir / "rgb.png").convert("RGB"))
        volume_cm3 = self.volume_lookup[dish_id]

        rgb_tensor = self.transform(Image.fromarray(rgb))

        return {
            "rgb": rgb_tensor,
            "volume_cm3": torch.tensor(volume_cm3, dtype=torch.float32),
            "dish_id": dish_id
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
    Augmentation for training: flip, color jitter.
    Deterministic transforms for val/test (resize + normalise only).
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # ImageNet mean
            std=[0.229, 0.224, 0.225] # ImageNet std
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

    return train_transform, val_transform


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
    all_preds  = []
    all_targets = []
    t_start    = time.perf_counter()

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

    all_preds   = torch.cat(all_preds)
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

    all_preds   = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    return {
        "loss": total_loss / len(loader),
        "mape": get_mean_absolute_percentage_error(all_preds, all_targets),
        "mae": get_mean_absolute_error(all_preds, all_targets)
    }

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

    full_dataset = Nutrition5KDataset(
        data_root = args.data_root,
        dish_ids = dish_ids,
        volume_lookup = volume_lookup,
        transform = train_transform,
        augment_tilt = True
    )

    # Split: 80% train, 15% val, 5% test
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * 0.15))
    n_test = max(1, int(n_total * 0.05))
    n_train = n_total - n_val - n_test

    train_set, val_set, test_set = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    # Use deterministic transforms for val/test (no augmentation)
    val_set.dataset = Nutrition5KDataset(
        data_root = args.data_root,
        dish_ids = dish_ids,
        volume_lookup = volume_lookup,
        transform = val_transform,
        augment_tilt = False
    )
    test_set.dataset = val_set.dataset

    logger.info(f"Split: train={n_train} | val={n_val} | test={n_test}")

    # --- DataLoaders ---
    train_loader = DataLoader(
        train_set,
        batch_size = args.batch,
        shuffle = True,
        num_workers = args.workers,
        pin_memory = device.type != "mps" # pin_memory unsupported on MPS
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
    optimiser   = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=1e-4)
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