"""
Train the CLIP size-prior head: frozen CLIP image features -> log real-world food footprint (cm)
A reference-free metric-scale anchor for the geometry/reconstruction routes. Only the small MLP head trains while the
CLIP trunk stays frozen
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor

from logging_config import get_logger
from scale_prior import SizePriorHead, CLIP_MODEL_NAME

logger = get_logger(__name__)


def load_footprint_cache(path: str) -> dict[str, float]:
    """dish_id -> footprint diameter (cm), from cache_size_prior.py output"""
    df = pd.read_csv(path)
    lookup = {
        str(r.dish_id): float(r.footprint_cm)
        for r in df.itertuples(index=False)
        if pd.notna(r.footprint_cm) and float(r.footprint_cm) > 0
    }
    logger.info(f"Footprint cache: {len(lookup)} dishes with a usable size from {path}")
    return lookup


class OverheadFootprintDataset(Dataset):
    """Overhead RGB -> CLIP pixel_values + log-cm target. CLIP preprocessing is deterministic (no augmentation)"""

    def __init__(self, data_root: str, dish_ids: list[str], footprint_lookup: dict[str, float], clip_processor):
        self.root = Path(data_root)
        self.clip_processor = clip_processor
        self.footprint_lookup = footprint_lookup
        self.samples = [dish for dish in dish_ids if dish in footprint_lookup]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        dish_id = self.samples[idx]
        rgb_path = self.root / "imagery/realsense_overhead" / dish_id / "rgb.png"
        image = Image.open(rgb_path).convert("RGB")
        pixel_values = self.clip_processor(images=image, return_tensors="pt")["pixel_values"][0]
        return {
            "pixel_values": pixel_values,
            "log_target": torch.tensor(np.log(self.footprint_lookup[dish_id]), dtype=torch.float32)
        }


def size_mape(pred_cm: torch.Tensor, target_cm: torch.Tensor) -> float:
    """MAPE on the real cm size, so the number is directly interpretable as scale error"""
    return (torch.mean(torch.abs(pred_cm - target_cm) / torch.clamp(target_cm, min=1.0)) * 100.0).item()


@torch.no_grad()
def clip_features(pixel_values: torch.Tensor, clip_model) -> torch.Tensor:
    features = clip_model.get_image_features(pixel_values=pixel_values)
    image_features = features.pooler_output
    return (image_features / image_features.norm(dim=-1, keepdim=True)).float()


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Device: {device}")

    footprint_lookup = load_footprint_cache(args.footprint_cache)
    dish_ids = list(footprint_lookup.keys())
    random.Random(42).shuffle(dish_ids) # same seed as train.py so the split family is comparable

    n_values = max(1, int(len(dish_ids) * 0.15))
    value_ids, train_ids = dish_ids[:n_values], dish_ids[n_values:]
    logger.info(f"Split: train={len(train_ids)} | val={len(value_ids)}")

    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).eval().to(device)
    for param in clip_model.parameters():
        param.requires_grad = False

    train_loader = DataLoader(
        OverheadFootprintDataset(args.data_root, train_ids, footprint_lookup, clip_processor),
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type != "mps"
    )
    val_loader = DataLoader(
        OverheadFootprintDataset(args.data_root, value_ids, footprint_lookup, clip_processor),
        batch_size=args.batch * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type != "mps"
    )

    head = SizePriorHead().to(device)
    # Warm-start the read-out bias at the mean log-size so training starts near the prior rather than at zero?
    mean_log_cm = float(np.mean([np.log(footprint_lookup[d]) for d in train_ids]))
    head.head[3].bias.data.fill_(mean_log_cm)
    logger.info(f"Head bias init = {mean_log_cm:.4f} (log cm, mean footprint {np.exp(mean_log_cm):.1f}cm)")

    loss_fn = nn.HuberLoss(delta=0.15) # log-space Huber: ~16% relative-error transition, matching what we have in train.py
    optimiser = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs * len(train_loader))

    best_val_mape = float("inf")
    out_path = Path(args.output) / "size_prior.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        head.train()
        t_start = time.perf_counter()
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            log_target = batch["log_target"].to(device)

            pred_log_cm = head(clip_features(pixel_values, clip_model))
            loss = loss_fn(pred_log_cm, log_target)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            scheduler.step()

        head.eval()
        predictions, targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                pred_log_cm = head(clip_features(batch["pixel_values"].to(device), clip_model))
                predictions.append(torch.exp(pred_log_cm).cpu())
                targets.append(torch.exp(batch["log_target"]))
        val_mape = size_mape(torch.cat(predictions), torch.cat(targets))
        logger.info(f"Epoch {epoch:3d}/{args.epochs} | val size MAPE={val_mape:.1f}% | time={time.perf_counter()-t_start:.1f}s")

        if val_mape < best_val_mape:
            best_val_mape = val_mape
            torch.save({
                "state_dict": head.state_dict(),
                "feature_dim": head.head[0].in_features,
                "val_mape": best_val_mape,
                "args": vars(args),
            }, out_path)
            logger.info(f"  ✓ New best val size MAPE={best_val_mape:.1f}% - saved to {out_path}")

    logger.info(f"Done. Best val size MAPE={best_val_mape:.1f}%. Head -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CLIP size-prior head on cached food footprints")
    parser.add_argument("--data_root", type=str, default="./data/nutrition5k_dataset")
    parser.add_argument("--footprint_cache", type=str, default="./data/footprint_sizes.csv")
    parser.add_argument("--output", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3) # Head-only, so a higher LR than the backbone fine-tune is fine
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    main(args)
