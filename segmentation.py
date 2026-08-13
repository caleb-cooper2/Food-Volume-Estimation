"""SAM 3 segmentation: the food itself (as a union, or one mask per named item) and the reference utensils"""

from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Processor, Sam3Model

from logging_config import get_logger
from model_manage import register_loader, get_model, torch_device
from schemas import FoodItemResult

logger = get_logger(__name__)

sam3_processor = Sam3Processor.from_pretrained("facebook/sam3")


def _load_sam3_model():
    m = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.bfloat16)
    return m.eval()


register_loader("sam3", _load_sam3_model)


def segment_food(pillow_image: Image.Image, threshold: float = 0.3, prompt="food or drink") -> tuple[np.ndarray, list[float], list[np.ndarray]]:
    """
    Segment food region using SAM 3 with text prompt "food or drink".
    Falls back to full-image mask if no instances found.
    :return: union mask of all detected instances and per-instance scores.
    """
    img_w, img_h = pillow_image.size
    total_pixels = img_w * img_h

    inputs = sam3_processor(images=pillow_image, text=prompt, return_tensors="pt").to(torch_device)

    sam3 = get_model("sam3")
    with torch.no_grad():
        outputs = sam3(**inputs)

    results = sam3_processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]

    instance_masks = results["masks"]
    instance_scores = results["scores"]

    if instance_masks.shape[0] == 0:
        logger.warning("  SAM 3 | No food instances found -> using full-image fallback mask")
        fallback = np.ones((img_h, img_w), dtype=np.uint8)
        return fallback, [], [fallback]

    instance_scores_list = instance_scores.tolist()
    instance_masks_list = [m.cpu().numpy().astype(np.uint8) for m in instance_masks]

    # Log per-instance breakdown before union
    for i, s in enumerate(instance_scores_list):
        inst_px = int(instance_masks[i].sum().item())
        inst_pct = inst_px / total_pixels * 100
        logger.info(f"  SAM 3 | Instance {i}: score={s:.3f}  pixels={inst_px}  coverage={inst_pct:.1f}%")

    union_mask = instance_masks.any(dim=0).cpu().numpy().astype(np.uint8)
    union_px = int(union_mask.sum())
    union_pct = union_px / total_pixels * 100
    bg_pct = 100.0 - union_pct

    logger.info(
        f"  SAM 3 | Union mask: {union_px} px food ({union_pct:.1f}%)  |  "
        f"{total_pixels - union_px} px background ({bg_pct:.1f}%)"
    )

    if union_pct > 80.0:
        logger.warning(f"  SAM 3 | Food mask covers {union_pct:.1f}% of image -> background region too small for reliable plane fitting")
    if union_pct < 2.0:
        logger.warning(f"  SAM 3 | Food mask covers only {union_pct:.1f}% of image -> possible segmentation failure")

    return union_mask, instance_scores_list, instance_masks_list


def segment_food_items(pillow_image: Image.Image, entities: list[dict], threshold: float = 0.5) -> list[FoodItemResult]:
    """Segment one mask per named food, prompting SAM 3 with each entity's noun phrase"""
    items = []
    for entity in entities:
        prompt = entity["text"]
        mask, scores, _ = segment_food(pillow_image, threshold=threshold, prompt=prompt)
        if not scores:
            logger.warning(f"  Items | '{prompt}' not found in image -> excluded from the total")
            continue
        items.append(FoodItemResult(prompt=prompt, mask=mask, score=float(np.mean(scores)), entity=entity))
    return items


def make_masks_disjoint(items: list[FoodItemResult]) -> None:
    """
    Give every contested pixel to the highest-scoring item, in place
    e.g Prompting separately for "rice" and "frozen veg" on a mixed plate returns heavily overlapping masks, and summing their volumes would count the shared region twice
    """
    if not items:
        return

    claimed = np.zeros(items[0].mask.shape, dtype=bool)
    for item in sorted(items, key=lambda i: i.score, reverse=True):
        overlap_px = int(np.sum((item.mask > 0) & claimed))
        if overlap_px:
            logger.info(f"  Items | '{item.prompt}' overlapped {overlap_px} px already claimed -> reassigned")
        kept = (item.mask > 0) & ~claimed
        claimed |= kept
        item.mask = kept.astype(np.uint8)


def segment_reference_object_with_score(pillow_image: Image.Image, utensil: str = "fork", threshold: float = 0.5) -> Optional[tuple[np.ndarray, float]]:
    """Segment a reference utensil with SAM 3 and return the highest-scoring instance mask and score, or None if nothing confident is found"""
    inputs = sam3_processor(images=pillow_image, text=utensil, return_tensors="pt").to(torch_device)
    sam3 = get_model("sam3")
    with torch.no_grad():
        outputs = sam3(**inputs)

    results = sam3_processor.post_process_instance_segmentation(
        outputs, threshold=threshold, mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]

    masks, scores = results["masks"], results["scores"]
    if masks.shape[0] == 0:
        logger.warning(f"  Reference | no '{utensil}' found -> cannot anchor scale from a reference object")
        return None

    best = int(torch.argmax(scores).item())
    score = float(scores[best].item())
    logger.info(f"  Reference | '{utensil}' found: score={score:.3f}  ({masks.shape[0]} candidate(s))")
    return masks[best].cpu().numpy().astype(np.uint8), score

def segment_reference_object(pillow_image: Image.Image, utensil: str = "fork", threshold: float = 0.5) -> Optional[np.ndarray]:
    """Segment a reference utensil with SAM 3 and return the single highest-scoring instance mask, or None if nothing confident is found"""
    result = segment_reference_object_with_score(pillow_image, utensil=utensil, threshold=threshold)
    if result is None:
        return None
    mask, _ = result
    return mask