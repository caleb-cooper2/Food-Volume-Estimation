import argparse
import csv
import logging
import time
from pathlib import Path

from PIL import Image

from main import (CameraInfo, segment_food, estimate_depth, fit_support_plane, compute_volume)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CSV_FIELDS = ["dish_id", "volume_cm3", "food_coverage_pct", "plane_inliers", "mean_height_cm", "max_height_cm", "clipped_high_pct"]


def already_cached(out_path: Path) -> set[str]:
    "dish_ids already in the CSV so a killed run can just be re-run"
    if not out_path.exists():
        return set()
    with out_path.open() as f:
        return {row["dish_id"] for row in csv.DictReader(f)}


def volume_for_dish(pil_image: Image.Image) -> dict:
    """
    Run the inference geometry on one overhead RGB -> volume scalar + QC fields. Mirrors main.py's /estimate-volume endpoint: SAM 3 mask -> DepthPro metric
    depth+focal -> support-plane fit -> height integration, so the training scalar and the deployed scalar come out of the same code.
    """
    mask, scores, _ = segment_food(pil_image)
    coverage = float(mask.sum()) / mask.size * 100.0

    depth_map, focal_px = estimate_depth(pil_image)
    w, h = pil_image.size
    cam = CameraInfo(
        fx=focal_px, fy=focal_px, cx=w / 2.0, cy=h / 2.0,
        image_width=w, image_height=h, source="depthpro_fov"
    )

    plane_n, plane_p0, inliers = fit_support_plane(depth_map, mask, cam)
    vr = compute_volume(depth_map, mask, cam, plane_n, plane_p0)

    return {
        "volume_cm3": round(vr.volume_cm3, 2),
        "food_coverage_pct": round(coverage, 1),
        "plane_inliers": round(inliers, 3),
        "mean_height_cm": round(vr.mean_food_height_cm, 2),
        "max_height_cm": round(vr.max_food_height_cm, 2),
        "clipped_high_pct": round(vr.clipped_high_pct, 1),
    }


def main(args):
    root = Path(args.data_root)
    overhead_dir = root / "imagery/realsense_overhead"
    if not overhead_dir.is_dir():
        raise SystemExit(f"No overhead imagery at {overhead_dir}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = already_cached(out_path)
    logger.info(f"{len(done)} dishes already cached, resuming")

    dish_dirs = sorted(d for d in overhead_dir.iterdir() if d.is_dir())
    if args.limit:
        dish_dirs = dish_dirs[:args.limit]

    write_header = not out_path.exists()
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        n_ok, n_skip, t0 = 0, 0, time.perf_counter()
        for i, dish_dir in enumerate(dish_dirs):
            dish_id = dish_dir.name
            if dish_id in done:
                continue

            rgb_path = dish_dir / "rgb.png"
            if not rgb_path.exists():
                logger.warning(f"{dish_id}: no rgb.png, skipping")
                n_skip += 1
                continue

            try:
                pil = Image.open(rgb_path).convert("RGB")
                row = volume_for_dish(pil)
            except Exception as e:
                logger.error(f"{dish_id}: failed ({e}), skipping")
                n_skip += 1
                continue

            # QC: SAM 3 covering almost none or almost all of the frame -> plane fit unreliable, drop it
            if row["volume_cm3"] <= 0 or not (2.0 <= row["food_coverage_pct"] <= 80.0):
                logger.warning(f"{dish_id}: QC reject (vol={row['volume_cm3']} cov={row['food_coverage_pct']}%)")
                n_skip += 1
                continue

            # Physically implausible relief -> DepthPro almost certainly mis-scaled this scene, drop it
            if row["mean_height_cm"] > 8.0 or row["clipped_high_pct"] > 10.0:
                logger.warning(f"{dish_id}: QC reject relief (mean={row['mean_height_cm']}cm clipped={row['clipped_high_pct']:.0f}%)")
                n_skip += 1
                continue

            writer.writerow({"dish_id": dish_id, **row})
            f.flush() # flush per row so a killed run keeps everything up to here
            n_ok += 1

            if (i + 1) % args.log_every == 0:
                rate = (i + 1) / (time.perf_counter() - t0)
                logger.info(f"  {i+1}/{len(dish_dirs)} dishes | ok={n_ok} skip={n_skip} | {rate:.2f} dish/s")

    logger.info(f"Done. Wrote {n_ok} scalars, skipped {n_skip}. Cache -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache per-dish geometric volume scalars for --use_volume training")
    parser.add_argument("--data_root", type=str, default="./data/nutrition5k_dataset/imagery")
    parser.add_argument("--out", type=str, default="./data/volume_scalars.csv")
    parser.add_argument("--limit", type=int, default=0, help="Cap dishes processed (0 = all), for a quick smoke test")
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()
    main(args)