import argparse
import csv
import logging
import time
from pathlib import Path

from PIL import Image

from main import CameraInfo, segment_food, estimate_depth, metric_footprint_diameter_cm
from train import load_overhead_depth_m

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CSV_FIELDS = ["dish_id", "footprint_cm", "food_coverage_pct"]


def already_cached(out_path: Path) -> set[str]:
    "dish_ids already in the CSV so a killed run can just be re-run"
    if not out_path.exists():
        return set()
    with out_path.open() as f:
        return {row["dish_id"] for row in csv.DictReader(f)}


def footprint_for_dish(overhead_rgb_path: Path) -> dict:
    """
    Real-world food footprint for one overhead dish -> size-prior training target. Uses the RealSense sensor depth (true metric) for
    Z and DepthPro's focal for the FOV, so the target is a genuine cm size. Inference measures same quantity from DepthPro depth,
    so head learns the true size that corrects DepthPro's scale
    """
    image = Image.open(overhead_rgb_path).convert("RGB")
    mask, _, _ = segment_food(image)
    coverage = float(mask.sum()) / mask.size * 100.0

    sensor_depth_m = load_overhead_depth_m(overhead_rgb_path) # true metric Z from the RealSense sensor
    _, focal_px = estimate_depth(image) # the metric Z comes from the sensor above
    width, height = image.size
    camera = CameraInfo(fx=focal_px, fy=focal_px, cx=width / 2.0, cy=height / 2.0, image_width=width, image_height=height, source="depthpro_fov")

    diameter_cm = metric_footprint_diameter_cm(sensor_depth_m, mask, camera)
    return {"footprint_cm": round(diameter_cm, 2), "food_coverage_pct": round(coverage, 1)}


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
        for i, dish_directory in enumerate(dish_dirs):
            dish_id = dish_directory.name
            if dish_id in done:
                continue

            rgb_path = dish_directory / "rgb.png"
            if not rgb_path.exists() or not rgb_path.with_name("depth_raw.png").exists():
                logger.warning(f"{dish_id}: missing rgb.png or depth_raw.png, skipping")
                n_skip += 1
                continue

            try:
                row = footprint_for_dish(rgb_path)
            except Exception as e:
                logger.error(f"{dish_id}: failed ({e}), skipping")
                n_skip += 1
                continue

            # QC: bad coverage -> mask unreliable;
            # likely an implausible footprint -> depth/mask failure
            if not (2.0 <= row["food_coverage_pct"] <= 80.0) or not (2.0 <= row["footprint_cm"] <= 60.0):
                logger.warning(f"{dish_id}: QC reject (cm={row['footprint_cm']} cov={row['food_coverage_pct']}%)")
                n_skip += 1
                continue

            writer.writerow({"dish_id": dish_id, **row})
            f.flush() # flush per row so a killed run keeps everything up to here
            n_ok += 1

            if (i + 1) % args.log_every == 0:
                rate = (i + 1) / (time.perf_counter() - t0)
                logger.info(f"  {i+1}/{len(dish_dirs)} dishes | ok={n_ok} skip={n_skip} | {rate:.2f} dish/s")

    logger.info(f"Done. Wrote {n_ok} footprints, skipped {n_skip}. Cache -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache per-dish real-world food footprint (cm)")
    parser.add_argument("--data_root", type=str, default="./data/nutrition5k_dataset")
    parser.add_argument("--out", type=str, default="./data/footprint_sizes.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()
    main(args)
