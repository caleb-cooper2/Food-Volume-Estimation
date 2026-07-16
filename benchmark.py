import argparse
import csv
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Benchmark for the three approaches using deployed HTTP endpoints, testing the path the phone would use

# SimpleFood45 dataset contains food images with 5x4 checkerboard that gives two exact references our deployment path only guesses at (scale and tilt)
# We use the checkerboard tilt as ground truth to stratify error by view obliquity (does the geometric approach degrade as the paper's single-axis assumption predicts,
# and does its own geometry_confidence track the real tilt?)

ENDPOINTS = {
    "monocular-geometric": "http://localhost:8000/api/v1/estimate-volume",
    "deep-learning": "http://localhost:8000/api/v1/estimate-volume-dl",
    "multi-view": "http://localhost:8001/api/v1/estimate-volume-multiview",
}

SINGLE_IMAGE_APPROACHES = {"monocular-geometric", "deep-learning"}

DEFAULT_DENSITY_G_CM3 = 0.8  # only used to bridge units when a sample has no measured density
MULTIVIEW_MAX_FRAMES = 8  # evenly spaced frames handed to the multi-view endpoint
REQUEST_TIMEOUT_S = 600 # model inference on CPU/MPS is slow so give it room

CHECKERBOARD_SQUARE_CM = 1.2      # SimpleFood45 says corners 1.2 cm apart
CHECKERBOARD_INNER = (4, 3)       # (cols, rows) of inner corners on the 5x4 board
TILT_BUCKETS = ((0.0, 15.0, "overhead"), (15.0, 35.0, "mild"), (35.0, 999.0, "oblique"))

MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".heic": "image/heic"}

MANIFEST_FIELDS = ["sample_id", "tier", "images", "approaches", "gt_mass_g", "gt_volume_cm3",
                   "density", "scale_ref", "ref_scale_cm_per_px", "ref_tilt_deg", "source"]
RESULT_FIELDS = [
    "sample_id", "tier", "source", "approach", "ok",
    "pred_mass_g", "pred_volume_cm3", "gt_mass_g", "gt_volume_cm3",
    "density_used", "density_assumed", "ref_tilt_deg", "confidence", "error"
]


@dataclass
class ResultRow:
    sample_id: str
    tier: str
    source: str
    approach: str
    ok: bool
    pred_mass_g: Optional[float]
    pred_volume_cm3: Optional[float]
    gt_mass_g: Optional[float]
    gt_volume_cm3: Optional[float]
    density_used: Optional[float]
    density_assumed: bool
    ref_tilt_deg: Optional[float]
    confidence: Optional[str]
    error: str = ""


# Manifest I/O
def read_manifest(path: Path) -> list[dict]:
    """A manifest row = one food sample: image(s), which approaches to run, and whatever GT was measured"""
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"Empty manifest: {path}")
    missing = set(MANIFEST_FIELDS) - set(rows[0].keys())
    if missing:
        raise SystemExit(f"Manifest {path} missing columns: {sorted(missing)}")
    return rows


def resolve_images(images_field: str) -> list[Path]:
    """images column is either a glob ('.../frame_*.png') or a ';'-separated explicit list"""
    if any(ch in images_field for ch in "*?[") and ";" not in images_field:
        paths = sorted(Path().glob(images_field)) or sorted(Path("/").glob(images_field.lstrip("/")))
    else:
        paths = [Path(p.strip()) for p in images_field.split(";") if p.strip()]
    return [p for p in paths if p.exists()]


def as_float(value: str) -> Optional[float]:
    value = (value or "").strip()
    return float(value) if value else None


# Endpoint dispatch
def file_tuple(path: Path):
    mime = MIME_BY_SUFFIX.get(path.suffix.lower(), "application/octet-stream")
    return path.name, path.read_bytes(), mime


def call_volume_endpoint(approach: str, image_paths: list[Path]) -> dict:
    """POST to the deployed endpoint and returns the EstimationResponse json"""
    url = ENDPOINTS[approach]
    if approach in SINGLE_IMAGE_APPROACHES:
        files = {"file": file_tuple(image_paths[len(image_paths) // 2])}
    else:
        if len(image_paths) <= MULTIVIEW_MAX_FRAMES:
            frames = image_paths
        else:
            idx = np.linspace(0, len(image_paths) - 1, MULTIVIEW_MAX_FRAMES).round().astype(int)
            frames = [image_paths[i] for i in idx]
        files = [("files", file_tuple(p)) for p in frames]
    resp = requests.post(url, files=files, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json()


# Checkerboard scale + angle reference
@dataclass
class CheckerboardPose:
    found: bool
    scale_cm_per_px: Optional[float] = None
    tilt_deg: Optional[float] = None  # board-plane angle off the optical axis -> 0 = straight-on / overhead
    n_corners: int = 0


def detect_checkerboard(image_path: Path) -> CheckerboardPose:
    img = cv2.imread(str(image_path))
    if img is None:
        return CheckerboardPose(False)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

    for cols, rows in (CHECKERBOARD_INNER, CHECKERBOARD_INNER[::-1]):  # board may sit either orientation
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
        if found:
            break
    if not found:
        return CheckerboardPose(False)

    corners = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )
    scale = corner_spacing_scale(corners, cols, rows)
    tilt = checkerboard_tilt(corners, cols, rows, img.shape)
    return CheckerboardPose(True, round(scale, 5), round(tilt, 1), len(corners))


def corner_spacing_scale(corners: np.ndarray, cols: int, rows: int) -> float:
    """cm-per-pixel from the median neighbour spacing. Mild tilt foreshortens this slightly but good enough as a scale reference"""
    grid = corners.reshape(rows, cols, 2)
    dx = np.linalg.norm(np.diff(grid, axis=1), axis=2)
    dy = np.linalg.norm(np.diff(grid, axis=0), axis=2)
    spacing_px = float(np.median(np.concatenate([dx.ravel(), dy.ravel()])))
    return CHECKERBOARD_SQUARE_CM / spacing_px


def checkerboard_tilt(corners: np.ndarray, cols: int, rows: int, img_shape) -> float:
    image_height, image_width = img_shape[:2]
    focal_px = 1.2 * max(image_width, image_height)
    camera_matrix = np.array([[focal_px, 0, image_width / 2.0], [0, focal_px, image_height / 2.0], [0, 0, 1]], dtype=np.float64)

    board_corners_cm = np.zeros((cols * rows, 3), np.float32)
    board_corners_cm[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * CHECKERBOARD_SQUARE_CM
    solved, board_rotation_vector, board_translation_vector = cv2.solvePnP(board_corners_cm, corners, camera_matrix, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not solved:
        return float("nan")

    rotation_matrix, _ = cv2.Rodrigues(board_rotation_vector)
    board_normal = rotation_matrix[:, 2]  # board +z axis expressed in the camera frame
    view_direction = board_translation_vector.ravel() / np.linalg.norm(board_translation_vector)  # camera -> board centre
    normal_view_cos = abs(float(board_normal @ view_direction))
    return float(np.degrees(np.arccos(np.clip(normal_view_cos, 0.0, 1.0))))


# Unit bridge, need to handle either volume or mass when needed
def bridge_units(resp: dict, sample_density: Optional[float]) -> tuple[Optional[float], Optional[float], float, bool]:
    pred_mass = resp.get("mass_g")
    pred_volume = resp.get("volume_cm3")
    density = sample_density if sample_density else DEFAULT_DENSITY_G_CM3
    assumed = sample_density is None

    if pred_volume is not None and pred_mass is None:
        pred_mass = pred_volume * density
    elif pred_mass is not None and pred_volume is None:
        pred_volume = pred_mass / density
    else:
        assumed = False # nothing was converted, so no density assumption was leaned on
    return pred_mass, pred_volume, density, assumed


# Metrics (numpy)
def mape(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target) / np.clip(np.abs(target), 1.0, None)) * 100.0)


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def bias_pct(pred: np.ndarray, target: np.ndarray) -> float: # (mean signed % error), tells us if method systematically over/under-shoots
    return float(np.mean((pred - target) / np.clip(np.abs(target), 1.0, None)) * 100.0)


def r2(pred: np.ndarray, target: np.ndarray) -> float:
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - target.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


# Run harness
def already_done(out_path: Path) -> set[tuple[str, str]]:
    """(sample_id, approach) pairs already scored, so an interrupted run resumes instead of re-inferring"""
    if not out_path.exists():
        return set()
    with out_path.open() as f:
        return {(r["sample_id"], r["approach"]) for r in csv.DictReader(f)}


def run_benchmark(args) -> None:
    manifest = read_manifest(Path(args.manifest))
    if args.limit:
        manifest = manifest[:args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = already_done(out_path)
    logger.info(f"{len(done)} (sample, approach) results already present, resuming")

    write_header = not out_path.exists()
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()

        t0 = time.perf_counter()
        for row in manifest:
            sample_id, tier, source = row["sample_id"], row["tier"], row["source"]
            gt_mass, gt_volume = as_float(row["gt_mass_g"]), as_float(row["gt_volume_cm3"])
            density, ref_tilt = as_float(row["density"]), as_float(row["ref_tilt_deg"])
            image_paths = resolve_images(row["images"])
            approaches = [a.strip() for a in row["approaches"].split(";") if a.strip()]

            if not image_paths:
                logger.warning(f"{sample_id}: no images resolved from '{row['images']}', skipping")
                continue

            for approach in approaches:
                if (sample_id, approach) in done:
                    continue
                result = score_one(sample_id, tier, source, approach, image_paths, gt_mass, gt_volume, density, ref_tilt)
                writer.writerow({k: v for k, v in asdict(result).items() if k in RESULT_FIELDS})
                f.flush()  # flush per row so an interrupted run keeps everything up to here

    logger.info(f"Done in {time.perf_counter() - t0:.0f}s. Results -> {out_path}")
    summarise(out_path)


def score_one(sample_id, tier, source, approach, image_paths, gt_mass, gt_volume, density, ref_tilt) -> ResultRow:
    try:
        resp = call_volume_endpoint(approach, image_paths)
    except Exception as e:
        logger.error(f"{sample_id} [{approach}]: request failed ({e})")
        return ResultRow(sample_id, tier, source, approach, False, None, None, gt_mass, gt_volume, None, False, ref_tilt, None, str(e))

    pred_mass, pred_volume, density_used, assumed = bridge_units(resp, density)
    logger.info(
        f"{sample_id} [{approach}]: mass={safe_format(pred_mass)}g vol={safe_format(pred_volume)}cm3 "
        f"(gt mass={safe_format(gt_mass)} vol={safe_format(gt_volume)}) tilt={safe_format(ref_tilt)} conf={resp.get('confidence')}"
    )
    return ResultRow(
        sample_id, tier, source, approach, True,
        safe_round(pred_mass), safe_round(pred_volume), gt_mass, gt_volume,
        round(density_used, 3), assumed, ref_tilt, resp.get("confidence")
    )


# Reporting
def summarise(out_path: Path) -> None:
    """
    Group by (source, approach) and report MAE/MAPE/bias/R2 in each unit, then a second pass that stratifies the monocular-geometric volume error
    by checkerboard tilt bucket -> the benefit of the angle reference
    """
    with out_path.open() as f:
        rows = [r for r in csv.DictReader(f) if r["ok"] == "True"]
    if not rows:
        logger.warning("No successful results to summarise")
        return

    logger.info("=" * 96)
    logger.info(f"{'source':<14}{'approach':<22}{'unit':<12}{'n':>4}{'MAE':>10}{'MAPE%':>9}{'bias%':>9}{'R2':>8}  {'note':<12}")
    logger.info("-" * 96)
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        groups.setdefault((r["source"], r["approach"]), []).append(r)

    for (source, approach), grp in sorted(groups.items()):
        for unit, pred_key, gt_key in (("mass_g", "pred_mass_g", "gt_mass_g"), ("volume_cm3", "pred_volume_cm3", "gt_volume_cm3")):
            pairs = [(float(r[pred_key]), float(r[gt_key]), r["density_assumed"] == "True")
                     for r in grp if r[pred_key] and r[gt_key]]
            if not pairs:
                continue
            pred = np.array([p for p, _, _ in pairs])
            gt = np.array([g for _, g, _ in pairs])
            note = "density-assumed" if any(a for _, _, a in pairs) else ""
            logger.info(
                f"{source:<14}{approach:<22}{unit:<12}{len(pairs):>4}"
                f"{mae(pred, gt):>10.1f}{mape(pred, gt):>9.1f}{bias_pct(pred, gt):>9.1f}{r2(pred, gt):>8.2f}  {note:<12}"
            )
    logger.info("=" * 96)
    summarise_by_tilt(rows)


def summarise_by_tilt(rows: list[dict]) -> None:
    """monocular-geometric volume error vs checkerboard camera tilt -> does obliqueness inflate error?"""
    geo = [r for r in rows if r["approach"] == "monocular-geometric" and r["pred_volume_cm3"] and r["gt_volume_cm3"] and r["ref_tilt_deg"]]
    if not geo:
        return
    logger.info(f"monocular-geometric volume by checkerboard tilt:")
    logger.info(f"{'tilt bucket':<20}{'n':>4}{'MAPE%':>9}{'bias%':>9}")
    for lo, hi, name in TILT_BUCKETS:
        bucket = [r for r in geo if lo <= float(r["ref_tilt_deg"]) < hi]
        if not bucket:
            continue
        pred = np.array([float(r["pred_volume_cm3"]) for r in bucket])
        gt = np.array([float(r["gt_volume_cm3"]) for r in bucket])
        logger.info(f"{name + f' ({lo:.0f}-{hi:.0f} deg)':<20}{len(bucket):>4}{mape(pred, gt):>9.1f}{bias_pct(pred, gt):>9.1f}")
    logger.info("=" * 96)


def safe_format(x: Optional[float]) -> str:
    return f"{x:.1f}" if x is not None else "-"


def safe_round(x: Optional[float]) -> Optional[float]:
    return round(x, 2) if x is not None else None

# SimpleFood45 manifest builder
SESSION_GAP_S = 30  # capture-time gap that separates one physical item's shots from the next's

def split_by_session(frames: list[Path], max_gap_s: float = SESSION_GAP_S) -> list[list[Path]]:
    """Split time-sorted frames wherever the capture gap exceeds max_gap_s -> one chunk per physical item"""
    timed = sorted((datetime.strptime(p.stem[:15], "%Y%m%d_%H%M%S"), p) for p in frames)
    if any(t is None for t, _ in timed):
        return [sorted(frames)]  # unparseable names -> can't split, keep as one item
    chunks, current = [], [timed[0][1]]
    for (prev_t, _), (cur_t, cur_p) in zip(timed, timed[1:]):
        if (cur_t - prev_t).total_seconds() > max_gap_s:
            chunks.append(current)
            current = []
        current.append(cur_p)
    chunks.append(current)
    return chunks


def build_simplefood45_manifest(args) -> None:
    labels_path = Path(args.labels)
    images_root = Path(args.images_root) if args.images_root else labels_path.parent

    with labels_path.open() as file:
        rows = list(csv.DictReader(file))
    cols = {col.lower(): col for col in rows[0].keys()}
    img_col = find_columns(cols, ["image", "file", "name"])
    label_col = find_columns(cols, ["label", "class", "food"])
    weight_col = find_columns(cols, ["weight", "mass"])
    volume_col = find_columns(cols, ["volume", "ml"])
    if not (img_col and label_col and weight_col and volume_col):
        raise SystemExit(f"Could not find image/label/weight/volume columns in {list(cols.values())}")
    logger.info(f"Label columns -> image='{img_col}' label='{label_col}' weight='{weight_col}' volume='{volume_col}'")

    # group images by (label, weight, volume) = one GT, labelled images missing from disk are dropped
    groups: dict[tuple, list[Path]] = defaultdict(list)
    n_missing = 0
    for row in rows:
        img_path = images_root / str(row[img_col])
        if not img_path.exists():
            n_missing += 1
            continue
        groups[(row[label_col], row[weight_col], row[volume_col])].append(img_path)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_label_count: dict[str, int] = defaultdict(int)
    n_items, n_no_board = 0, 0
    with out_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for (label, weight, volume), frames in sorted(groups.items()):
            mass_g, volume_cm3 = float(weight), float(volume)  # volume mL == cm^3
            for item_frames in split_by_session(frames):
                sample_id = f"{label}_{per_label_count[label]:02d}"
                per_label_count[label] += 1

                pose = detect_checkerboard(item_frames[len(item_frames) // 2])
                if not pose.found:
                    logger.warning(f"{sample_id}: no checkerboard in representative frame")
                    n_no_board += 1

                density = mass_g / volume_cm3 if volume_cm3 > 0 else ""
                writer.writerow({
                    "sample_id": sample_id,
                    "tier": "simplefood45",
                    "images": ";".join(str(p) for p in item_frames),
                    "approaches": args.approaches,
                    "gt_mass_g": round(mass_g, 2),
                    "gt_volume_cm3": round(volume_cm3, 2),
                    "density": round(density, 4) if density else "",
                    "scale_ref": "checkerboard",
                    "ref_scale_cm_per_px": pose.scale_cm_per_px if pose.found else "",
                    "ref_tilt_deg": pose.tilt_deg if pose.found else "",
                    "source": "simplefood45",
                })
                n_items += 1
    logger.info(f"Wrote {n_items} items across {len(groups)} GT groups "
                f"({n_missing} labelled images not found on disk, {n_no_board} items without a detected board) -> {out_path}")


def find_columns(cols_lower: dict, hints: list[str]) -> Optional[str]:
    for hint in hints:
        for lower, original in cols_lower.items():
            if hint in lower:
                return original
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarker for the three volume/mass approaches")
    sub = parser.add_subparsers(dest="command", required=True)

    parser_run = sub.add_parser("run")
    parser_run.add_argument("--manifest", type=str, required=True)
    parser_run.add_argument("--out", type=str, default="./data/benchmark_results.csv")
    parser_run.add_argument("--limit", type=int, default=0)
    parser_run.set_defaults(func=run_benchmark)

    parser_simplefood45 = sub.add_parser("build-simplefood45")
    parser_simplefood45.add_argument("--labels", type=str, required=True)
    parser_simplefood45.add_argument("--images_root", type=str, default="")
    parser_simplefood45.add_argument("--out", type=str, default="./data/benchmark_manifest_simplefood45.csv")
    parser_simplefood45.add_argument("--approaches", type=str, default="monocular-geometric;deep-learning;multi-view")
    parser_simplefood45.set_defaults(func=build_simplefood45_manifest)

    args = parser.parse_args()
    args.func(args)
