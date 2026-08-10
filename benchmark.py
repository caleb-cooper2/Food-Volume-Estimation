import argparse
import csv
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import requests

from checkerboard import detect_pose
from logging_config import get_logger

logger = get_logger(__name__)

# SimpleFood45 dataset contains food images with 5x4 checkerboard that gives two exact references our deployment path only guesses at (scale and tilt)
# We use the checkerboard tilt as ground truth to figure out error by view obliquity (does the geometric approach degrade as the paper's single-axis assumption predicts,
# and does its own geometry_confidence track the real tilt?)

API_BASE_URL = os.environ.get("VOLUME_API_URL", "http://localhost:8001")

ENDPOINTS = {
    "monocular-geometric": f"{API_BASE_URL}/api/v1/estimate-volume",
    "deep-learning": f"{API_BASE_URL}/api/v1/estimate-volume-dl",
    "multi-view": f"{API_BASE_URL}/api/v1/estimate-volume-multiview"
}

# Every request now requires participant_code -> not a real participant, but satisfies the P\d{3} format the endpoints validate
BENCHMARK_PARTICIPANT_CODE = "P000"

BENCHMARK_ARMS = {
    "utensil_nlp": {"scale_ref": "utensil", "use_text": True},
    "utensil_notext": {"scale_ref": "utensil", "use_text": False},
    "sizeprior_nlp": {"scale_ref": "size_prior", "use_text": True},
    "sizeprior_notext": {"scale_ref": "size_prior", "use_text": False},
    "checkerboard": {"scale_ref": "checkerboard", "use_text": False},
    "none": {"scale_ref": "size_prior", "use_text": False}
}


SINGLE_IMAGE_APPROACHES = {"monocular-geometric", "deep-learning"}

DEFAULT_ARMS_BY_SOURCE = {
    "simplefood45": ["checkerboard", "sizeprior_notext"],
    "custom": ["utensil_nlp", "utensil_notext", "sizeprior_nlp", "sizeprior_notext"],
}

DEFAULT_DENSITY_G_CM3 = 0.8  # only used to bridge units when a sample has no measured density
MULTIVIEW_MAX_FRAMES = 3  # evenly spaced frames handed to the multi-view endpoint
REQUEST_TIMEOUT_S = 600 # model inference on CPU/MPS is slow so give it room

TILT_BUCKETS = ((0.0, 15.0, "overhead"), (15.0, 35.0, "mild"), (35.0, 999.0, "oblique"))

MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".heic": "image/heic"}

CUSTOM_POSES = ("overhead", "tilt", "side_left", "side_right")
CUSTOM_MULTIVIEW_POSES = ("tilt", "side_left", "side_right")
CUSTOM_UTENSILS = ("fork", "knife", "spoon")

MANIFEST_FIELDS = ["sample_id", "tier", "images", "approaches", "gt_mass_g", "gt_volume_cm3", "density", "scale_ref",
                   "ref_scale_cm_per_px", "ref_tilt_deg", "source", "text", "pose", "utensil"]
RESULT_FIELDS = [
    "sample_id", "tier", "source", "approach", "arm", "pose", "utensil", "ok", "pred_mass_g", "pred_volume_cm3",
    "gt_mass_g", "gt_volume_cm3", "density_used", "density_assumed", "ref_tilt_deg", "confidence", "scale_source",
    "scale_factor", "n_items", "latency_s", "error"
]


@dataclass
class ResultRow:
    sample_id: str
    tier: str
    source: str
    approach: str
    arm: str
    pose: str
    utensil: str
    ok: bool
    pred_mass_g: Optional[float]
    pred_volume_cm3: Optional[float]
    gt_mass_g: Optional[float]
    gt_volume_cm3: Optional[float]
    density_used: Optional[float]
    density_assumed: bool
    ref_tilt_deg: Optional[float]
    confidence: Optional[str]
    scale_source: str = ""
    scale_factor: Optional[float] = None
    n_items: Optional[int] = None
    latency_s: Optional[float] = None
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


def call_volume_endpoint(approach: str, image_paths: list[Path], scale_ref: str, text: str = "") -> dict:
    """
    POST to a deployed endpoint and return the EstimationResponse json. scale_ref selects the scale anchor, text supplies
    the food description that drives the SAM 3 prompt and the density lookup
    """
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
    resp = requests.post(
        url,
        files=files,
        data={"participant_code": BENCHMARK_PARTICIPANT_CODE, "scale_ref": scale_ref, "text": text},
        timeout=REQUEST_TIMEOUT_S
    )
    resp.raise_for_status()
    return resp.json()


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
def already_done(out_path: Path) -> set[tuple[str, str, str]]:
    """(sample_id, approach, arm) combinations already scored OK, so an interrupted run resumes and failures get retried"""
    if not out_path.exists():
        return set()
    with out_path.open() as f:
        return {(r["sample_id"], r["approach"], r.get("arm", "")) for r in csv.DictReader(f) if r["ok"] == "True"}


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

            arms = args.arms or DEFAULT_ARMS_BY_SOURCE.get(source, list(BENCHMARK_ARMS))
            for approach in approaches:
                for arm in arms_for_approach(approach, arms):
                    if (sample_id, approach, arm) in done:
                        continue
                    result = score_one(sample_id, tier, source, approach, arm, row, image_paths, gt_mass, gt_volume, density, ref_tilt)
                    writer.writerow({k: v for k, v in asdict(result).items() if k in RESULT_FIELDS})
                    f.flush()  # flush per row so an interrupted run keeps everything up to here

    logger.info(f"Done in {time.perf_counter() - t0:.0f}s. Results -> {out_path}")
    summarise(out_path)


def arms_for_approach(approach: str, arms: list[str]) -> list[str]:
    """deep-learning ignores scale_ref and text, multi-view ignores text -> don't re-send identical requests"""
    if approach == "deep-learning":
        return ["none"]
    if approach == "multi-view":
        no_text = [a for a in arms if not BENCHMARK_ARMS[a]["use_text"]]
        return no_text or arms[:1]
    return arms


def score_one(sample_id, tier, source, approach, arm, row, image_paths, gt_mass, gt_volume, density, ref_tilt) -> ResultRow:
    arm_config = BENCHMARK_ARMS[arm]
    scale_ref = arm_config["scale_ref"]
    text = row["text"] if arm_config["use_text"] else ""
    pose, utensil = row.get("pose", ""), row.get("utensil", "")

    for attempt in (1, 2):  # one automatic retry so a transient server hiccup doesn't leave a hole in the data
        t_request = time.perf_counter()
        try:
            resp = call_volume_endpoint(approach, image_paths, scale_ref, text)
            latency_s = round(time.perf_counter() - t_request, 1)
            break
        except Exception as e:
            if attempt == 2:
                logger.error(f"{sample_id} [{approach}/{arm}]: request failed twice ({e})")
                return ResultRow(sample_id, tier, source, approach, arm, pose, utensil, False,
                                 None, None, gt_mass, gt_volume, None, False, ref_tilt, None, error=str(e))
            logger.warning(f"{sample_id} [{approach}/{arm}]: request failed ({e}), retrying")

    diagnostics = resp.get("diagnostics") or {}
    pred_mass, pred_volume, density_used, assumed = bridge_units(resp, density)
    logger.info(
        f"{sample_id} [{approach}/{arm}] {pose}/{utensil}: vol={safe_format(pred_volume)}cm3 "
        f"mass={safe_format(pred_mass)}g (gt vol={safe_format(gt_volume)} mass={safe_format(gt_mass)}) "
        f"anchor={diagnostics.get('scale_source')} conf={resp.get('confidence')}"
    )
    return ResultRow(
        sample_id, tier, source, approach, arm, pose, utensil, True,
        safe_round(pred_mass), safe_round(pred_volume), gt_mass, gt_volume,
        round(density_used, 3), assumed, ref_tilt, resp.get("confidence"),
        scale_source=diagnostics.get("scale_source", ""),
        scale_factor=diagnostics.get("scale_factor"),
        n_items=len(diagnostics.get("items") or []),
        latency_s=latency_s
    )

# Reporting
SUCCESS_MAPE_PCT = 20.0  # RQ1 success criterion: mean relative volume error under 20%


def summarise(out_path: Path) -> None:
    """
    RQ verdicts first, then one table per axis that actually varies: approaches when several were run
    (SimpleFood45 runs all three), arms when several were run (the custom A/B), tilt buckets when the
    board gave a reference tilt. Logged, and written next to the results as markdown
    """
    if not out_path.exists():
        raise SystemExit(f"No results file at {out_path}")
    with out_path.open() as f:
        # last row per (sample, approach, arm) wins, so a retried failure never double-counts
        latest = {(r["sample_id"], r["approach"], r.get("arm", "")): r for r in csv.DictReader(f)}
    rows = [r for r in latest.values() if r["ok"] == "True"]
    if not rows:
        logger.warning("No successful results to summarise")
        return

    lines = summarise_rq_verdicts(rows)
    if len({r["approach"] for r in rows}) > 1:
        lines += summarise_by_approach(rows)
    if len({r["arm"] for r in rows}) > 1:
        lines += summarise_by_arm(rows)
    lines += summarise_by_tilt(rows)

    for line in lines:
        logger.info(line)

    summary_path = out_path.with_name(out_path.stem + "_summary.md")
    summary_path.write_text(
        f"# Benchmark summary\n\nGenerated {datetime.now():%Y-%m-%d %H:%M} from `{out_path.name}` "
        f"({len(rows)} scored results)\n\n```\n" + "\n".join(lines) + "\n```\n"
    )
    logger.info(f"Summary -> {summary_path}")


def summarise_rq_verdicts(rows: list[dict]) -> list[str]:
    """The two success criteria, stated as pass/fail so the run answers the research questions directly"""
    lines = [f"RQ1 - volume MAPE vs ground truth (success < {SUCCESS_MAPE_PCT:.0f}%):"]
    for approach in sorted({r["approach"] for r in rows}):
        volume_pred, volume_gt = paired_arrays([r for r in rows if r["approach"] == approach], "pred_volume_cm3", "gt_volume_cm3")
        if len(volume_pred):
            error_pct = mape(volume_pred, volume_gt)
            lines.append(f"  {approach:<22}{error_pct:>7.1f}%  (n={len(volume_pred)})  -> {'PASS' if error_pct < SUCCESS_MAPE_PCT else 'FAIL'}")

    nlp_lines = summarise_nlp_effect(rows)
    lines += ["", "RQ2 - NLP text vs image-only, paired per image (success: text reduces error):"]
    lines += nlp_lines or ["  no paired nlp/notext arms in this results file"]
    return lines + [""]


def summarise_by_approach(rows: list[dict]) -> list[str]:
    """
    Approach comparison, for datasets that run more than one
    """
    header = f"{'approach':<22}{'n':>4}{'V-MAPE%':>9}{'V-bias%':>9}{'M-MAPE%':>9}{'R2(mass)':>10}{'lat_s':>7}"
    lines = ["=" * len(header), header, "-" * len(header)]

    for approach in sorted({r["approach"] for r in rows}):
        grp = [r for r in rows if r["approach"] == approach]
        volume_pred, volume_gt = paired_arrays(grp, "pred_volume_cm3", "gt_volume_cm3")
        mass_pred, mass_gt = paired_arrays(grp, "pred_mass_g", "gt_mass_g")

        line = f"{approach:<22}{len(grp):>4}"
        line += f"{mape(volume_pred, volume_gt):>9.1f}{bias_pct(volume_pred, volume_gt):>9.1f}" if len(volume_pred) else f"{'-':>9}{'-':>9}"
        line += f"{mape(mass_pred, mass_gt):>9.1f}{r2(mass_pred, mass_gt):>10.2f}" if len(mass_pred) else f"{'-':>9}{'-':>10}"
        line += mean_latency(grp)
        lines.append(line)
    return lines + ["=" * len(header), ""]


def summarise_by_arm(rows: list[dict]) -> list[str]:
    """A/B testing volume errors, broken out by pose. Mass is shown only for the arms that supply text, because without it,
     the endpoint returns no density and the mass would be seperated from ground truth"""
    header = f"{'approach':<22}{'arm':<18}{'n':>4}{'V-MAPE%':>9}{'V-bias%':>9}{'M-MAPE%':>9}{'lat_s':>7}" + "".join(f"{p:>12}" for p in CUSTOM_POSES)
    lines = ["=" * len(header), header, "-" * len(header)]

    for approach in sorted({r["approach"] for r in rows}):
        for arm in sorted({r["arm"] for r in rows if r["approach"] == approach}):
            grp = [r for r in rows if r["approach"] == approach and r["arm"] == arm]
            volume_pred, volume_gt = paired_arrays(grp, "pred_volume_cm3", "gt_volume_cm3")
            line = f"{approach:<22}{arm:<18}{len(grp):>4}"
            line += f"{mape(volume_pred, volume_gt):>9.1f}{bias_pct(volume_pred, volume_gt):>9.1f}" if len(volume_pred) else f"{'-':>9}{'-':>9}"
            mass_pred, mass_gt = paired_arrays(grp, "pred_mass_g", "gt_mass_g") if BENCHMARK_ARMS[arm]["use_text"] else ([], [])
            line += f"{mape(mass_pred, mass_gt):>9.1f}" if len(mass_pred) else f"{'-':>9}"
            line += mean_latency(grp)

            for pose in CUSTOM_POSES:
                pose_pred, pose_gt = paired_arrays([r for r in grp if r["pose"] == pose], "pred_volume_cm3", "gt_volume_cm3")
                line += f"{mape(pose_pred, pose_gt):>12.1f}" if len(pose_pred) else f"{'-':>12}"
            lines.append(line)
    return lines + ["=" * len(header), ""]


def summarise_by_tilt(rows: list[dict]) -> list[str]:
    """Volume error stratified by the checkerboard's reference tilt (SimpleFood45), per approach:
    does accuracy degrade as the view goes oblique?"""
    tilted = [r for r in rows if r["ref_tilt_deg"]]
    if not tilted:
        return []
    header = f"{'approach':<22}{'tilt':<10}{'n':>4}{'V-MAPE%':>9}{'V-bias%':>9}"
    lines = ["=" * len(header), header, "-" * len(header)]
    for approach in sorted({r["approach"] for r in tilted}):
        for low, high, bucket in TILT_BUCKETS:
            grp = [r for r in tilted if r["approach"] == approach and low <= float(r["ref_tilt_deg"]) < high]
            volume_pred, volume_gt = paired_arrays(grp, "pred_volume_cm3", "gt_volume_cm3")
            if len(volume_pred):
                lines.append(f"{approach:<22}{bucket:<10}{len(volume_pred):>4}"
                             f"{mape(volume_pred, volume_gt):>9.1f}{bias_pct(volume_pred, volume_gt):>9.1f}")
    return lines + ["=" * len(header), ""]


def mean_latency(grp: list[dict]) -> str:
    values = [float(r["latency_s"]) for r in grp if r.get("latency_s")]
    return f"{np.mean(values):>7.1f}" if values else f"{'-':>7}"


def paired_arrays(rows: list[dict], pred_key: str, gt_key: str) -> tuple[np.ndarray, np.ndarray]:
    """Prediction and ground-truth arrays for the rows where both values are present"""
    pairs = [(float(r[pred_key]), float(r[gt_key])) for r in rows if r[pred_key] and r[gt_key]]
    return np.array([p for p, _ in pairs]), np.array([g for _, g in pairs])


def summarise_nlp_effect(rows: list[dict]) -> list[str]:
    """Paired per-image volume error with and without the NLP text, holding the scale anchor fixed"""
    errors = {
        (r["approach"], r["sample_id"], r["arm"]):
            abs(float(r["pred_volume_cm3"]) - float(r["gt_volume_cm3"])) / max(float(r["gt_volume_cm3"]), 1.0) * 100.0
        for r in rows if r["pred_volume_cm3"] and r["gt_volume_cm3"]
    }

    lines = []
    for anchor in ("utensil", "sizeprior"):
        pairs = [(errors[(approach, sid, f"{anchor}_nlp")], errors[(approach, sid, f"{anchor}_notext")])
                 for approach, sid, arm in errors
                 if arm == f"{anchor}_nlp" and (approach, sid, f"{anchor}_notext") in errors]
        if not pairs:
            continue
        differences = np.array([with_text - without_text for with_text, without_text in pairs])  # negative = text helped
        n_better = int(np.sum(differences < 0))
        lines.append(
            f"  {anchor:<10} {differences.mean():+7.1f}pp over {len(pairs)} pairs, {n_better}/{len(pairs)} better, "
            f"sign p={sign_test_p(n_better, len(pairs)):.3f}  -> {'PASS' if differences.mean() < 0 else 'FAIL'}"
        )
    return lines


def sign_test_p(n_better: int, n_total: int) -> float:
    """Two-sided sign-test p-value, so the paired comparison has a number attached without scipy"""
    if n_total == 0:
        return float("nan")
    tail = min(n_better, n_total - n_better)
    cumulative = sum(math.comb(n_total, k) for k in range(tail + 1))
    return min(1.0, 2.0 * cumulative / 2 ** n_total)


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

                pose = detect_pose(item_frames[len(item_frames) // 2])
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
                    "source": "simplefood45"
                })
                n_items += 1
    logger.info(f"Wrote {n_items} items across {len(groups)} GT groups ({n_missing} labelled images not found on disk, {n_no_board} items without a detected board) -> {out_path}")


def parse_capture(stem: str) -> tuple[str, str]:
    """Pull the pose and the in-frame utensil out of a filename like 'side_left_fork'"""
    lower = stem.lower()
    pose = next((p for p in CUSTOM_POSES if lower.startswith(p)), "")
    utensil = next((u for u in CUSTOM_UTENSILS if u in lower), "")
    return pose, utensil


def build_custom_manifest(args) -> None:
    """One manifest row per image. Ground truth (mass, volume, density) is per folder and repeats across that folder's images"""
    labels_path = Path(args.labels)
    images_root = Path(args.images_root)

    with labels_path.open() as file:
        label_rows = [row for row in csv.reader(file) if any(cell.strip() for cell in row)][1:]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows, n_mv_rows, n_missing_folders = 0, 0, 0
    with out_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()

        for label_row in label_rows:
            folder_id, food_label = label_row[0].strip(), label_row[1].strip()
            gt_mass_g, gt_volume_cm3 = float(label_row[2]), float(label_row[4])

            # "mixed" marks a combination meal where no single density applies
            raw_density = label_row[3].strip().lower()
            density = "" if raw_density in ("mixed", "") else round(float(raw_density), 4)

            folder = images_root / folder_id
            if not folder.is_dir():
                logger.warning(f"{folder_id}: folder not found at {folder}, skipping")
                n_missing_folders += 1
                continue

            base = {
                "tier": "custom",
                "approaches": args.approaches,
                "gt_mass_g": round(gt_mass_g, 2),
                "gt_volume_cm3": round(gt_volume_cm3, 2),
                "density": density, "scale_ref": "",
                "ref_scale_cm_per_px": "",
                "ref_tilt_deg": "",
                "source": "custom",
                "text": food_label
            }

            captures: dict[tuple[str, str], Path] = {}
            for image_path in sorted(folder.iterdir()):
                if image_path.suffix.lower() not in MIME_BY_SUFFIX:
                    continue
                pose, utensil = parse_capture(image_path.stem)
                if not pose:
                    logger.warning(f"{folder_id}/{image_path.name}: unrecognised pose in filename, skipping")
                    continue
                captures.setdefault((pose, utensil), image_path)
                writer.writerow({
                    **base,
                    "sample_id": f"{folder_id}_{image_path.stem}",
                    "images": str(image_path),
                    "pose": pose,
                    "utensil": utensil
                })
                n_rows += 1

            if args.no_multiview:
                continue
            # one multi-view row per utensil so the in-frame scale anchor is consistent across the three views
            for utensil in sorted({u for _, u in captures}):
                frames = [captures.get((pose, utensil)) for pose in CUSTOM_MULTIVIEW_POSES]
                if any(f is None for f in frames):
                    missing = [p for p, f in zip(CUSTOM_MULTIVIEW_POSES, frames) if f is None]
                    logger.warning(f"{folder_id} [{utensil or 'none'}]: missing {missing}, no multi-view row")
                    continue
                writer.writerow({
                    **base,
                    "approaches": "multi-view",
                    "sample_id": f"{folder_id}_mv_{utensil or 'none'}",
                    "images": ";".join(str(f) for f in frames),
                    "pose": "multiview", "utensil": utensil
                })
                n_mv_rows += 1

    logger.info(f"Wrote {n_rows} image rows, {n_mv_rows} from {len(label_rows)} folders ({n_missing_folders} folders missing on disk) -> {out_path}")

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
    parser_run.add_argument("--arms", nargs="+", default=None, choices=list(BENCHMARK_ARMS)) #default: picked per sample source (simplefood45 -> checkerboard, custom -> the four utensil/size-prior arms
    parser_run.set_defaults(func=run_benchmark)

    parser_simplefood45 = sub.add_parser("build-simplefood45")
    parser_simplefood45.add_argument("--labels", type=str, default="./data/ordered_dataset/labels.csv")
    parser_simplefood45.add_argument("--images_root", type=str, default="./data/ordered_dataset")
    parser_simplefood45.add_argument("--out", type=str, default="./data/benchmark_manifest_simplefood45.csv")
    parser_simplefood45.add_argument("--approaches", type=str, default="monocular-geometric;deep-learning;multi-view")
    parser_simplefood45.set_defaults(func=build_simplefood45_manifest)

    parser_custom = sub.add_parser("build-custom")
    parser_custom.add_argument("--labels", type=str, default="./data/custom_dataset/label-info.csv")
    parser_custom.add_argument("--images_root", type=str, default="./data/custom_dataset")
    parser_custom.add_argument("--out", type=str, default="./data/benchmark_manifest_custom.csv")
    parser_custom.add_argument("--approaches", type=str, default="monocular-geometric;deep-learning")
    parser_custom.add_argument("--no-multiview", action="store_true")
    parser_custom.set_defaults(func=build_custom_manifest)

    # regenerate the summary from an existing results csv without re-running anything
    parser_report = sub.add_parser("report")
    parser_report.add_argument("--results", type=str, default="./data/benchmark_results.csv")
    parser_report.set_defaults(func=lambda a: summarise(Path(a.results)))

    args = parser.parse_args()
    args.func(args)
