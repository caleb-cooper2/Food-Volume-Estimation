"""
Participant data capture for the study

    AppUsage/
        transfer_log.csv        one row per file written, synced_at left empty for the sync job to eventually fill
        P0NN/
            meals/              what the participant submitted (text, scale_ref) plus the full NLP entities JSON
            images/             the uploaded photo, byte-for-byte (keeps EXIF)
            estimates/          the full EstimationResponse as JSON

The three files of one request share a timestamp id so they can be joined later if needed
"""

import csv
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from logging_config import get_logger
from schemas import EstimationResponse

logger = get_logger(__name__)

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "AppUsage"))

LOG_COLUMNS = ["logged_at", "participant_code", "file", "synced_at"]

# Save uploads as received
EXTENSIONS = {"image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png", "image/heic": ".heic"}


def record_request(participant_code: str, image_bytes: bytes, content_type: str, meal: dict, estimate: EstimationResponse) -> None:
    """
    Archive one estimation request. Capture failing must not fail the participant's estimate, so errors are logged instead
    """
    if participant_code == "P000":  # skip for test user
        return
    try:
        now = datetime.now()
        record_id = now.strftime("%Y%m%dT%H%M%S%f") # simple way to set up a unique enough id
        participant_dir = DATA_ROOT / participant_code
        for sub in ("meals", "images", "estimates"):
            (participant_dir / sub).mkdir(parents=True, exist_ok=True)

        image_path = participant_dir / "images" / f"{record_id}{EXTENSIONS.get(content_type, '.jpg')}"
        image_path.write_bytes(image_bytes)

        meal_path = participant_dir / "meals" / f"{record_id}.json"
        meal_path.write_text(json.dumps({"timestamp": now.isoformat(), **meal}, indent=2))

        estimate_path = participant_dir / "estimates" / f"{record_id}.json"
        estimate_path.write_text(json.dumps(asdict(estimate), indent=2, default=float))

        log_path = DATA_ROOT / "transfer_log.csv"
        write_header = not log_path.exists()
        with open(log_path, "a", newline="") as fh:
            writer = csv.writer(fh)
            if write_header:
                writer.writerow(LOG_COLUMNS)
            for path in (image_path, meal_path, estimate_path):
                writer.writerow([now.isoformat(timespec="seconds"), participant_code, str(path.relative_to(DATA_ROOT)), ""])

        logger.info(f"[log] archived request {record_id} for {participant_code} under {participant_dir}")
    except Exception:
        logger.exception(f"Participant data capture failed for {participant_code} -> estimate still returned")