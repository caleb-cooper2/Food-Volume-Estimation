import io
import re

from PIL import Image
from fastapi import HTTPException, UploadFile
from pillow_heif import register_heif_opener

register_heif_opener()

ACCEPTED_MIME_TYPES = ("image/jpeg", "image/jpg", "image/png", "image/heic")
MIN_IMAGE_BYTES = 1024
MAX_IMAGE_BYTES = 30 * 1024 * 1024

PARTICIPANT_CODE_PATTERN = re.compile(r"^P\d{3}$")


def validate_participant_code(participant_code: str) -> str:
    """
    Every study request has to be attributable to a participant, so this is required on every estimation endpoint.
    Matches the format enforced client-side during onboarding.
    :raises HTTPException: when the code is missing or malformed
    """
    code = (participant_code or "").strip().upper()
    if not PARTICIPANT_CODE_PATTERN.match(code):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid or missing participant_code '{participant_code}' -> expected format like 'P014'",
        )
    return code


async def read_upload(file: UploadFile) -> tuple[bytes, Image.Image]:
    """
    Validate an uploaded image and decode it once, so callers have both the raw bytes (EXIF) and true pixel geometry
    :raises HTTPException: on an unsupported type, an implausible size, or bytes PIL cannot decode
    """
    if file.content_type not in ACCEPTED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Send JPEG or PNG only",
        )

    image_bytes = await file.read()

    if len(image_bytes) < MIN_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Image too small -> minimum 1 KB")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image too large -> maximum 30 MB")

    try:
        pillow_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot decode image: {exc}")

    return image_bytes, pillow_image
