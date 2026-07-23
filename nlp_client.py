import logging
import os
import httpx

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

NLP_URL = os.environ.get("NLP_URL", "http://localhost:8001")
NLP_TIMEOUT = float(os.environ.get("NLP_TIMEOUT", "30.0"))

async def extract_entities(text):
    if not text or not text.strip():
        logger.warning("No text to extract entities from")
        return []

    try:
        async with httpx.AsyncClient(timeout=NLP_TIMEOUT) as client:
            response = await client.post(f"{NLP_URL}/extract", json={"text": text})
            response.raise_for_status()
    except (httpx.HTTPError, httpx.TimeoutException) as exc:
        logger.warning(f"NLP /extract failed ({exc})... Proceeding without densities")
        return []

    payload = response.json()
    return payload.get("entities", payload) if isinstance(payload, dict) else []