import asyncio
import json
import secrets
import sqlite3
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Optional

DB_PATH = "jobs.db"
JOB_TTL_SECONDS = 60 * 60 # stale job records get cleaned up after this
POLL_INTERVAL_SECONDS = 0.15 # how often dequeue_job checks for new work


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobResult:
    job_id: str
    status: JobStatus
    result: Optional[dict] = None
    error_message: Optional[str] = None
    poll_token: str = ""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")     # let readers and the writer coexist
    conn.execute("PRAGMA busy_timeout=30000")   # wait instead of erroring under contention
    return conn


def init_db():
    conn = connect()
    try:
        conn.execute("""
                     CREATE TABLE IF NOT EXISTS jobs (
                                                         job_id TEXT PRIMARY KEY,
                                                         status TEXT NOT NULL,
                                                         result TEXT,
                                                         error_message TEXT,
                                                         poll_token TEXT NOT NULL,
                                                         created_at REAL NOT NULL
                     )
                     """)
        conn.execute("""
                     CREATE TABLE IF NOT EXISTS queue (
                                                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                          job_id TEXT NOT NULL,
                                                          payload TEXT NOT NULL,
                                                          created_at REAL NOT NULL
                     )
                     """)
        conn.commit()
    finally:
        conn.close()


init_db()


def create_job_sync() -> tuple[str, str]:
    job_id = str(uuid.uuid4())
    poll_token = secrets.token_hex(32)
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO jobs (job_id, status, result, error_message, poll_token, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, JobStatus.PENDING.value, None, None, poll_token, time.time())
        )
        conn.commit()
    finally:
        conn.close()
    return job_id, poll_token


async def create_job() -> tuple[str, str]:
    return await asyncio.to_thread(create_job_sync)


def _get_job_sync(job_id: str) -> Optional[JobResult]:
    conn = connect()
    try:
        row = conn.execute(
            "SELECT job_id, status, result, error_message, poll_token FROM jobs WHERE job_id = ?",
            (job_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    job_id, status, result, error_message, poll_token = row
    return JobResult(
        job_id=job_id,
        status=JobStatus(status),
        result=json.loads(result) if result else None,
        error_message=error_message,
        poll_token=poll_token
    )


async def get_job(job_id: str) -> Optional[JobResult]:
    return await asyncio.to_thread(_get_job_sync, job_id)


def update_job_status_sync(job_id: str, status: JobStatus, result: Optional[dict], error_message: Optional[str]):
    conn = connect()
    try:
        conn.execute(
            "UPDATE jobs SET status = ?, result = ?, error_message = ? WHERE job_id = ?",
            (status.value, json.dumps(result, default=str) if result is not None else None, error_message, job_id)
        )
        conn.commit()
    finally:
        conn.close()


async def update_job_status(job_id: str, status: JobStatus, result: Optional[dict] = None, error_message: Optional[str] = None):
    await asyncio.to_thread(update_job_status_sync, job_id, status, result, error_message)


def enqueue_job_sync(job_id: str, payload: dict):
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO queue (job_id, payload, created_at) VALUES (?, ?, ?)",
            (job_id, json.dumps(payload), time.time())
        )
        conn.commit()
    finally:
        conn.close()


async def enqueue_job(job_id: str, image_bytes: bytes, content_type: str | None, participant_code: str, scale_ref: str, text: str):
    import base64
    payload = {
        "image_b64": base64.b64encode(image_bytes).decode("ascii"),
        "content_type": content_type,
        "participant_code": participant_code,
        "scale_ref": scale_ref,
        "text": text
    }
    await asyncio.to_thread(enqueue_job_sync, job_id, payload)


def dequeue_job_once_sync() -> Optional[tuple[str, dict]]:
    """
    First in, last out, grab oldest job row from the table.
    """
    conn = connect()
    try:
        conn.execute("BEGIN IMMEDIATE") #takes the write lock up front so two concurrent callers can't both grab the same row.
        row = conn.execute("SELECT id, job_id, payload FROM queue ORDER BY id ASC LIMIT 1").fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        row_id, job_id, payload = row
        conn.execute("DELETE FROM queue WHERE id = ?", (row_id,))
        conn.execute("COMMIT")
        return job_id, json.loads(payload)
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


async def dequeue_job() -> Optional[tuple[str, dict]]:
    """
    Polls DB instead of blocking. Returns None if no new jobs are available.
    """
    while True:
        item = await asyncio.to_thread(dequeue_job_once_sync)
        if item is not None:
            return item
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


def cleanup_old_jobs_sync():
    cutoff = time.time() - JOB_TTL_SECONDS
    conn = connect()
    try:
        conn.execute("DELETE FROM jobs WHERE created_at < ?", (cutoff,))
        conn.commit()
    finally:
        conn.close()


async def cleanup_old_jobs():
    await asyncio.to_thread(cleanup_old_jobs_sync)