import secrets
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Optional


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

jobs: dict[str, JobResult] = {}

def create_job() -> tuple[str, str]:
    job_id = str(uuid.uuid4())
    poll_token = secrets.token_hex(32)
    jobs[job_id] = JobResult(job_id=job_id, status=JobStatus.PENDING, poll_token=poll_token)
    return job_id, poll_token

def get_job(job_id: str) -> Optional[JobResult]:
    return jobs.get(job_id)

def update_job_status(job_id: str, status: JobStatus, result: Optional[dict] = None, error_message: Optional[str] = None):
    if job_id in jobs:
        jobs[job_id].status = status
        jobs[job_id].result = result
        jobs[job_id].error_message = error_message