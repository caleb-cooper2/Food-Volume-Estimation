import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from approaches import deep_learning, monocular
from approaches.monocular import job_worker
from model_manage import preload_all

@asynccontextmanager
async def lifespan(app: FastAPI):
    preload_all()
    worker_task = asyncio.create_task(job_worker())
    yield
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass

app = FastAPI(title="Volume Estimation API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(monocular.router)
app.include_router(deep_learning.router)