from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from approaches import deep_learning, monocular

app = FastAPI(title="Volume Estimation API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(monocular.router)
app.include_router(deep_learning.router)
