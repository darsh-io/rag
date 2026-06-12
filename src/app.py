"""FastAPI application — thin wiring layer."""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# ragPipeline files use bare imports (e.g. `from ragPipeline.chunk import`)
# that assume src/ is on the path — add it so they resolve correctly.
sys.path.insert(0, str(Path(__file__).parent))
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.db import init_db
from src.routers import auth, users, classes, topics, documents, chats, feedback


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="rewise", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(users.router)
app.include_router(classes.router)
app.include_router(topics.router)
app.include_router(documents.router)
app.include_router(chats.router)
app.include_router(feedback.router)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def serve_frontend():
    return FileResponse(STATIC_DIR / "index.html")
