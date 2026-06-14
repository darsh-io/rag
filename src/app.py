"""FastAPI application — thin wiring layer."""
import os, sys, uuid
from contextlib import asynccontextmanager
from pathlib import Path

# ragPipeline files use bare imports (e.g. `from ragPipeline.chunk import`)
# that assume src/ is on the path — add it so they resolve correctly.
sys.path.insert(0, str(Path(__file__).parent))
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.db import init_db
from src.limiter import limiter
from src.logger import get_logger
from src.routers import auth, users, classes, topics, documents, chats, feedback

log = get_logger("rewise.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="rewise", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    log.info("request", extra={"method": request.method, "path": request.url.path, "status": response.status_code, "request_id": rid})
    return response


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
