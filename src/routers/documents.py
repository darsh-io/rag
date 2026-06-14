"""Document upload → background Chroma ingest → DB record."""
import asyncio, os, tempfile
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from src.db import get_db, new_id, now
from src.deps import require_role, class_access, UserInToken
from src.config import cfg, CHROMA_DIR
from src.ragPipeline.vectorstore import get_collection, ingest

router = APIRouter(tags=["documents"])
_teacher_or_admin = require_role("teacher", "supradmin")

_MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024

_ALLOWED_MIME_PREFIXES = (
    "application/pdf",
    "application/vnd.openxmlformats-officedocument",
    "application/vnd.ms-",
    "application/msword",
    "application/vnd.oasis.opendocument",
    "text/plain",
    "text/csv",
    "application/epub+zip",
)


def _check_upload(data: bytes, filename: str) -> None:
    if len(data) > _MAX_BYTES:
        raise HTTPException(413, f"File exceeds {os.getenv('MAX_UPLOAD_MB', '20')} MB limit")
    try:
        import magic
        mime = magic.from_buffer(data, mime=True)
        if not any(mime.startswith(p) for p in _ALLOWED_MIME_PREFIXES):
            raise HTTPException(415, f"Unsupported file type: {mime}")
    except ImportError:
        ext = Path(filename).suffix.lower()
        allowed_exts = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
                        ".odt", ".ods", ".odp", ".txt", ".csv", ".epub"}
        if ext not in allowed_exts:
            raise HTTPException(415, f"Unsupported file extension: {ext}")


def _do_ingest(doc_id: str, tmp_path: str, class_id: str, source_name: str, vision_cfg: dict) -> None:
    """Blocking ingest — run in a thread. Updates DB status on completion or error."""
    collection = get_collection(f"class_{class_id}", CHROMA_DIR)
    try:
        chunks_ingested = ingest(
            tmp_path, collection,
            cfg["api_key"], cfg["embed_url"], cfg["embed_model"],
            vision_cfg=vision_cfg,
            source_name=source_name,
        )
        with get_db() as conn:
            conn.execute(
                "UPDATE topic_documents SET status='ready', chunks_ingested=? WHERE id=?",
                (chunks_ingested, doc_id),
            )
    except Exception as e:
        with get_db() as conn:
            conn.execute(
                "UPDATE topic_documents SET status='error', error_message=? WHERE id=?",
                (str(e), doc_id),
            )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.post("/classes/{class_id}/topics/{topic_id}/documents", status_code=202)
async def upload_document(
    class_id: str,
    topic_id: str,
    file: UploadFile = File(...),
    caller: UserInToken = Depends(_teacher_or_admin),
    cls=Depends(class_access("teacher")),
):
    with get_db() as conn:
        topic = conn.execute(
            "SELECT id FROM topics WHERE id=? AND class_id=?", (topic_id, class_id)
        ).fetchone()
    if not topic:
        raise HTTPException(404, "Topic not found in this class")

    filename = file.filename or "upload"
    ext = Path(filename).suffix.lower()
    if not ext:
        ext = ".pdf"

    data = await file.read()
    _check_upload(data, filename)

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    source_name = f"{Path(filename).stem}_topic{topic_id[:8]}"

    vision_cfg = {
        "api_key":      cfg["api_key"],
        "chat_url":     cfg["chat_url"],
        "vision_model": cfg["vision_model"],
    }

    # Remove any stale DB record + Chroma chunks for same filename in this topic
    collection = get_collection(f"class_{class_id}", CHROMA_DIR)
    with get_db() as conn:
        stale = conn.execute(
            "SELECT id, source_name FROM topic_documents WHERE topic_id=? AND filename=?",
            (topic_id, filename),
        ).fetchall()
        for row in stale:
            try:
                collection.delete(where={"source": {"$eq": row["source_name"]}})
            except Exception:
                pass
            conn.execute("DELETE FROM topic_documents WHERE id=?", (row["id"],))

    doc_id = new_id()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO topic_documents "
            "(id,topic_id,filename,source_name,chunks_ingested,uploaded_by,uploaded_at,status) "
            "VALUES (?,?,?,?,0,?,?,'processing')",
            (doc_id, topic_id, filename, source_name, caller.id, now()),
        )

    asyncio.create_task(
        asyncio.to_thread(_do_ingest, doc_id, tmp_path, class_id, source_name, vision_cfg)
    )

    return {"id": doc_id, "filename": filename, "status": "processing"}


@router.delete("/classes/{class_id}/topics/{topic_id}/documents/{doc_id}")
async def delete_document(
    class_id: str,
    topic_id: str,
    doc_id: str,
    caller: UserInToken = Depends(_teacher_or_admin),
    cls=Depends(class_access("teacher")),
):
    with get_db() as conn:
        row = conn.execute(
            "SELECT source_name FROM topic_documents WHERE id=? AND topic_id=?",
            (doc_id, topic_id),
        ).fetchone()
    if not row:
        raise HTTPException(404, "Document not found")

    collection = get_collection(f"class_{class_id}", CHROMA_DIR)
    try:
        collection.delete(where={"source": {"$eq": row["source_name"]}})
    except Exception:
        pass

    with get_db() as conn:
        conn.execute("DELETE FROM topic_documents WHERE id=?", (doc_id,))

    return {"ok": True}
