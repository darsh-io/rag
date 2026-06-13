"""Document upload → Chroma ingest → DB record."""
import os, tempfile
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from src.db import get_db, new_id, now
from src.deps import require_role, class_access, UserInToken
from src.config import cfg, CHROMA_DIR
from src.ragPipeline.vectorstore import get_collection, ingest

router = APIRouter(tags=["documents"])
_teacher_or_admin = require_role("teacher", "supradmin")


@router.post("/classes/{class_id}/topics/{topic_id}/documents")
def upload_document(
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

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file.file.read())
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

    try:
        chunks_ingested = ingest(
            tmp_path, collection,
            cfg["api_key"], cfg["embed_url"], cfg["embed_model"],
            vision_cfg=vision_cfg,
            source_name=source_name,
        )
    except Exception as e:
        raise HTTPException(422, str(e))
    finally:
        os.unlink(tmp_path)

    doc_id = new_id()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO topic_documents (id,topic_id,filename,source_name,chunks_ingested,uploaded_by,uploaded_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (doc_id, topic_id, filename, source_name, chunks_ingested, caller.id, now()),
        )

    return {"id": doc_id, "filename": filename, "chunks_ingested": chunks_ingested}


@router.delete("/classes/{class_id}/topics/{topic_id}/documents/{doc_id}")
def delete_document(
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
