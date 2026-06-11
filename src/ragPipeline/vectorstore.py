import chromadb
from ragPipeline.chunk import chunk_file
from ragPipeline.embeddings import getEmbeddings


def get_collection(collection_name="documents"):
    """Create an in-memory Chroma collection with cosine similarity space."""
    client = chromadb.EphemeralClient()
    # hnsw:space must be set at creation time; changing it later has no effect
    return client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})


def ingest(file_path, collection, api_key, api_url, model, vision_cfg=None, source_name=None):
    """Chunk a file, embed each chunk, and upsert everything into the Chroma collection."""
    chunks = chunk_file(file_path, vision_cfg=vision_cfg, source_name=source_name)

    if not chunks:
        raise ValueError(
            "No text could be extracted from this file. "
            "If it's a scanned PDF or image-based document, it must be OCR'd first."
        )

    # drop chunks whose text is empty or whitespace — the embeddings API rejects them
    chunks = [c for c in chunks if c["text"].strip()]
    if not chunks:
        raise ValueError("File produced only empty chunks after extraction.")

    # stable id per chunk so re-ingesting the same file overwrites rather than duplicates
    ids = [f"{chunk['source']}_chunk{chunk['chunk_index']}" for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [{"source": chunk["source"], "page": chunk["page"], "chunk_index": chunk["chunk_index"]} for chunk in chunks]
    embeddings = [getEmbeddings(api_key, api_url, model, text) for text in texts]

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"Ingested {len(chunks)} chunks from {file_path}")
    return len(chunks)


def query(query_text, collection, api_key, api_url, model, n_results=5, source_filter=None):
    """Embed query_text and return the n_results nearest chunks from the Chroma collection."""
    embedding = getEmbeddings(api_key, api_url, model, query_text)
    kwargs = dict(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    if source_filter:
        where = {"source": {"$in": list(source_filter)}}
        kwargs["where"] = where
        # Chroma errors if n_results > matching docs — cap it
        available = len(collection.get(where=where, include=[])["ids"])
        if available == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        kwargs["n_results"] = min(n_results, available)
    return collection.query(**kwargs)


def get_filtered(collection, source_filter=None):
    """Get all chunks, optionally restricted to a list of source names."""
    if source_filter:
        return collection.get(
            where={"source": {"$in": list(source_filter)}},
            include=["documents", "metadatas"],
        )
    return collection.get()


def list_sources(collection):
    """Return unique source names in the collection with chunk counts."""
    data = collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in data["metadatas"]:
        src = meta.get("source", "")
        if src:
            counts[src] = counts.get(src, 0) + 1
    return [{"source": s, "chunks": c} for s, c in sorted(counts.items())]
