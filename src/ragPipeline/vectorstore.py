import chromadb
from ragPipeline.chunk import chunk_file
from ragPipeline.embeddings import getEmbeddings


def get_collection(db_path="chroma_db", collection_name="documents"):
    """Open or create a persistent Chroma collection with cosine similarity space."""
    client = chromadb.PersistentClient(path=db_path)
    # hnsw:space must be set at creation time; changing it later has no effect
    return client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})


def ingest(file_path, collection, api_key, api_url, model):
    """Chunk a PDF, embed each chunk, and upsert everything into the Chroma collection."""
    chunks = chunk_file(file_path)

    # stable id per chunk so re-ingesting the same file overwrites rather than duplicates
    ids = [f"{chunk['source']}_chunk{chunk['chunk_index']}" for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [{"source": chunk["source"], "page": chunk["page"], "chunk_index": chunk["chunk_index"]} for chunk in chunks]
    embeddings = [getEmbeddings(api_key, api_url, model, text) for text in texts]

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"Ingested {len(chunks)} chunks from {file_path}")
    return len(chunks)


def query(query_text, collection, api_key, api_url, model, n_results=5):
    """Embed query_text and return the n_results nearest chunks from the Chroma collection."""
    embedding = getEmbeddings(api_key, api_url, model, query_text)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        # ids are always returned by Chroma and cannot be listed in include
        include=["documents", "metadatas", "distances"],
    )
    return results
