import chromadb
from ragPipeline.chunk import chunk_pdf
from ragPipeline.embeddings import getEmbeddings


def get_collection(db_path="chroma_db", collection_name="documents"):
    client = chromadb.PersistentClient(path=db_path)
    return client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})


def ingest(file_path, collection, api_key, api_url, model):
    chunks = chunk_pdf(file_path)

    ids = [f"{chunk['source']}_chunk{chunk['chunk_index']}" for chunk in chunks]
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [{"source": chunk["source"], "page": chunk["page"], "chunk_index": chunk["chunk_index"]} for chunk in chunks]
    embeddings = [getEmbeddings(api_key, api_url, model, text) for text in texts]

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"Ingested {len(chunks)} chunks from {file_path}")


def query(query_text, collection, api_key, api_url, model, n_results=5):
    embedding = getEmbeddings(api_key, api_url, model, query_text)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances", "ids"],
    )
    return results
