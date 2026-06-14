import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.http_client import post_with_retry


def getEmbeddings(api_key, api_url, model, text):
    """Return the embedding vector for a single text using the given OpenRouter model."""
    rq = post_with_retry(
        url=api_url,
        headers={"Authorization": f"Bearer {api_key}"},
        # OpenRouter follows the OpenAI embeddings schema: "input" not "text"
        data=json.dumps({"model": model, "input": text}),
    )
    body = rq.json()
    if "data" not in body:
        raise RuntimeError(f"Embeddings API error: {body}")
    return body["data"][0]["embedding"]


def getEmbeddingsBatch(api_key, api_url, model, texts, batch_size=100):
    """Return a list of embedding vectors for a list of texts, batched."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        rq = post_with_retry(
            url=api_url,
            headers={"Authorization": f"Bearer {api_key}"},
            data=json.dumps({"model": model, "input": batch}),
        )
        body = rq.json()
        if "data" not in body:
            raise RuntimeError(f"Embeddings API error: {body}")
        # OpenAI-compatible response: data is ordered by index
        batch_vecs = sorted(body["data"], key=lambda x: x["index"])
        embeddings.extend([item["embedding"] for item in batch_vecs])
    return embeddings
