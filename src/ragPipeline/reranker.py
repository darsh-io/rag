import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.http_client import post_with_retry

# v2 is the current Cohere endpoint; v1 is deprecated
COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"


def rerank(question, chunks, api_key, model, top_n=None):
    """Re-order chunks by Cohere relevance score and return the top_n with updated ranks."""
    documents = [doc for _, doc, _, _ in chunks]

    rq = post_with_retry(
        url=COHERE_RERANK_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": model,
            "query": question,
            "documents": documents,
            "top_n": top_n if top_n is not None else len(documents),
        }),
    )
    body = rq.json()
    if "results" not in body:
        raise RuntimeError(f"Cohere rerank error: {body}")

    reranked = []
    for new_rank, result in enumerate(body["results"], start=1):
        _, doc, meta, _ = chunks[result["index"]]
        reranked.append((new_rank, doc, meta, result["relevance_score"]))

    return reranked
