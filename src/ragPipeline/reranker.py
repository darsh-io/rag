import requests
import json

# v2 is the current Cohere endpoint; v1 is deprecated
COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"


def rerank(question, chunks, api_key, model, top_n=5):
    """Re-order chunks by Cohere relevance score and return the top_n with updated ranks."""
    # Cohere only needs the raw text; metadata is carried through locally
    documents = [doc for _, doc, _, _ in chunks]

    rq = requests.post(
        url=COHERE_RERANK_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps({
            "model": model,
            "query": question,
            "documents": documents,
            "top_n": top_n,
        }),
    )
    body = rq.json()
    if "results" not in body:
        raise RuntimeError(f"Cohere rerank error: {body}")

    reranked = []
    for new_rank, result in enumerate(body["results"], start=1):
        # result["index"] maps back to the original chunks list position
        _, doc, meta, _ = chunks[result["index"]]
        reranked.append((new_rank, doc, meta, result["relevance_score"]))

    return reranked
