import requests
import json

COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"


def rerank(question, chunks, api_key, model, top_n=5):
    """
    chunks: list of (rank, doc, meta, score) from build_context
    returns: top_n chunks re-ordered and re-numbered by Cohere relevance score
    """
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
        _, doc, meta, _ = chunks[result["index"]]
        reranked.append((new_rank, doc, meta, result["relevance_score"]))

    return reranked
