from rank_bm25 import BM25Okapi


def bm25_search(question, ids, docs, metadatas, top_n=10):
    """Score all docs with BM25Okapi and return the top_n as ranked dicts."""
    # lowercase + whitespace split is intentionally naive — consistent with how BM25Okapi tokenizes
    tokenized_corpus = [doc.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(question.lower().split())

    # sort by score descending and take the top_n indices
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return [
        {"id": ids[i], "doc": docs[i], "meta": metadatas[i], "rank": rank}
        for rank, i in enumerate(ranked_indices, start=1)
    ]
