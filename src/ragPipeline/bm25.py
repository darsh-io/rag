from rank_bm25 import BM25Okapi


def bm25_search(question, ids, docs, metadatas, top_n=10):
    tokenized_corpus = [doc.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(question.lower().split())

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return [
        {"id": ids[i], "doc": docs[i], "meta": metadatas[i], "rank": rank}
        for rank, i in enumerate(ranked_indices, start=1)
    ]
