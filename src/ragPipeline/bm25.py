import math
import threading

_cache_lock = threading.Lock()
# key: cache_key (str) → (ids, docs, metadatas, df, doc_freqs, avgdl, tokenized_corpus)
_bm25_cache: dict = {}


def _build_index(corpus):
    """Return (df, doc_freqs, avgdl) for BM25 scoring."""
    df = {}
    doc_freqs = []

    for doc in corpus:
        freqs = {}
        for term in doc:
            freqs[term] = freqs.get(term, 0) + 1
        doc_freqs.append(freqs)
        for term in freqs:
            df[term] = df.get(term, 0) + 1

    avgdl = sum(len(d) for d in corpus) / len(corpus) if corpus else 0
    return df, doc_freqs, avgdl


def _bm25_scores(query_terms, corpus, df, doc_freqs, avgdl, k1=1.5, b=0.75):
    """Score every document against query_terms using BM25Okapi."""
    N = len(corpus)
    scores = []

    for i, doc in enumerate(corpus):
        dl = len(doc)
        score = 0.0
        for term in query_terms:
            tf = doc_freqs[i].get(term, 0)
            if tf == 0:
                continue
            n = df.get(term, 0)
            # BM25 IDF with +1 smoothing (Okapi variant)
            idf = math.log((N - n + 0.5) / (n + 0.5) + 1)
            score += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
        scores.append(score)

    return scores


def invalidate_bm25_cache(cache_key: str) -> None:
    """Remove a cached BM25 index. Call after upload or delete for that collection."""
    with _cache_lock:
        _bm25_cache.pop(cache_key, None)


def get_or_build_corpus(ids, docs, metadatas, cache_key: str):
    """Return (tokenized_corpus, df, doc_freqs, avgdl) from cache or built fresh."""
    with _cache_lock:
        if cache_key in _bm25_cache:
            return _bm25_cache[cache_key]
    tokenized = [doc.lower().split() for doc in docs]
    df, doc_freqs, avgdl = _build_index(tokenized)
    entry = (tokenized, df, doc_freqs, avgdl)
    with _cache_lock:
        _bm25_cache[cache_key] = entry
    return entry


def bm25_search(question, ids, docs, metadatas, top_n=10, cache_key: str = None):
    """Score all docs with BM25Okapi and return the top_n as ranked dicts."""
    if cache_key:
        tokenized_corpus, df, doc_freqs, avgdl = get_or_build_corpus(ids, docs, metadatas, cache_key)
    else:
        # lowercase + whitespace split is intentionally naive — consistent with how BM25Okapi tokenizes
        tokenized_corpus = [doc.lower().split() for doc in docs]
        df, doc_freqs, avgdl = _build_index(tokenized_corpus)

    query_terms = question.lower().split()
    scores = _bm25_scores(query_terms, tokenized_corpus, df, doc_freqs, avgdl)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return [
        {"id": ids[i], "doc": docs[i], "meta": metadatas[i], "rank": rank}
        for rank, i in enumerate(ranked_indices, start=1)
    ]
