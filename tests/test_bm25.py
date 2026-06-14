"""BM25 unit tests."""
from src.ragPipeline.bm25 import bm25_search, invalidate_bm25_cache, get_or_build_corpus, _bm25_cache


def _corpus():
    ids  = ["doc1", "doc2", "doc3"]
    docs = [
        "the transformer architecture uses attention mechanisms",
        "recurrent neural networks process sequences step by step",
        "attention is all you need for sequence to sequence tasks",
    ]
    metas = [{"source": f"s{i}", "page": 1} for i in range(3)]
    return ids, docs, metas


def test_bm25_top_result():
    ids, docs, metas = _corpus()
    results = bm25_search("attention mechanism transformer", ids, docs, metas, top_n=1)
    assert results[0]["id"] in ("doc1", "doc3")


def test_bm25_returns_top_n():
    ids, docs, metas = _corpus()
    results = bm25_search("attention", ids, docs, metas, top_n=2)
    assert len(results) == 2


def test_bm25_cache_stores_and_hits():
    ids, docs, metas = _corpus()
    key = "test_cache_key"
    _bm25_cache.pop(key, None)
    get_or_build_corpus(ids, docs, metas, key)
    assert key in _bm25_cache


def test_bm25_cache_invalidate():
    key = "test_invalidate_key"
    _bm25_cache[key] = ("dummy",)
    invalidate_bm25_cache(key)
    assert key not in _bm25_cache


def test_bm25_with_cache_key():
    ids, docs, metas = _corpus()
    key = "bm25_functional_test"
    r1 = bm25_search("neural network", ids, docs, metas, top_n=1, cache_key=key)
    r2 = bm25_search("neural network", ids, docs, metas, top_n=1, cache_key=key)
    assert r1[0]["id"] == r2[0]["id"]
