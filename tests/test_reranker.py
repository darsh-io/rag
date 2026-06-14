"""Cohere reranker tests with mocked HTTP."""
import pytest
from unittest.mock import patch, MagicMock


def _mock_resp(data):
    r = MagicMock()
    r.json.return_value = data
    return r


def _chunks():
    return [
        (1, "Attention is all you need", {"source": "paper", "page": 1}, 0.5),
        (2, "RNNs process sequences", {"source": "paper", "page": 2}, 0.4),
        (3, "CNNs for images", {"source": "other", "page": 1}, 0.3),
    ]


def test_rerank_success():
    from src.ragPipeline.reranker import rerank
    results = [
        {"index": 0, "relevance_score": 0.95},
        {"index": 2, "relevance_score": 0.60},
        {"index": 1, "relevance_score": 0.40},
    ]
    with patch("src.ragPipeline.reranker.post_with_retry",
               return_value=_mock_resp({"results": results})):
        reranked = rerank("attention mechanism", _chunks(), "key", "rerank-model")
    assert len(reranked) == 3
    assert reranked[0][1] == "Attention is all you need"
    assert reranked[0][3] == 0.95


def test_rerank_api_error():
    from src.ragPipeline.reranker import rerank
    with patch("src.ragPipeline.reranker.post_with_retry",
               return_value=_mock_resp({"message": "invalid key"})):
        with pytest.raises(RuntimeError, match="Cohere rerank error"):
            rerank("query", _chunks(), "bad-key", "model")


def test_rerank_preserves_metadata():
    from src.ragPipeline.reranker import rerank
    results = [{"index": 1, "relevance_score": 0.8}, {"index": 0, "relevance_score": 0.6}]
    with patch("src.ragPipeline.reranker.post_with_retry",
               return_value=_mock_resp({"results": results})):
        reranked = rerank("query", _chunks(), "key", "model")
    assert reranked[0][2]["source"] == "paper"
    assert reranked[0][2]["page"] == 2
