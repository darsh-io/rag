"""Embedding API client tests with mocked HTTP."""
import pytest
from unittest.mock import patch, MagicMock


def _mock_resp(data):
    r = MagicMock()
    r.json.return_value = data
    return r


def test_get_embeddings_success():
    from src.ragPipeline.embeddings import getEmbeddings
    vec = [0.1, 0.2, 0.3]
    with patch("src.ragPipeline.embeddings.post_with_retry",
               return_value=_mock_resp({"data": [{"embedding": vec}]})):
        result = getEmbeddings("key", "http://api", "model", "hello")
    assert result == vec


def test_get_embeddings_api_error():
    from src.ragPipeline.embeddings import getEmbeddings
    with patch("src.ragPipeline.embeddings.post_with_retry",
               return_value=_mock_resp({"error": "bad key"})):
        with pytest.raises(RuntimeError, match="Embeddings API error"):
            getEmbeddings("key", "http://api", "model", "hello")


def test_get_embeddings_batch_single_batch():
    from src.ragPipeline.embeddings import getEmbeddingsBatch
    vecs = [[0.1, 0.2], [0.3, 0.4]]
    resp_data = {"data": [{"index": 0, "embedding": vecs[0]}, {"index": 1, "embedding": vecs[1]}]}
    with patch("src.ragPipeline.embeddings.post_with_retry",
               return_value=_mock_resp(resp_data)) as mock_post:
        result = getEmbeddingsBatch("key", "http://api", "model", ["a", "b"])
    assert result == vecs
    assert mock_post.call_count == 1


def test_get_embeddings_batch_multiple_batches():
    from src.ragPipeline.embeddings import getEmbeddingsBatch
    texts = [f"text{i}" for i in range(5)]
    # batch_size=3 → 2 calls: first returns 3 vecs, second returns 2 vecs
    batch1 = {"data": [{"index": i, "embedding": [float(i)]} for i in range(3)]}
    batch2 = {"data": [{"index": i, "embedding": [float(i+3)]} for i in range(2)]}
    with patch("src.ragPipeline.embeddings.post_with_retry",
               side_effect=[_mock_resp(batch1), _mock_resp(batch2)]) as mock_post:
        result = getEmbeddingsBatch("key", "http://api", "model", texts, batch_size=3)
    assert len(result) == 5
    assert mock_post.call_count == 2
