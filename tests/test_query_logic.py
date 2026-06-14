"""RAG query pipeline tests with fully mocked sub-functions."""
import pytest
from unittest.mock import patch, MagicMock


def _fake_chroma_results(n=3):
    return {
        "ids":       [[f"id{i}" for i in range(n)]],
        "documents": [[f"doc text {i}" for i in range(n)]],
        "metadatas": [[{"source": f"src{i}", "page": i+1} for i in range(n)]],
        "distances": [[0.1 * i for i in range(n)]],
    }


def _fake_filtered_data(n=3):
    return {
        "ids":       [f"id{i}" for i in range(n)],
        "documents": [f"doc text {i}" for i in range(n)],
        "metadatas": [{"source": f"src{i}", "page": i+1} for i in range(n)],
    }


def _fake_chunks():
    return [(1, "doc text 0", {"source": "src0", "page": 1}, 0.95)]


@patch("query.generate_hypothetical_doc", return_value="hypothetical text")
@patch("query.chroma_query", return_value=_fake_chroma_results())
@patch("query.get_filtered", return_value=_fake_filtered_data())
@patch("query.bm25_search", return_value=[{"id": "id0", "doc": "doc text 0", "meta": {"source": "src0", "page": 1}, "rank": 1}])
@patch("query.reciprocal_rank_fusion", return_value=[(1, "doc text 0", {"source": "src0", "page": 1}, 0.5)])
@patch("query.rerank", return_value=_fake_chunks())
def test_build_rag_context(mock_rerank, mock_rrf, mock_bm25, mock_filtered, mock_chroma, mock_hyde):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from query import build_rag_context

    collection = MagicMock()
    collection.name = "class_test"
    chunks, messages, hyde_doc = build_rag_context(
        "What is attention?", [], collection,
        "api-key", "http://embed", "embed-model",
        "http://chat", "llm-model",
        "cohere-key", "rerank-model",
    )
    assert hyde_doc == "hypothetical text"
    assert len(chunks) >= 1
    assert any("system" in m["role"] for m in messages)


@patch("query.generate_hypothetical_doc", return_value="hyp")
@patch("query.chroma_query", return_value=_fake_chroma_results())
@patch("query.get_filtered", return_value={"ids": [], "documents": [], "metadatas": []})
def test_build_rag_context_no_docs_raises(mock_filtered, mock_chroma, mock_hyde):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from query import build_rag_context
    collection = MagicMock()
    collection.name = "class_test"
    with pytest.raises(ValueError, match="No documents"):
        build_rag_context(
            "question", [], collection,
            "key", "http://embed", "model",
            "http://chat", "llm",
            "cohere", "reranker",
        )


@patch("query.post_with_retry")
def test_call_llm(mock_post):
    import sys, json
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from query import call_llm

    mock_post.return_value.json.return_value = {
        "choices": [{"message": {"content": "The answer is 42.", "reasoning": None}}]
    }
    result = call_llm([{"role": "user", "content": "hi"}], "key", "http://chat", "model")
    assert result == "The answer is 42."


@patch("query.post_with_retry_stream")
def test_call_llm_stream(mock_post):
    import sys, json
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from query import call_llm_stream

    lines = [
        b'data: {"choices":[{"delta":{"content":"Hello "}}]}',
        b'data: {"choices":[{"delta":{"content":"world"}}]}',
        b'data: [DONE]',
    ]
    mock_post.return_value.iter_lines.return_value = lines
    deltas = list(call_llm_stream([{"role":"user","content":"hi"}], "key", "http://chat", "model"))
    assert deltas == ["Hello ", "world"]
