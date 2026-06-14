"""Vectorstore tests with mocked ChromaDB and embeddings."""
import pytest
from unittest.mock import patch, MagicMock, call


def _make_collection(name="test_col"):
    col = MagicMock()
    col.name = name
    return col


def test_get_collection_ephemeral():
    from src.ragPipeline.vectorstore import get_collection
    with patch("src.ragPipeline.vectorstore.chromadb.EphemeralClient") as mock_client:
        mock_client.return_value.get_or_create_collection.return_value = MagicMock(name="col")
        col = get_collection("docs")
    mock_client.assert_called_once()


def test_get_collection_persistent():
    from src.ragPipeline.vectorstore import get_collection
    with patch("src.ragPipeline.vectorstore.chromadb.PersistentClient") as mock_client:
        mock_client.return_value.get_or_create_collection.return_value = MagicMock(name="col")
        col = get_collection("docs", persist_dir="/tmp/test")
    mock_client.assert_called_once_with(path="/tmp/test")


def test_ingest_calls_batch_embeddings():
    from src.ragPipeline.vectorstore import ingest

    fake_chunks = [
        {"text": "chunk one text", "source": "doc_topic1234", "page": 1, "chunk_index": 0},
        {"text": "chunk two text", "source": "doc_topic1234", "page": 1, "chunk_index": 1},
    ]
    fake_embeddings = [[0.1, 0.2], [0.3, 0.4]]
    collection = _make_collection()

    with patch("src.ragPipeline.vectorstore.chunk_file", return_value=fake_chunks), \
         patch("src.ragPipeline.vectorstore.getEmbeddingsBatch", return_value=fake_embeddings):
        count = ingest("/fake/file.pdf", collection, "key", "http://api", "model")

    assert count == 2
    collection.add.assert_called_once()
    args = collection.add.call_args
    assert len(args.kwargs["ids"]) == 2


def test_ingest_empty_file_raises():
    from src.ragPipeline.vectorstore import ingest
    collection = _make_collection()
    with patch("src.ragPipeline.vectorstore.chunk_file", return_value=[]):
        with pytest.raises(ValueError, match="No text could be extracted"):
            ingest("/fake.pdf", collection, "key", "http://api", "model")


def test_ingest_all_whitespace_raises():
    from src.ragPipeline.vectorstore import ingest
    collection = _make_collection()
    chunks = [{"text": "   ", "source": "s", "page": 1, "chunk_index": 0}]
    with patch("src.ragPipeline.vectorstore.chunk_file", return_value=chunks):
        with pytest.raises(ValueError, match="only empty chunks"):
            ingest("/fake.pdf", collection, "key", "http://api", "model")


def test_query_returns_results():
    from src.ragPipeline.vectorstore import query
    fake_result = {
        "ids": [["id1", "id2"]],
        "documents": [["doc1", "doc2"]],
        "metadatas": [[{"source": "s", "page": 1}, {"source": "s", "page": 2}]],
        "distances": [[0.1, 0.2]],
    }
    collection = _make_collection()
    collection.query.return_value = fake_result

    with patch("src.ragPipeline.vectorstore.getEmbeddings", return_value=[0.1, 0.2]):
        result = query("test question", collection, "key", "http://api", "model", n_results=2)

    assert result["ids"] == [["id1", "id2"]]


def test_query_with_source_filter():
    from src.ragPipeline.vectorstore import query
    collection = _make_collection()
    collection.get.return_value = {"ids": ["id1"], "documents": ["d"], "metadatas": [{}]}
    collection.query.return_value = {
        "ids": [["id1"]], "documents": [["d"]], "metadatas": [[{}]], "distances": [[0.1]]
    }
    with patch("src.ragPipeline.vectorstore.getEmbeddings", return_value=[0.1]):
        result = query("q", collection, "k", "u", "m", n_results=5, source_filter=["src1"])
    collection.query.assert_called_once()


def test_query_empty_source_filter_returns_empty():
    from src.ragPipeline.vectorstore import query
    collection = _make_collection()
    collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    with patch("src.ragPipeline.vectorstore.getEmbeddings", return_value=[0.1]):
        result = query("q", collection, "k", "u", "m", source_filter=["nonexistent"])
    assert result["ids"] == [[]]


def test_list_sources():
    from src.ragPipeline.vectorstore import list_sources
    collection = _make_collection()
    collection.get.return_value = {
        "metadatas": [{"source": "a"}, {"source": "b"}, {"source": "a"}]
    }
    sources = list_sources(collection)
    assert {"source": "a", "chunks": 2} in sources
    assert {"source": "b", "chunks": 1} in sources


def test_get_filtered_no_filter():
    from src.ragPipeline.vectorstore import get_filtered
    collection = _make_collection()
    collection.get.return_value = {"ids": ["id1"], "documents": ["d"], "metadatas": [{}]}
    result = get_filtered(collection)
    collection.get.assert_called_once_with()


def test_get_filtered_with_filter():
    from src.ragPipeline.vectorstore import get_filtered
    collection = _make_collection()
    collection.get.return_value = {"ids": [], "documents": [], "metadatas": []}
    result = get_filtered(collection, source_filter=["src1"])
    collection.get.assert_called_once()
