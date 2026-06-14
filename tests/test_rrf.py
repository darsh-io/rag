"""RRF unit tests."""
from src.ragPipeline.rrf import reciprocal_rank_fusion


def _make_list(ids_with_ranks):
    return [{"id": id_, "doc": f"doc {id_}", "meta": {}, "rank": rank}
            for id_, rank in ids_with_ranks]


def test_rrf_merges_lists():
    a = _make_list([("x", 1), ("y", 2)])
    b = _make_list([("y", 1), ("z", 2)])
    fused = reciprocal_rank_fusion([a, b], top_n=3)
    assert len(fused) == 3
    # y appears in both lists so should rank first
    assert fused[0][1] == "doc y"


def test_rrf_top_n_cap():
    a = _make_list([(f"d{i}", i+1) for i in range(10)])
    b = _make_list([(f"d{i}", i+1) for i in range(10)])
    fused = reciprocal_rank_fusion([a, b], top_n=5)
    assert len(fused) == 5


def test_rrf_single_list():
    a = _make_list([("a", 1), ("b", 2), ("c", 3)])
    fused = reciprocal_rank_fusion([a], top_n=2)
    assert len(fused) == 2
    assert fused[0][1] == "doc a"
