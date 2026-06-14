"""Dynamic cutoff unit tests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from query import _dynamic_cutoff


def _chunks(scores):
    return [(i+1, f"doc{i}", {"source": f"s{i}", "page": 1}, s) for i, s in enumerate(scores)]


def test_empty_returns_empty():
    assert _dynamic_cutoff([]) == []


def test_all_below_floor_returns_first():
    chunks = _chunks([0.04, 0.03, 0.01])
    result = _dynamic_cutoff(chunks)
    assert len(result) == 1
    assert result[0][3] == 0.04


def test_clear_gap_cuts_correctly():
    # big gap between index 1 and 2: 0.8 → 0.1 (87% drop)
    chunks = _chunks([0.9, 0.8, 0.1, 0.09])
    result = _dynamic_cutoff(chunks)
    assert len(result) == 2
    assert result[-1][3] == 0.8


def test_no_gap_returns_all_above_floor():
    # scores all close together, no single large gap
    chunks = _chunks([0.9, 0.85, 0.80, 0.75])
    result = _dynamic_cutoff(chunks)
    assert len(result) == 4


def test_single_above_floor_returns_one():
    chunks = _chunks([0.9, 0.01, 0.01])
    result = _dynamic_cutoff(chunks)
    assert len(result) == 1
    assert result[0][3] == 0.9


def test_always_returns_at_least_one():
    chunks = _chunks([0.001])
    result = _dynamic_cutoff(chunks)
    assert len(result) == 1
