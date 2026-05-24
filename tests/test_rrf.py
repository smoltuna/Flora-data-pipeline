"""Unit tests for Reciprocal Rank Fusion."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from services.rag.retriever import RetrievedChunk, reciprocal_rank_fusion


def _chunk(cid: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, chunk_text=f"chunk {cid}", source="test", rrf_score=0.0, embedding=[]
    )


def test_rrf_single_list():
    """Single list returns items in the same order."""
    lst = [_chunk(1), _chunk(2), _chunk(3)]
    result = reciprocal_rank_fusion([lst])
    assert [c.chunk_id for c, _ in result] == [1, 2, 3]


def test_rrf_two_identical_lists():
    """Same list twice → same order, higher scores than a single list."""
    lst = [_chunk(1), _chunk(2), _chunk(3)]
    single = reciprocal_rank_fusion([lst])
    double = reciprocal_rank_fusion([lst, lst])

    assert [c.chunk_id for c, _ in double] == [c.chunk_id for c, _ in single]
    for (_, s_score), (_, d_score) in zip(single, double):
        assert d_score > s_score


def test_rrf_disjoint_lists():
    """Non-overlapping lists are interleaved by rank position."""
    # chunk 1 and 3 both occupy rank 0 in their respective lists (score 1/60 each)
    # chunk 2 and 4 both occupy rank 1 (score 1/61 each)
    lst_a = [_chunk(1), _chunk(2)]
    lst_b = [_chunk(3), _chunk(4)]
    result = reciprocal_rank_fusion([lst_a, lst_b])
    ids = [c.chunk_id for c, _ in result]
    assert set(ids[:2]) == {1, 3}
    assert set(ids[2:]) == {2, 4}


def test_rrf_overlap_boosts_rank():
    """Item appearing in 2 lists ranks higher than an item in only 1 list."""
    # chunk 2: rank 1 in lst_a + rank 0 in lst_b → 1/61 + 1/60 ≈ 0.0331
    # chunk 1: rank 0 in lst_a only              → 1/60 ≈ 0.0167
    lst_a = [_chunk(1), _chunk(2)]
    lst_b = [_chunk(2), _chunk(3)]
    result = reciprocal_rank_fusion([lst_a, lst_b])
    assert result[0][0].chunk_id == 2


def test_rrf_k_parameter():
    """Higher k flattens the score distribution (smaller gap between best and worst)."""
    lst = [_chunk(1), _chunk(2), _chunk(3)]
    scores_k10 = [s for _, s in reciprocal_rank_fusion([lst], k=10)]
    scores_k100 = [s for _, s in reciprocal_rank_fusion([lst], k=100)]

    diff_k10 = scores_k10[0] - scores_k10[-1]
    diff_k100 = scores_k100[0] - scores_k100[-1]
    assert diff_k10 > diff_k100
