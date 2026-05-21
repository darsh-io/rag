def reciprocal_rank_fusion(ranked_lists, k=60, top_n=10):
    """Fuse multiple ranked lists via RRF and return the top_n as (rank, doc, meta, score) tuples."""
    scores = {}
    items = {}

    for ranked_list in ranked_lists:
        for item in ranked_list:
            doc_id = item["id"]
            # k=60 is the standard constant from the original RRF paper — dampens the impact of top ranks
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + item["rank"])
            items[doc_id] = item

    top_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_n]
    return [
        (rank, items[doc_id]["doc"], items[doc_id]["meta"], scores[doc_id])
        for rank, doc_id in enumerate(top_ids, start=1)
    ]
