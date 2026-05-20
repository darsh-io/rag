def reciprocal_rank_fusion(ranked_lists, k=60, top_n=10):
    """
    ranked_lists: list of lists, each item is {"id", "doc", "meta", "rank"}
    returns: top_n as [(rank, doc, meta, rrf_score)]
    """
    scores = {}
    items = {}

    for ranked_list in ranked_lists:
        for item in ranked_list:
            doc_id = item["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + item["rank"])
            items[doc_id] = item

    top_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_n]
    return [
        (rank, items[doc_id]["doc"], items[doc_id]["meta"], scores[doc_id])
        for rank, doc_id in enumerate(top_ids, start=1)
    ]
