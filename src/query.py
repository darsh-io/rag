import requests
import json

from ragPipeline.vectorstore import query as chroma_query
from ragPipeline.reranker import rerank
from ragPipeline.bm25 import bm25_search
from ragPipeline.rrf import reciprocal_rank_fusion


SYSTEM_PROMPT = """\
You are a precise research assistant. Answer questions strictly from the provided context chunks.

Rules:
- Cite every factual claim inline as [Source: <filename>, Page: <n>].
- If the context is insufficient, say "I don't have enough information to answer that."
- Be concise — no filler or padding.
- You may use prior conversation turns to resolve follow-up questions, but still ground all answers in the retrieved context."""


def call_llm(messages, api_key, chat_url, llm_model):
    rq = requests.post(
        url=chat_url,
        headers={"Authorization": f"Bearer {api_key}"},
        data=json.dumps({"model": llm_model, "messages": messages}),
    )
    body = rq.json()
    if "choices" not in body:
        raise RuntimeError(f"OpenRouter error: {body}")
    message = body["choices"][0]["message"]
    return message["content"] or message.get("reasoning")


def rag_query(question, history, collection, api_key, embed_url, embed_model, chat_url, llm_model, cohere_api_key, reranker_model):
    # Dense retrieval (top 10)
    dense_results = chroma_query(question, collection, api_key, embed_url, embed_model, n_results=10)
    dense_ranked = [
        {"id": id_, "doc": doc, "meta": meta, "rank": rank}
        for rank, (id_, doc, meta) in enumerate(zip(
            dense_results["ids"][0],
            dense_results["documents"][0],
            dense_results["metadatas"][0],
        ), start=1)
    ]

    # Sparse BM25 retrieval (top 10)
    all_data = collection.get()
    sparse_ranked = bm25_search(question, all_data["ids"], all_data["documents"], all_data["metadatas"], top_n=10)

    # RRF fusion → top 10, then Cohere rerank → top 5
    fused = reciprocal_rank_fusion([dense_ranked, sparse_ranked], top_n=10)
    chunks = rerank(question, fused, cohere_api_key, reranker_model, top_n=5)

    context_block = "\n\n".join(
        f"[{i}] Source: {meta['source']} | Page: {meta['page']}\n{doc}"
        for i, doc, meta, _ in chunks
    )

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"}]
    )
    answer = call_llm(messages, api_key, chat_url, llm_model)

    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  QUESTION: {question}")
    print(sep)
    print("\n  RETRIEVED CHUNKS (reranked, top 5)\n")
    for i, doc, meta, score in chunks:
        preview = doc[:200].replace("\n", " ")
        print(f"  [{i}] {meta['source']} | p.{meta['page']} | relevance: {score:.4f}")
        print(f"      {preview}…")
        print()
    print(sep)
    print("\n  ANSWER\n")
    print(f"  {answer.strip()}")
    print(f"\n{sep}\n")

    return answer
