import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from src.http_client import post_with_retry, post_with_retry_stream
from ragPipeline.vectorstore import query as chroma_query, get_filtered
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

# HyDE prompt asks for a short passage, not a full answer — embedding quality degrades with length
HYDE_PROMPT = """\
Write a short technical passage (2-3 sentences) that directly answers the question below.
Write as if the passage appears in a research paper or textbook. No preamble, just the content."""


def generate_hypothetical_doc(question, api_key, chat_url, llm_model):
    """Generate a hypothetical document that would answer the question, used to improve retrieval."""
    messages = [
        {"role": "system", "content": HYDE_PROMPT},
        {"role": "user", "content": question},
    ]
    return call_llm(messages, api_key, chat_url, llm_model)


def _dynamic_cutoff(chunks, min_score=0.05, gap_threshold=0.40):
    """
    Return the subset of reranked chunks that are genuinely relevant.

    Strategy:
    1. Drop anything below min_score (absolute noise floor).
    2. Among what remains, find the largest *relative* score drop between
       consecutive chunks.  If that drop exceeds gap_threshold (e.g. 0.40 = 40%
       of the higher score), cut there — it marks the boundary between the
       relevant cluster and the trailing noise.
    3. Always return at least one chunk.
    """
    if not chunks:
        return chunks

    above_floor = [c for c in chunks if c[3] >= min_score]
    if not above_floor:
        return [chunks[0]]

    if len(above_floor) == 1:
        return above_floor

    max_rel_gap = 0.0
    cut_at = len(above_floor)
    for i in range(1, len(above_floor)):
        prev_score = above_floor[i - 1][3]
        curr_score = above_floor[i][3]
        rel_gap = (prev_score - curr_score) / prev_score if prev_score > 0 else 0.0
        if rel_gap > max_rel_gap:
            max_rel_gap = rel_gap
            cut_at = i

    return above_floor[:cut_at] if max_rel_gap >= gap_threshold else above_floor


def build_rag_context(question, history, collection, api_key, embed_url, embed_model, chat_url, llm_model, cohere_api_key, reranker_model, source_filter=None):
    """Run HyDE retrieval, fusion, and reranking; return (chunks, messages, hyde_doc) ready for the LLM."""
    # HyDE: embed a hypothetical answer rather than the raw question for denser semantic match
    hyde_doc = generate_hypothetical_doc(question, api_key, chat_url, llm_model)

    # Dense retrieval using the hypothetical doc embedding (top 20)
    dense_results = chroma_query(hyde_doc, collection, api_key, embed_url, embed_model, n_results=20, source_filter=source_filter)
    dense_ranked = [
        {"id": id_, "doc": doc, "meta": meta, "rank": rank}
        for rank, (id_, doc, meta) in enumerate(zip(
            dense_results["ids"][0],
            dense_results["documents"][0],
            dense_results["metadatas"][0],
        ), start=1)
    ]

    # Sparse BM25 uses the original question — keyword matching works better on the real query
    filtered_data = get_filtered(collection, source_filter)
    if not filtered_data["ids"]:
        msg = (
            "No searchable content found for this topic. Try deleting the file in Admin → Topics and re-uploading it."
            if source_filter else
            "No documents have been ingested yet. Upload files inside a topic to get started."
        )
        raise ValueError(msg)
    sparse_ranked = bm25_search(question, filtered_data["ids"], filtered_data["documents"], filtered_data["metadatas"], top_n=20)

    # RRF fusion → top 20, Cohere scores all candidates, dynamic cutoff selects final set
    fused = reciprocal_rank_fusion([dense_ranked, sparse_ranked], top_n=20)
    all_reranked = rerank(question, fused, cohere_api_key, reranker_model)
    chunks = _dynamic_cutoff(all_reranked)

    context_block = "\n\n".join(
        f"[{i}] Source: {meta['source']} | Page: {meta['page']}\n{doc}"
        for i, doc, meta, _ in chunks
    )

    # context goes in the user turn so the model sees it alongside the question each time
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"}]
    )

    return chunks, messages, hyde_doc


def call_llm(messages, api_key, chat_url, llm_model):
    """Send a messages list to the OpenRouter chat API and return the assistant's reply."""
    rq = post_with_retry(
        url=chat_url,
        headers={"Authorization": f"Bearer {api_key}"},
        data=json.dumps({"model": llm_model, "messages": messages}),
    )
    body = rq.json()
    if "choices" not in body:
        raise RuntimeError(f"OpenRouter error: {body}")
    message = body["choices"][0]["message"]
    # reasoning models return content=None and put their output in "reasoning"
    return message["content"] or message.get("reasoning")


def call_llm_stream(messages, api_key, chat_url, llm_model):
    """Yield text deltas from the LLM as they arrive using OpenRouter SSE streaming."""
    rq = post_with_retry_stream(
        url=chat_url,
        headers={"Authorization": f"Bearer {api_key}"},
        data=json.dumps({"model": llm_model, "messages": messages, "stream": True}),
    )
    for line in rq.iter_lines():
        if not line:
            continue
        text = line.decode("utf-8") if isinstance(line, bytes) else line
        if not text.startswith("data: "):
            continue
        payload = text[6:]
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
            if "error" in chunk:
                raise RuntimeError(f"OpenRouter: {chunk['error'].get('message', str(chunk['error']))}")
            d = chunk["choices"][0]["delta"]
            text = d.get("content") or d.get("reasoning_content") or d.get("reasoning") or ""
            if text:
                yield text
        except RuntimeError:
            raise
        except (json.JSONDecodeError, KeyError, IndexError):
            continue


def rag_query(question, history, collection, api_key, embed_url, embed_model, chat_url, llm_model, cohere_api_key, reranker_model, source_filter=None):
    """Run the full RAG pipeline with HyDE — retrieve, fuse, rerank, then answer with conversation history."""
    chunks, messages, hyde_doc = build_rag_context(
        question, history, collection,
        api_key, embed_url, embed_model,
        chat_url, llm_model,
        cohere_api_key, reranker_model,
        source_filter=source_filter,
    )
    answer = call_llm(messages, api_key, chat_url, llm_model)

    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  QUESTION: {question}")
    print(sep)
    print(f"\n  HYPOTHETICAL DOC (HyDE)\n  {hyde_doc.strip()}\n")
    print(f"\n  RETRIEVED CHUNKS (dynamic cutoff: {len(chunks)} selected)\n")
    for i, doc, meta, score in chunks:
        preview = doc[:200].replace("\n", " ")
        print(f"  [{i}] {meta['source']} | p.{meta['page']} | relevance: {score:.4f}")
        print(f"      {preview}…")
        print()
    print(sep)
    print("\n  ANSWER\n")
    print(f"  {answer.strip()}")
    print(f"\n{sep}\n")

    return answer, chunks
