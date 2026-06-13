import requests
import json

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


def build_rag_context(question, history, collection, api_key, embed_url, embed_model, chat_url, llm_model, cohere_api_key, reranker_model, top_k=5, source_filter=None):
    """Run HyDE retrieval, fusion, and reranking; return (chunks, messages, hyde_doc) ready for the LLM."""
    # HyDE: embed a hypothetical answer rather than the raw question for denser semantic match
    hyde_doc = generate_hypothetical_doc(question, api_key, chat_url, llm_model)

    # Dense retrieval using the hypothetical doc embedding (top 10)
    dense_results = chroma_query(hyde_doc, collection, api_key, embed_url, embed_model, n_results=10, source_filter=source_filter)
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
            "This topic's files aren't loaded — documents are stored in memory and cleared on server restart. "
            "Re-upload them in Admin → Topics."
            if source_filter else
            "No documents have been ingested yet. Upload files inside a topic to get started."
        )
        raise ValueError(msg)
    sparse_ranked = bm25_search(question, filtered_data["ids"], filtered_data["documents"], filtered_data["metadatas"], top_n=10)

    # RRF fusion → top 10, then Cohere rerank → top_k
    fused = reciprocal_rank_fusion([dense_ranked, sparse_ranked], top_n=10)
    chunks = rerank(question, fused, cohere_api_key, reranker_model, top_n=top_k)

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
    rq = requests.post(
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
    rq = requests.post(
        url=chat_url,
        headers={"Authorization": f"Bearer {api_key}"},
        data=json.dumps({"model": llm_model, "messages": messages, "stream": True}),
        stream=True,
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


def rag_query(question, history, collection, api_key, embed_url, embed_model, chat_url, llm_model, cohere_api_key, reranker_model, top_k=5, source_filter=None):
    """Run the full RAG pipeline with HyDE — retrieve, fuse, rerank, then answer with conversation history."""
    chunks, messages, hyde_doc = build_rag_context(
        question, history, collection,
        api_key, embed_url, embed_model,
        chat_url, llm_model,
        cohere_api_key, reranker_model, top_k,
        source_filter=source_filter,
    )
    answer = call_llm(messages, api_key, chat_url, llm_model)

    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  QUESTION: {question}")
    print(sep)
    print(f"\n  HYPOTHETICAL DOC (HyDE)\n  {hyde_doc.strip()}\n")
    print(f"\n  RETRIEVED CHUNKS (reranked, top {top_k})\n")
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
