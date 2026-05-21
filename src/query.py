from dotenv import load_dotenv
import os
import yaml
import requests
import json
from pathlib import Path

from ragPipeline.vectorstore import get_collection, query as chroma_query
from ragPipeline.reranker import rerank
from ragPipeline.bm25 import bm25_search
from ragPipeline.rrf import reciprocal_rank_fusion


SYSTEM_PROMPT = """\
You are a precise research assistant. Answer questions using ONLY the provided context chunks.
Cite every claim inline as [Source: <filename>, Page: <n>].
If the context lacks sufficient information to answer, say so explicitly — do not speculate.
You have access to the conversation history and may use it to resolve follow-up questions."""

USER_PROMPT_TEMPLATE = """\
Context:
{context}

Question: {question}"""


def call_llm(messages, api_key, chat_url, llm_model):
    rq = requests.post(
        url=chat_url,
        headers={"Authorization": f"Bearer {api_key}"},
        data=json.dumps({"model": llm_model, "messages": messages}),
    )
    body = rq.json()
    if "choices" not in body:
        raise RuntimeError(f"OpenRouter error: {body}")
    msg = body["choices"][0]["message"]
    return msg["content"] or msg.get("reasoning")


def rag_query(question, history, collection, all_data, cfg):
    # Dense retrieval (top 10)
    dense_results = chroma_query(
        question, collection,
        cfg["api_key"], cfg["embed_url"], cfg["embed_model"],
        n_results=10,
    )
    dense_ranked = [
        {"id": id_, "doc": doc, "meta": meta, "rank": rank}
        for rank, (id_, doc, meta) in enumerate(zip(
            dense_results["ids"][0],
            dense_results["documents"][0],
            dense_results["metadatas"][0],
        ), start=1)
    ]

    # Sparse BM25 retrieval (top 10) over pre-fetched corpus
    sparse_ranked = bm25_search(
        question,
        all_data["ids"], all_data["documents"], all_data["metadatas"],
        top_n=10,
    )

    # RRF fusion → top 10 candidates, then Cohere rerank → top 5
    fused = reciprocal_rank_fusion([dense_ranked, sparse_ranked], top_n=10)
    chunks = rerank(question, fused, cfg["cohere_api_key"], cfg["reranker_model"], top_n=5)

    context_block = "\n\n".join(
        f"[{i}] Source: {meta['source']} | Page: {meta['page']}\n{doc}"
        for i, doc, meta, _ in chunks
    )

    user_msg = USER_PROMPT_TEMPLATE.format(context=context_block, question=question)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": user_msg}]
    answer = call_llm(messages, cfg["api_key"], cfg["chat_url"], cfg["llm_model"])

    # Append turn to history so follow-up questions have context
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": answer})

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


if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY missing in .env")
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY missing in .env")

    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    cfg = {
        "api_key": api_key,
        "cohere_api_key": cohere_api_key,
        "embed_url": "https://openrouter.ai/api/v1/embeddings",
        "chat_url": "https://openrouter.ai/api/v1/chat/completions",
        "embed_model": config["embeddings-model"]["name"],
        "llm_model": config["llm-model"]["name"],
        "reranker_model": config["reranker-model"]["name"],
    }

    db_path = str(Path(__file__).parent.parent / "chroma_db")
    collection = get_collection(db_path=db_path)
    all_data = collection.get()
    history = []

    print("Query mode — type /quit to exit\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() == "/quit":
            break
        if not question:
            continue
        rag_query(question, history, collection, all_data, cfg)
