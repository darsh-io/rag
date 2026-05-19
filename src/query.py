from dotenv import load_dotenv
import os
import yaml
import requests
import json
from pathlib import Path

from ragPipeline.vectorstore import get_collection, query as chroma_query
from ragPipeline.reranker import rerank


PROMPT_TEMPLATE = """\
You are a research assistant. Answer the question using ONLY the provided context chunks.
For every claim you make, cite the source inline as [Source: <filename>, Page: <n>].
If the context does not contain enough information to answer, say so explicitly.

Context:
{context}

Question: {question}

Answer (with inline citations):"""


def build_context(results):
    parts = []
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        similarity = 1 - dist  # Chroma cosine distance → similarity
        parts.append((i + 1, doc, meta, similarity))
    return parts


def call_llm(prompt, api_key, chat_url, llm_model):
    rq = requests.post(
        url=chat_url,
        headers={"Authorization": f"Bearer {api_key}"},
        data=json.dumps({
            "model": llm_model,
            "messages": [{"role": "user", "content": prompt}],
        }),
    )
    body = rq.json()
    if "choices" not in body:
        raise RuntimeError(f"OpenRouter error: {body}")
    message = body["choices"][0]["message"]
    return message["content"] or message.get("reasoning")


def rag_query(question, collection, api_key, embed_url, embed_model, chat_url, llm_model, cohere_api_key, reranker_model):
    results = chroma_query(question, collection, api_key, embed_url, embed_model, n_results=10)
    candidates = build_context(results)
    chunks = rerank(question, candidates, cohere_api_key, reranker_model, top_n=5)

    context_block = "\n\n".join(
        f"[{i}] Source: {meta['source']} | Page: {meta['page']}\n{doc}"
        for i, doc, meta, _ in chunks
    )
    prompt = PROMPT_TEMPLATE.format(context=context_block, question=question)
    answer = call_llm(prompt, api_key, chat_url, llm_model)

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

    embed_model = config["embeddings-model"]["name"]
    llm_model = config["llm-model"]["name"]
    reranker_model = config["reranker-model"]["name"]
    embed_url = "https://openrouter.ai/api/v1/embeddings"
    chat_url = "https://openrouter.ai/api/v1/chat/completions"

    db_path = str(Path(__file__).parent.parent / "chroma_db")
    collection = get_collection(db_path=db_path)

    question = input("Question: ").strip()
    rag_query(question, collection, api_key, embed_url, embed_model, chat_url, llm_model, cohere_api_key, reranker_model)
