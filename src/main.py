from dotenv import load_dotenv
import os
import yaml
from pathlib import Path

from ragPipeline.vectorstore import get_collection, ingest
from query import rag_query


def setup():
    """Load env vars and config.yaml, returning all credentials and model settings as a dict."""
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY missing in .env")
    if not cohere_api_key:
        raise ValueError("COHERE_API_KEY missing in .env")

    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    return {
        "api_key": api_key,
        "cohere_api_key": cohere_api_key,
        "embed_model": config["embeddings-model"]["name"],
        "llm_model": config["llm-model"]["name"],
        "reranker_model": config["reranker-model"]["name"],
        "embed_url": "https://openrouter.ai/api/v1/embeddings",
        "chat_url": "https://openrouter.ai/api/v1/chat/completions",
    }


def run_ingest(cfg, collection):
    """Prompt for a PDF path and ingest it into the collection."""
    file_path = input("PDF path: ").strip().strip('"')
    ingest(file_path, collection, cfg["api_key"], cfg["embed_url"], cfg["embed_model"])


def run_query_loop(cfg, collection):
    """Run an interactive question loop with persistent history until /quit is entered."""
    history = []
    print("Query mode — /quit to exit, /clear to reset history.\n")

    while True:
        question = input("Question: ").strip()

        if question == "/quit":
            break
        if question == "/clear":
            history.clear()
            print("History cleared.\n")
            continue
        if not question:
            continue

        answer = rag_query(
            question, history, collection,
            cfg["api_key"], cfg["embed_url"], cfg["embed_model"],
            cfg["chat_url"], cfg["llm_model"],
            cfg["cohere_api_key"], cfg["reranker_model"],
        )

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


def main():
    cfg = setup()
    db_path = str(Path(__file__).parent.parent / "chroma_db")
    collection = get_collection(db_path=db_path)

    while True:
        mode = input("\nMode (ingest/query/quit): ").strip().lower()
        if mode == "quit":
            break
        elif mode == "ingest":
            run_ingest(cfg, collection)
        elif mode == "query":
            run_query_loop(cfg, collection)
        else:
            print("Unknown mode. Use 'ingest', 'query', or 'quit'.")


if __name__ == "__main__":
    main()
