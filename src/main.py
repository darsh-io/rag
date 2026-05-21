from dotenv import load_dotenv
import os
import yaml
from pathlib import Path

from ragPipeline.vectorstore import get_collection, ingest
from query import rag_query


def setup():
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

    return {
        "api_key": api_key,
        "cohere_api_key": cohere_api_key,
        "embed_url": "https://openrouter.ai/api/v1/embeddings",
        "chat_url": "https://openrouter.ai/api/v1/chat/completions",
        "embed_model": config["embeddings-model"]["name"],
        "llm_model": config["llm-model"]["name"],
        "reranker_model": config["reranker-model"]["name"],
    }


def query_loop(collection, cfg):
    print("Query mode — conversation history is active. Type /quit to return to menu.\n")
    all_data = collection.get()  # fetch once; reused for BM25 every turn
    history = []
    while True:
        question = input("Question: ").strip()
        if question.lower() == "/quit":
            break
        if not question:
            continue
        rag_query(question, history, collection, all_data, cfg)


def main():
    cfg = setup()
    db_path = str(Path(__file__).parent.parent / "chroma_db")
    collection = get_collection(db_path=db_path)

    print("RAG System — type /quit to exit\n")
    while True:
        mode = input("Mode (ingest/query/quit): ").strip().lower()
        if mode in ("/quit", "quit"):
            print("Bye.")
            break
        elif mode == "ingest":
            file_path = input("PDF path: ").strip().strip('"')
            ingest(file_path, collection, cfg["api_key"], cfg["embed_url"], cfg["embed_model"])
        elif mode == "query":
            query_loop(collection, cfg)
        else:
            print("Unknown mode. Use 'ingest', 'query', or 'quit'.")


if __name__ == "__main__":
    main()
