from dotenv import load_dotenv
import os
import yaml
from pathlib import Path

from ragPipeline.vectorstore import get_collection, ingest, query
from ragPipeline.llm import answer


def setup():
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY missing in .env")

    config_path = Path(__file__).parent.parent / "config" / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    embed_model = config["embeddings-model"]["name"]
    llm_model = config["llm-model"]["name"]
    embed_url = "https://openrouter.ai/api/v1/embeddings"
    chat_url = "https://openrouter.ai/api/v1/chat/completions"

    return api_key, embed_url, embed_model, chat_url, llm_model


def main():
    api_key, embed_url, embed_model, chat_url, llm_model = setup()

    db_path = str(Path(__file__).parent.parent / "chroma_db")
    collection = get_collection(db_path=db_path)

    mode = input("Mode (ingest/query): ").strip().lower()

    if mode == "ingest":
        file_path = input("PDF path: ").strip().strip('"')
        ingest(file_path, collection, api_key, embed_url, embed_model)

    elif mode == "query":
        question = input("Question: ").strip()
        results = query(question, collection, api_key, embed_url, embed_model)
        response = answer(question, results, api_key, chat_url, llm_model)
        print(f"\nAnswer: {response}")

    else:
        print("Unknown mode. Use 'ingest' or 'query'.")


if __name__ == "__main__":
    main()
