from dotenv import load_dotenv
import os
import yaml
from pathlib import Path

from ragPipeline.vectorstore import get_collection, ingest, query


def setup():
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY missing in .env")

    config_path = Path(__file__).parent.parent / "config" / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_name = config["embeddings-model"]["name"]
    api_url = "https://openrouter.ai/api/v1/embeddings"

    return api_key, api_url, model_name


def main():
    api_key, api_url, model_name = setup()

    db_path = str(Path(__file__).parent.parent / "chroma_db")
    collection = get_collection(db_path=db_path)

    mode = input("Mode (ingest/query): ").strip().lower()

    if mode == "ingest":
        file_path = input("PDF path: ").strip()
        ingest(file_path, collection, api_key, api_url, model_name)

    elif mode == "query":
        question = input("Question: ").strip()
        results = query(question, collection, api_key, api_url, model_name)
        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            print(f"\n[{i+1}] Source: {meta['source']} | Page: {meta['page']}")
            print(doc)

    else:
        print("Unknown mode. Use 'ingest' or 'query'.")


if __name__ == "__main__":
    main()
