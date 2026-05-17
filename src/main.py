from dotenv import load_dotenv
import os
import yaml
from pathlib import Path

from ragPipeline.cosineSim import cosineSimilarity
from ragPipeline.embeddings import getEmbeddings


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


def get_embedding_similarity(q1, q2, api_key, api_url, model_name):
    e1 = getEmbeddings(api_key, api_url, model_name, q1)
    e2 = getEmbeddings(api_key, api_url, model_name, q2)
    return cosineSimilarity(e1, e2)


def main():
    api_key, api_url, model_name = setup()

    q1 = input("Enter the first sentence: ")
    q2 = input("Enter the second sentence: ")

    similarity = get_embedding_similarity(
        q1,
        q2,
        api_key,
        api_url,
        model_name
    )

    print(f"Cosine similarity: {similarity}")


if __name__ == "__main__":
    main()