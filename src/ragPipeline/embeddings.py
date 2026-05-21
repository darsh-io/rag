import requests
import json


def getEmbeddings(api_key, api_url, model, text):
    """Return the embedding vector for text using the given OpenRouter model."""
    rq = requests.post(
        url=api_url,
        headers={
            "Authorization": f"Bearer {api_key}",
        },
        # OpenRouter follows the OpenAI embeddings schema: "input" not "text"
        data=json.dumps({
            "model": model,
            "input": text,
        }),
    )
    # response is a list; we always embed one string so index 0 is the only entry
    return rq.json()["data"][0]["embedding"]
