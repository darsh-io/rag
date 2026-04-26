from litellm import embedding

def embed_texts(texts):
    res = embedding(
        model="openrouter/text-embedding-3-small",
        input=texts
    )
    return [e["embedding"] for e in res["data"]]