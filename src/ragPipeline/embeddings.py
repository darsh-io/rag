import requests
import json


def getEmbeddings(api_key, api_url, model, text):
    rq = requests.post(
        url=api_url,
        headers={
            "Authorization": f"Bearer {api_key}",
        },
        data=json.dumps({
            "model": model,
            "input": text,
        }),
    )
    return rq.json()["data"][0]["embedding"]