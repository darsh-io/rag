from dotenv import load_dotenv
import requests
import json
import os
from cosineSim import cosineSimilarity
def setup():
    global OPENROUTER_API_KEY, OPENROUTER_API_URL
    load_dotenv()
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1/embeddings"

setup()

q1 = input("Enter the first sentence: ")
q2 = input("Enter the second sentence: ")

rq1 = requests.post(
    url=OPENROUTER_API_URL,
    headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    },
    data=json.dumps({
        "model": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
        "input": q1
    })
    )

rq2 = requests.post(
    url=OPENROUTER_API_URL,
    headers={
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    },
    data=json.dumps({
        "model": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
        "input": q2
    })
)

print(cosineSimilarity(rq1.json()["data"][0]["embedding"], rq2.json()["data"][0]["embedding"]))