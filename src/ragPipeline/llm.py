import requests
import json


def answer(question, chunks, api_key, api_url, model):
    context = "\n\n".join(
        f"[Source: {m['source']} | Page: {m['page']}]\n{doc}"
        for doc, m in zip(chunks["documents"][0], chunks["metadatas"][0])
    )

    prompt = f"""You are a helpful assistant. Answer the question using only the provided context.

Context:
{context}

Question: {question}
Answer:"""

    rq = requests.post(
        url=api_url,
        headers={"Authorization": f"Bearer {api_key}"},
        data=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }),
    )
    message = rq.json()["choices"][0]["message"]
    return message["content"] or message.get("reasoning")
