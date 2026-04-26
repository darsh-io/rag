from similarity import cosine_similarity
from embedding import embed_texts

def retrieve(query, chunks, embeds, k=3):
    query_embed = embed_texts([query])[0]

    scores = []
    for i, emb in enumerate(embeds):
        sim = cosine_similarity(query_embed, emb)
        scores.append((sim, i))

    scores.sort(reverse=True)

    return [chunks[i] for _, i in scores[:k]]