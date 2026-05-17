def cosineSimilarity(a, b):
    dot = 0

    for i in range(len(a)):
        dot += a[i] * b[i]
    
    ma = 0
    mb = 0

    for i in range(len(a)):
        ma += a[i] * a[i]
        mb += b[i] * b[i]
    ma **= 0.5
    mb **= 0.5

    if ma == 0 or mb == 0:
        return 0
    
    return dot / (ma * mb)
