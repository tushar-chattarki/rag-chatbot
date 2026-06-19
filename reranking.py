def ReRanking(query, nodes, top_k):
    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    pairs = [(query, node.get_content()) for node in nodes] 
    scores = reranker.predict(pairs) 

    reranked = sorted(
        zip(pairs, scores),
        key= lambda x: x[1],
        reverse=True
    )

    return reranked[:top_k]


def BGE_ReRanker(query, nodes, top_k):
    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(
        "BAAI/bge-reranker-base",
        device="cpu" 
    )

    texts = [node.get_content() for node in nodes]
    pairs = [(query, text) for text in texts]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(nodes, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_k]
