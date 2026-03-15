from config import TOP_K


def retrieve(vector_store, query, k=TOP_K):
    results = vector_store.similarity_search_with_score(query, k=k)
    docs = [doc for doc, _ in results]
    scores = [float(score) for _, score in results]
    return docs, scores
