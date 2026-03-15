def check_relevance(scores, threshold):
    """FAISS L2 distance: lower = more similar, reject if best score exceeds threshold."""
    return min(scores) <= threshold
