import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv
load_dotenv()

from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import FAISS
from embeddings import get_embedding_model
from retriever import retrieve
from config import FAISS_INDEX_PATH, TOP_K


def recall_at_k(retrieved_docs, question, k=TOP_K):
    """
    Proxy Recall@k: checks if any retrieved chunk contains keywords from the question.
    In a real setup you'd match against gold document IDs.
    """
    keywords = set(question.lower().split())
    for doc in retrieved_docs[:k]:
        if keywords & set(doc.page_content.lower().split()):
            return 1.0
    return 0.0


def answer_similarity(embed, pred, ref):
    v1 = embed.embed_query(pred)
    v2 = embed.embed_query(ref)
    return cosine_similarity([v1], [v2])[0][0]


def run_eval():
    with open("data/qa_dataset.json") as f:
        qa_pairs = json.load(f)

    embed = get_embedding_model()
    vs = FAISS.load_local(FAISS_INDEX_PATH, embed, allow_dangerous_deserialization=True)

    from langchain_openai import ChatOpenAI
    from rag_pipeline import run_rag

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    recalls, similarities = [], []

    for item in qa_pairs:
        question = item["question"]
        reference = item["answer"]

        docs, _ = retrieve(vs, question)
        recall = recall_at_k(docs, question)
        recalls.append(recall)

        answer, _ = run_rag(question, vs, llm, [])
        sim = answer_similarity(embed, answer, reference)
        similarities.append(sim)

        print(f"Q: {question}")
        print(f"  Recall@{TOP_K}: {recall:.2f}  |  Cosine similarity: {sim:.3f}")

    print(f"\nMean Recall@{TOP_K}: {sum(recalls)/len(recalls):.3f}")
    print(f"Mean Answer Cosine Similarity: {sum(similarities)/len(similarities):.3f}")


if __name__ == "__main__":
    run_eval()
