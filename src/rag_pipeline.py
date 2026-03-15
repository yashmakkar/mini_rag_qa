from langchain_core.messages import SystemMessage, HumanMessage
from retriever import retrieve
from prompt_builder import build_prompt
from guardrails import check_relevance
from config import SIMILARITY_THRESHOLD


def _contextualize_query(question, chat_history, llm):
    """Rewrite the question to be standalone using chat history."""
    if not chat_history:
        return question

    history_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in chat_history[-6:]
    )
    messages = [
        SystemMessage(content=(
            "Given the conversation history and a follow-up question, "
            "rewrite the follow-up as a standalone question. "
            "Return ONLY the rewritten question, nothing else."
        )),
        HumanMessage(content=f"History:\n{history_text}\n\nFollow-up: {question}"),
    ]
    return llm.invoke(messages).content.strip()


def run_rag(question, vector_store, llm, chat_history):
    standalone_query = _contextualize_query(question, chat_history, llm)

    docs, scores = retrieve(vector_store, standalone_query)

    if not check_relevance(scores, SIMILARITY_THRESHOLD):
        return "I can only answer questions about PostgreSQL documentation.", []

    context = "\n\n".join(
        f"[source: {doc.metadata.get('url', doc.metadata.get('source', 'unknown'))}]\n{doc.page_content}"
        for doc in docs
    )

    messages = build_prompt(question, context, chat_history)
    answer = llm.invoke(messages)

    return answer.content, docs
