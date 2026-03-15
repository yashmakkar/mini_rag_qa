from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

MAX_HISTORY_TURNS = 6

SYSTEM_PROMPT = """You are a PostgreSQL documentation assistant.
Answer the question using the provided context.
If the context contains relevant information, use it to give a helpful answer even if the wording differs slightly.
Only say "I don't know" if the context has absolutely no relevant information.
Always cite the source filenames used in your answer."""


def build_prompt(question, context, history):
    messages = [SystemMessage(content=f"{SYSTEM_PROMPT}\n\nContext:\n{context}")]

    for msg in history[-MAX_HISTORY_TURNS:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=question))
    return messages
