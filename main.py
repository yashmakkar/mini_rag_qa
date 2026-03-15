import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from embeddings import get_embedding_model
from rag_pipeline import run_rag
from config import FAISS_INDEX_PATH


@st.cache_resource
def load_resources():
    embed = get_embedding_model()
    vs = FAISS.load_local(FAISS_INDEX_PATH, embed, allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return vs, llm


st.title("PostgreSQL Documentation Assistant")

vs, llm = load_resources()

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    st.chat_message(msg["role"]).write(msg["content"])

query = st.chat_input("Ask about PostgreSQL...")

if query:
    st.chat_message("user").write(query)

    answer, docs = run_rag(query, vs, llm, st.session_state.history)

    st.chat_message("assistant").write(answer)

    with st.expander("Sources"):
        seen = set()
        for doc in docs:
            url = doc.metadata.get("url", doc.metadata.get("source", "unknown"))
            if url not in seen:
                st.markdown(f"- [{url}]({url})")
                seen.add(url)

    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.history.append({"role": "assistant", "content": answer})
