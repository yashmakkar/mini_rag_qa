from langchain_openai import OpenAIEmbeddings
from config import EMBEDDING_MODEL


def get_embedding_model():
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)
