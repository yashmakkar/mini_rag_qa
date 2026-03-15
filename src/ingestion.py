import ssl
import urllib.request
from langchain_community.document_loaders import UnstructuredURLLoader


ssl._create_default_https_context = ssl._create_unverified_context


def load_documents(urls):
    loader = UnstructuredURLLoader(urls=urls, ssl_verify=False)
    documents = loader.load()

    for doc in documents:
        url = doc.metadata["source"]
        doc.metadata["source"] = url.split("/")[-1]
        doc.metadata["url"] = url

    return documents
