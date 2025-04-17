from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import os
import pickle

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

def load_index(index_path):
    with open(f"{index_path}/index.pkl", "rb") as f:
        texts = pickle.load(f)
    return FAISS.load_local(index_path, embeddings)

def retrieve_context(index, query):
    return index.similarity_search(query, k=5)
