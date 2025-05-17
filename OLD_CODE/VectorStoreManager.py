from typing import List, Dict
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

class VectorStoreManager:
    def __init__(self, embedding_function):
        self.embedding_function = embedding_function
        self.vector_store = None

    def create_vector_store(self, documents: List[Document]):
        self.vector_store = FAISS.from_documents(documents, self.embedding_function)

    def get_retriever(self, search_kwargs: Dict[str, int] = {"k": 2}):
        if self.vector_store:
            return self.vector_store.as_retriever(search_kwargs=search_kwargs)
        return None