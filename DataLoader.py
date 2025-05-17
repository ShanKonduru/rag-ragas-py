from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

class DataLoader:
    def __init__(self, documents: List[str]):
        self.documents = documents

    def load_documents(self) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=10
        )
        chunks = text_splitter.create_documents(self.documents)
        return chunks
    