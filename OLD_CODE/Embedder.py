from langchain_community.embeddings import SentenceTransformerEmbeddings

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embeddings = SentenceTransformerEmbeddings(model_name=self.model_name)

    def get_embeddings(self):
        return self.embeddings