from typing import Dict
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class RAGPipeline:
    def __init__(self, llm, retriever, prompt: PromptTemplate):
        self.llm = llm
        self.retriever = retriever
        self.prompt = prompt
        self.chain = RetrievalQA.from_llm(
            self.llm,
            retriever=self.retriever,
            prompt=self.prompt,
            return_source_documents=True,
        )

    def run(self, query: str) -> Dict:
        return self.chain({"query": query})