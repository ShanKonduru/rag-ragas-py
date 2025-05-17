import pytest
from DataLoader import DataLoader
from Embedder import Embedder
from VectorStoreManager import VectorStoreManager
from RAG_Pipeline import RAGPipeline
from Evaluator import Evaluator

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from ragas.metrics import (
    answer_relevancy,
    faithfulness
)

# Sample data for testing
sample_documents_text = [
    "The sky is blue.",
    "Grass is green.",
]
sample_eval_data = [
    {"question": "What color is the sky?", "answer": "blue", "contexts": ["The sky is blue."], "ground_truths": ["blue"]},
    {"question": "What color is the grass?", "answer": "greenish", "contexts": ["Grass is green."], "ground_truths": ["green"]},
]

@pytest.fixture
def data_loader():
    return DataLoader(sample_documents_text)

@pytest.fixture
def embedder():
    return Embedder()

@pytest.fixture
def vector_store_manager(embedder, data_loader):
    manager = VectorStoreManager(embedder.get_embeddings())
    manager.create_vector_store(data_loader.load_documents())
    return manager

@pytest.fixture
def ollama_llm():
    # Mock Ollama for testing if you don't want to rely on a running instance
    class MockOllama:
        def __call__(self, prompt):
            if "capital of" in prompt.lower():
                return "Ottawa"
            elif "calgary" in prompt.lower():
                return "known for stampede"
            else:
                return "I don't know"
    return MockOllama()

@pytest.fixture
def prompt_template():
    return PromptTemplate(
        template="Answer the question based on the context: {context}\nQuestion: {question}",
        input_variables=["context", "question"],
    )

@pytest.fixture
def rag_pipeline(ollama_llm, vector_store_manager, prompt_template):
    retriever = vector_store_manager.get_retriever()
    return RAGPipeline(ollama_llm, retriever, prompt_template)

@pytest.fixture
def evaluator(ollama_llm):
    metrics = [faithfulness(llm=ollama_llm), answer_relevancy(llm=ollama_llm)]
    return Evaluator(metrics=metrics, llm=ollama_llm)

def test_data_loader(data_loader):
    chunks = data_loader.load_documents()
    assert len(chunks) == 2
    assert isinstance(chunks[0], Document)

def test_embedder(embedder):
    embeddings = embedder.get_embeddings()
    assert embeddings is not None

def test_vector_store_creation(vector_store_manager, data_loader):
    assert vector_store_manager.vector_store is not None

def test_retriever(vector_store_manager):
    retriever = vector_store_manager.get_retriever()
    assert retriever is not None

def test_rag_pipeline_run(rag_pipeline):
    result = rag_pipeline.run("What color is the sky?")
    assert "result" in result

def test_evaluator(evaluator):
    results_df = evaluator.evaluate(sample_eval_data)
    assert isinstance(results_df, pd.DataFrame)
    assert not results_df.empty
    assert "faithfulness" in results_df.columns
    assert "answer_relevancy" in results_df.columns