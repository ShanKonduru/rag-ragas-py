import streamlit as st
import pandas as pd
from DataLoader import DataLoader
from Embedder import Embedder
from VectorStoreManager import VectorStoreManager
from RAG_Pipeline import RAGPipeline
from Evaluator import Evaluator

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
    ContextPrecision,
    AnswerCorrectness,
    AnswerSimilarity,
)
from typing import List, Dict

# --- Initialize Components (can be moved to a setup function if needed) ---
@st.cache_resource
def load_data():
    documents = [
        "The capital of Canada is Ottawa. It is located in the province of Ontario.",
        "Calgary is a major city in Alberta, known for the Calgary Stampede.",
        "The Rocky Mountains are a significant mountain range in western North America.",
        "Maple syrup is a famous Canadian delicacy, often produced in Quebec.",
        "Ice hockey is a popular sport in Canada, with the NHL having several Canadian teams.",
    ]
    data_loader = DataLoader(documents)
    return data_loader.load_documents()

@st.cache_resource
def load_embeddings():
    embedder = Embedder()
    return embedder.get_embeddings()

@st.cache_resource
def setup_vector_store(_documents, _embeddings):
    vector_store_manager = VectorStoreManager(_embeddings)
    vector_store_manager.create_vector_store(_documents)
    return vector_store_manager

@st.cache_resource
def load_llm():
    return Ollama(base_url="http://localhost:11434", model="llama2:13b")

@st.cache_resource
def create_prompt():
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say you don't know.

    {context}

    Question: {question}"""
    return PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

# Load resources
documents = load_data()
embeddings = load_embeddings()
vector_store_manager = setup_vector_store(documents, embeddings)
ollama_llm = load_llm()
prompt = create_prompt()
retriever = vector_store_manager.get_retriever()
rag_pipeline = RAGPipeline(ollama_llm, retriever, prompt)

# --- UI Components ---
st.title("Ask Your Knowledge Base")

user_question = st.text_input("Ask a question:", "")

if user_question:
    with st.spinner("Generating answer..."):
        rag_result = rag_pipeline.run(user_question)
        answer = rag_result["result"]
        contexts = [doc.page_content for doc in rag_result["source_documents"]]

    st.subheader("Answer:")
    st.info(answer)

    st.subheader("Context:")
    for context in contexts:
        st.markdown(f"- {context}")

    if st.button("Evaluate Answer"):
        with st.spinner("Evaluating answer..."):
            predictions = [
                {
                    "question": user_question,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truths": [],  # You might need actual ground truths for some metrics
                }
            ]
            metrics = [
                Faithfulness(llm=ollama_llm),
                AnswerRelevancy(llm=ollama_llm),
                ContextRecall(llm=ollama_llm),
                ContextPrecision(llm=ollama_llm),
                AnswerCorrectness(llm=ollama_llm),
                AnswerSimilarity(),
            ]
            evaluator = Evaluator(metrics=metrics, llm=ollama_llm)
            evaluation_results_df = evaluator.evaluate(predictions)

        st.subheader("Evaluation Results:")
        st.dataframe(evaluation_results_df)

        st.subheader("Understanding the Metrics:")
        st.markdown(
            """
            - **Faithfulness:** How factual is the answer compared to the context? (Higher is better)
            - **Answer Relevancy:** How relevant is the answer to the question? (Higher is better)
            - **Context Recall:** How well does the retrieved context cover the ground truth? (Higher is better)
            - **Context Precision:** How relevant are the retrieved context to the question? (Higher is better)
            - **Answer Correctness:** How accurate is the answer compared to the ground truth (if available)? (Higher is better)
            - **Answer Similarity:** How semantically similar is the answer to the ground truth (if available)? (Higher is better)
            """
        )

# --- Sidebar for Potential Configuration ---
with st.sidebar:
    st.header("Configuration")
    st.markdown("You can add configuration options here in the future (e.g., select different models, adjust retrieval parameters).")