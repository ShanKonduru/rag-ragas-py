import streamlit as st
from typing import List, Tuple
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ContextRelevance,
    ContextRecall,
    ContextPrecision
)
import pandas as pd
import tempfile  # For handling uploaded files
from collections import deque

class RAGPipeline:
    """
    A class for building and evaluating a Retrieval-Augmented Generation (RAG) pipeline.
    """

    def __init__(self, openai_api_key: str, chunk_size: int = 500, chunk_overlap: int = 50, llm_temperature: float = 0.6):
        """
        Initializes the RAGPipeline.

        Args:
            openai_api_key: Your OpenAI API key.
            chunk_size: The size of text chunks for splitting.
            chunk_overlap: The overlap between text chunks.
            llm_temperature: The temperature setting for the language model.
        """
        self.openai_api_key = openai_api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm_temperature = llm_temperature
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.llm = ChatOpenAI(temperature=self.llm_temperature,
                              openai_api_key=self.openai_api_key)
        self.vector_db = None
        self.retriever = None
        self.chain = None
        self.history = deque(maxlen=5)  # Keep a history of last 5 interactions

    def load_knowledge_base(self, file_path: str):
        """
        Loads the knowledge base from a text file.

        Args:
            file_path: The path to the text file.
        """
        loader = TextLoader(file_path, encoding="utf-8")
        self.document = loader.load()
        return self.document

    def load_knowledge_base_directory(self, dir_path: str):
        """Loads all text files from a directory."""
        loader = DirectoryLoader(
            dir_path, glob="**/*.txt", 
            loader_cls=TextLoader, 
            show_progress=True, 
            use_multithreading=True, 
            loader_kwargs={'encoding': 'utf-8'})
        self.document = loader.load()
        return self.document

    def chunk_data(self, document):
        """
        Chunks the loaded document into smaller pieces.

        Args:
            document: The loaded Langchain Document.

        Returns:
            A list of Langchain Documents representing the chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.chunks = text_splitter.split_documents(document)
        return self.chunks

    def create_vector_database(self, chunks: List):
        """
        Creates a vector database from the text chunks.

        Args:
            chunks: A list of Langchain Documents.
        """
        self.vector_db = FAISS.from_documents(chunks, self.embeddings)
        self.retriever = self.vector_db.as_retriever()

    def save_vector_store(self, save_path: str, index: str):
        """
        Saves the FAISS vector store to the local file system.

        Args:
            save_path: The directory where the vector store files will be saved.
        """
        if self.vector_db:
            self.vector_db.save_local(save_path, index)
            print(f"\nVector store saved to: {save_path}")
        else:
            print("\nVector store not initialized. Cannot save.")

    def load_vector_store(self, load_path: str):
        """
        Loads a FAISS vector store from the local file system.

        Args:
            load_path: The directory containing the saved vector store files.
        """
        try:
            self.vector_db = FAISS.load_local(load_path, self.embeddings)
            self.retriever = self.vector_db.as_retriever()
            print(f"\nVector store loaded from: {load_path}")
        except Exception as e:
            print(f"\nError loading vector store from {load_path}: {e}")
            self.vector_db = None
            self.retriever = None

    def setup_rag_pipeline(self, prompt_template: str):
        """
        Sets up the Retrieval-Augmented Generation (RAG) pipeline.

        Args:
            prompt_template: The prompt template to use for the RAG pipeline.
        """
        prompt = ChatPromptTemplate.from_template(template=prompt_template)
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str) -> str:
        """
        Queries the RAG pipeline with a question and updates history.

        Args:
            question: The question to ask.

        Returns:
            The answer generated by the RAG pipeline.
        """
        if self.chain is None:
            raise ValueError(
                "RAG pipeline not set up. Call setup_rag_pipeline() first.")
        answer = self.chain.invoke(question)
        # Store interaction
        self.history.append({"question": question, "answer": answer})
        return answer

    def generate_questions_and_ground_truths(self, document, num_questions: int = 5) -> Tuple[List[str], List[str]]:
        """
        Generates a list of questions and corresponding ground truth answers based on the document.
        This is a simplified example and might require a more sophisticated approach
        depending on the complexity of the knowledge base.

        Args:
            document: The loaded Langchain Document.
            num_questions: The number of questions and ground truths to generate.

        Returns:
            A tuple containing two lists: questions and ground_truths.
        """
        # This is a placeholder and would ideally involve a more advanced method
        # potentially using an LLM to generate relevant questions and answers
        # directly from the document content.
        # For now, we'll just take the first few sentences as potential questions
        # and their immediate next sentences as ground truths (very basic).
        questions = []
        ground_truths = []
        if not document:
            return [], []
        all_content = document[0].page_content.split('.')
        for i in range(min(num_questions, len(all_content) - 1)):
            questions.append(all_content[i].strip() + "?")
            ground_truths.append(all_content[i+1].strip() + ".")
        return questions, ground_truths

    def evaluate_rag(self, questions: List[str], ground_truths: List[str], answers: List[str], contexts: List[List[str]]) -> pd.DataFrame:
        """
        Evaluates the RAG pipeline using Ragas metrics, now accepts answers and contexts.

        Args:
            questions: A list of questions.
            ground_truths: A list of corresponding ground truth answers.
            answers: A list of answers
            contexts: A list of retrieved contexts

        Returns:
            A Pandas DataFrame containing the evaluation results.
        """
        if self.retriever is None:
            raise ValueError("Vector database and retriever not initialized.")

        data = {
            "question": questions,
            "ground_truths": ground_truths,
            "answer": answers,  # Pass provided answers
            "contexts": contexts,  # Pass provided contexts
            "reference": ground_truths, # Added reference
        }
        dataset = Dataset.from_dict(data)

        result = evaluate(
            dataset=dataset,
            metrics=[
                Faithfulness(),
                ContextRelevance(),
                ContextRecall(),
                ContextPrecision()
            ],
            llm=self.llm,
            embeddings=self.embeddings
        )
        return result.to_pandas()

    def save_evaluation_to_csv(self, df: pd.DataFrame, file_path: str = "rag_evaluation_results.csv"):
        """
        Saves the evaluation DataFrame to a CSV file.

        Args:
            df: The Pandas DataFrame containing the evaluation results.
            file_path: The path where the CSV file should be saved. Defaults to "rag_evaluation_results.csv".
        """
        df.to_csv(file_path, index=False)
        print(f"\nEvaluation results saved to: {file_path}")
