import os

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity
)

from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

from langchain.prompts import PromptTemplate

from DataLoader import DataLoader
from Embedder import Embedder
from Evaluator import Evaluator
from RAG_Pipeline import RAGPipeline
from VectorStoreManager import VectorStoreManager


def main():
    print("Initializing RAG System...")

    # 1. Load Data
    data_loader = DataLoader(
        [
            "The capital of Canada is Ottawa. It is located in the province of Ontario.",
            "Calgary is a major city in Alberta, known for the Calgary Stampede.",
            "The Rocky Mountains are a significant mountain range in western North America.",
            "Maple syrup is a famous Canadian delicacy, often produced in Quebec.",
            "Ice hockey is a popular sport in Canada, with the NHL having several Canadian teams.",
        ]
    )
    documents = data_loader.load_documents()

    # 2. Embed Data
    embedder = Embedder()
    embeddings = embedder.get_embeddings()

    # 3. Create Vector Store and Retriever
    vector_store_manager = VectorStoreManager(embeddings)
    vector_store_manager.create_vector_store(documents)
    retriever = vector_store_manager.get_retriever()

    # 4. Initialize LLM and Prompt
    ollama_llm = Ollama(base_url="http://localhost:11434", model="llama2:13b")
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say you don't know.

    {context}

    Question: {question}"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # 5. Create RAG Pipeline
    rag_pipeline = RAGPipeline(ollama_llm, retriever, PROMPT)

    # Example Usage
    print("\nRunning Example Queries:")
    queries = [
        "What is the capital of Canada?",
        "Tell me something about Calgary.",
        "What is the highest mountain in Canada?",
    ]
    for query in queries:
        result = rag_pipeline.run(query)
        print(f"Question: {query}")
        print(f"Answer: {result['result']}")

    # 6. Generate Evaluation Data
    eval_data = [
        {
            "question": "What is the capital of Canada?",
            "ground_truths": ["Ottawa"],
        },
        {
            "question": "Tell me something interesting about Calgary.",
            "ground_truths": ["known for the Calgary Stampede"],
        },
        {
            "question": "Where are the Rocky Mountains located?",
            "ground_truths": ["western North America"],
        },
        {
            "question": "What Canadian food is often made in Quebec?",
            "ground_truths": ["Maple syrup"],
        },
        {
            "question": "Which sport is popular in Canada and has NHL teams?",
            "ground_truths": ["Ice hockey"],
        },
        {
            "question": "What is the largest city in Canada?",
            # Ground truth not directly in our context
            "ground_truths": ["Toronto"],
        },
    ]

    # 7. Run RAG on Evaluation Data
    predictions = []
    for item in eval_data:
        result = rag_pipeline.run(item["question"])
        predictions.append(
            {
                "question": item["question"],
                "answer": result["result"],
                "contexts": [doc.page_content for doc in result["source_documents"]],
                "ground_truths": item["ground_truths"],
            }
        )

    # 8. Initialize Evaluator and Metrics
    metrics_list = [
        faithfulness(llm=ollama_llm),
        answer_relevancy(llm=ollama_llm),
        context_recall(llm=ollama_llm),
        context_precision(llm=ollama_llm),
        answer_correctness(llm=ollama_llm),
        answer_similarity(llm=ollama_llm),
    ]
    evaluator = Evaluator(metrics=metrics_list, llm=ollama_llm)

    # 9. Evaluate
    evaluation_results_df = evaluator.evaluate(predictions)

    # 10. Print Evaluation Results
    print("\nRagas Evaluation Results:")
    print(evaluation_results_df)


if __name__ == "__main__":
    main()