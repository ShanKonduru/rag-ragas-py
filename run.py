import os

from ragas import evaluate
from ragas.metrics import (
    AnswerRelevancy,
    Faithfulness,
    ContextRecall,
    ContextPrecision,
    AnswerCorrectness,
    AnswerSimilarity
)

from langchain.chains import RetrievalQA

from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings  import HuggingFaceEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

class OllamaWrapper(Ollama):
    def set_run_config(self, run_config):
        # Ollama doesn't inherently use a run config in the same way as OpenAI
        # You might need to adapt this if Ragas expects specific info in the config
        pass

def main():
    print("Hello World!!!")
    # 1. Load Documents (using sample text for now)
    documents = [
        "The capital of Canada is Ottawa. It is located in the province of Ontario.",
        "Calgary is a major city in Alberta, known for the Calgary Stampede.",
        "The Rocky Mountains are a significant mountain range in western North America.",
        "Maple syrup is a famous Canadian delicacy, often produced in Quebec.",
        "Ice hockey is a popular sport in Canada, with the NHL having several Canadian teams."
    ]

    # 2. Split Documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, chunk_overlap=10)
    chunks = text_splitter.create_documents(documents)

    # 3. Create Embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Create Vector Store
    db = FAISS.from_documents(chunks, embeddings)

    # 5. Create Retriever
    # Retrieve top 2 relevant chunks
    retriever = db.as_retriever(search_kwargs={"k": 2})

    # 6. Initialize Ollama LLM
    # Replace with your Ollama URL and model
    ollama_llm = OllamaWrapper(base_url='http://localhost:11434', model="llama2:13b")    
    # ollama_llm = Ollama(base_url='http://localhost:11434', model="llama2:13b")

    # 7. Create Prompt Template
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say you don't know.

    {context}

    Question: {question}"""
    PROMPT = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])

    # 8. Create RAG Chain
    rag_chain = RetrievalQA.from_llm(
        ollama_llm, retriever=retriever, prompt=PROMPT, return_source_documents=True)

    # Example Usage
    query = "What is the capital of Canada?"
    result = rag_chain({"query": query})
    print(f"Question: {query}")
    print(f"Answer: {result['result']}")

    query = "Tell me something about Calgary."
    result = rag_chain({"query": query})
    print(f"\nQuestion: {query}")
    print(f"Answer: {result['result']}")

    query = "What is the highest mountain in Canada?"
    result = rag_chain({"query": query})
    print(f"\nQuestion: {query}")
    print(f"Answer: {result['result']}")

    # 9. Generate Evaluation Data
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
        }
    ]

    # 10. Run the RAG System on the Evaluation Data
    predictions = []
    for item in eval_data:
        result = rag_chain({"query": item["question"]})
        predictions.append({
            "question": item["question"],
            "answer": result["result"],
            "contexts": [doc.page_content for doc in result["source_documents"]],
            "ground_truths": item["ground_truths"]
        })

    # 11. Evaluate using Ragas
    metrics = [
        Faithfulness(llm=ollama_llm),
        AnswerRelevancy(llm=ollama_llm),
        ContextRecall(llm=ollama_llm),
        ContextPrecision(llm=ollama_llm),
        AnswerCorrectness(llm=ollama_llm),
        AnswerSimilarity()
        ]
    eval_results = evaluate(predictions, metrics=metrics)

    # 12. Print Evaluation Results
    print("\nRagas Evaluation Results:")
    print(eval_results)

    # 13. Convert eval_results to a Pandas DataFrame
    df = eval_results.to_pandas()

    # 14. Print the DataFrame as a table
    print("\nRagas Evaluation Metrics:")
    print(df)


if __name__ == "__main__":
    main()
