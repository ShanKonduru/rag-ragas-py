import os
from dotenv import load_dotenv
from rag_pipe_line import RAGPipeline

def main():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        print("OPENAI_API_KEY not found in environment variables.")
        return

    rag_pipeline = RAGPipeline(openai_api_key=openai_api_key)
    vector_store_path = "vector_store"

    # 1. Load the input file to KnowledgeBase
    file_path = "inputs\\state_of_the_union.txt"
    document = rag_pipeline.load_knowledge_base(file_path)
    print("Loaded Document:", document)

    # 2. Chunking the Data
    chunks = rag_pipeline.chunk_data(document)
    print("\nChunked Data (first chunk):", chunks[0])

    # 3. Create Vector Database and Retriever
    rag_pipeline.create_vector_database(chunks)
    rag_pipeline.save_vector_store(vector_store_path)
    
    # 3.1 Load an Existing Vector Store (comment out the creation part above)
    # rag_pipeline.load_vector_store(vector_store_path)
    
    # 4. Define Prompt Template
    prompt_template = """You are an assistant for question-answering tasks.
    Use the following piece of retrieved context to answer the question.
    if you don't known the answer, just say that you don't know.
    Use two sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:
    """

    # 5. Set up RAG Pipeline
    rag_pipeline.setup_rag_pipeline(prompt_template)

    # 6. Test One Question
    test_question = "what did president say about Justic Breyer?"
    result = rag_pipeline.query(test_question)
    print(f"\nQuestion: {test_question}")
    print(f"Answer: {result}")

    #7. Use Hardcoded Questions and Ground Truths (basic example)
    questions = [
        "What was the President's main message regarding Russia's actions in Ukraine?",
        "What specific actions did the President announce the US would take against Russia?",
        "Which countries were mentioned as allies in response to the situation in Ukraine?",
        "What kind of assistance is the US providing to Ukraine?",
        "Did the President suggest any potential for US military involvement in Ukraine?",
        "What was the President's message to the Ukrainian Ambassador present at the address?"
    ]

    ground_truths = [
        "Russia's Vladimir Putin badly miscalculated by trying to shake the foundations of the free world and invade Ukraine.",
        "Enforcing powerful economic sanctions",
        "Twenty-seven members of the European Union (including France, Germany, Italy)",
        "Military assistance",
        "Let me be clear, our forces are not engaged and will not engage in conflict with Russian forces in Ukraine",
        "The President asked everyone in the Chamber to rise if able and show that the United States of America stands with the Ukrainian people, sending an unmistakable signal to Ukraine and the world."
    ]

    # # 7. Generate Questions and Ground Truths (basic example)
    # questions, ground_truths = rag_pipeline.generate_questions_and_ground_truths(document, num_questions=6)
    # print("\nGenerated Questions:", questions)
    # print("Generated Ground Truths:", ground_truths)

    # 8. Evaluate RAG Pipeline
    if rag_pipeline.retriever:
        evaluation_df = rag_pipeline.evaluate_rag(questions, ground_truths)
        print("\nRAG Evaluation Results:")
        print(evaluation_df)
        # 9. Save the DataFrame to CSV
        rag_pipeline.save_evaluation_to_csv(evaluation_df)
    else:
        print("\nRetriever not initialized, skipping evaluation and CSV saving.")

if __name__ == "__main__":
    main()