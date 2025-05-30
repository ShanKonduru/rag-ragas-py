import os
from dotenv import load_dotenv
from FolderInfoIterator import FolderInfoIterator
from rag_pipe_line import RAGPipeline

input_folder_info = [
    {
        "input_folder_path": "inputs\\Canada-KB",
        "vector_output_name": "Canada",
        "vector_output_path": "vector_store\\Canada-KB",
        "set_of_questions" : [
            "What are the two official languages of Canada?",
            "Who is the current Prime Minister of Canada?",
            "Describe the main geographical regions of Canada.",
            "Which national symbol of Canada?",
            "What was the weather like in Vancouver, British Columbia on March 15, 2024?",
            "What is the most popular Canadian television show currently streaming on major platforms?",
            "What is Canada's national winter sport?",
            "Why do we have maple leaf on the flag",
            "Considering its geography and major industries, what are some significant economic strengths of Canada?"            
        ],
        "set_of_ground_truths" : [
            "English and French",
            "Mark Carney",
            "the Rockies, the Appalachians, vast prairies, boreal forests, tundra, and extensive coastlines",
            "maple leaf",
            "likely mild and rainy, typical for that time of year. Specific details would include the temperature (around 8-12°C), precipitation",
            "Northern Lights Mystery on StreamFlix",
            "Hockey and Lacrosse",
            "national symbol",
            "Natural resources like oil, gas, minerals, and timber, as well as strong industries in automotive, aerospace, technology, and finance"
        ]
    }
]

def main():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        print("OPENAI_API_KEY not found in environment variables.")
        return

    rag_pipeline = RAGPipeline(openai_api_key=openai_api_key)

    run_evaluate_code = True # False #  
    kb_to_read =   "Canada" # "CTO-ISB" # "state_of_the_union" # "VedicMetaverses" #  
    
    folder_info_iterator = FolderInfoIterator(input_folder_info)
    vector_output_name = folder_info_iterator.get_by_attribute("vector_output_name", kb_to_read)
    vector_store_path  = vector_output_name.get("vector_output_path")
    print(f"Item with vector_output_name '{kb_to_read}': {vector_store_path}")

    # 1. Load the input file to KnowledgeBase
    folder_path  = vector_output_name.get("input_folder_path")
    print(f"Item with input_folder_path '{kb_to_read}': {folder_path}")

    document = rag_pipeline.load_knowledge_base_directory(folder_path)

    # file_path = "inputs\\state_of_the_union.txt"
    # file_path = "inputs\\about_canada.txt"
    # document = rag_pipeline.load_knowledge_base(file_path)
    print("Loaded Document:", document)

    # 2. Chunking the Data
    chunks = rag_pipeline.chunk_data(document)
    print("\nChunked Data (first chunk):", chunks [0])

    # 3. Create Vector Database and Retriever
    rag_pipeline.create_vector_database(chunks)
    rag_pipeline.save_vector_store(vector_store_path, kb_to_read)

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
    set_of_questions = vector_output_name.get("set_of_questions")
    print(f"Questions for '{vector_output_name.get('set_of_questions')}': {set_of_questions}")
    print("*******************************************")
    
    set_of_ground_truths = vector_output_name.get("set_of_ground_truths")
    print(f"Questions for '{vector_output_name.get('set_of_ground_truths')}': {    set_of_ground_truths}")
    print("*******************************************")

    additional_context = " The Canadian Shield is a vast, ancient geological formation covering much of central and eastern Canada. It is characterized by exposed bedrock, numerous lakes and forests, and significant mineral deposits., The Western Cordillera is a complex region of mountain ranges, including the Rocky Mountains and the Coast Mountains, plateaus, and valleys along the western edge of Canada. It is known for its rugged terrain and diverse ecosystems., The Interior Plains lie between the Canadian Shield and the Western Cordillera and are characterized by relatively flat grasslands, fertile agricultural land, and sedimentary rock formations., The Great Lakes-St. Lawrence Lowlands are a low-lying region in southeastern Canada bordering the Great Lakes and the St. Lawrence River. This area has fertile soil and is a major center for agriculture and population., The Appalachian Region, located in Eastern Canada, consists of older, eroded mountain ranges, rolling hills, and coastal plains. It has a diverse landscape of forests and maritime environments., The Arctic Region encompasses the northernmost parts of Canada, characterized by tundra vegetation, permafrost, ice-covered land and sea, and a harsh climate."
    
    results = []
    contexts = []
    for test_question in set_of_questions:
        result = rag_pipeline.query(test_question)
        results.append(result)
        print(f"\nQuestion: {test_question}")
        print(f"Answer: {result}")
        relevant_docs = rag_pipeline.retriever.get_relevant_documents(
            test_question)
        page_contents = [doc.page_content for doc in relevant_docs]
        contexts.append(page_contents)
    
        if (not run_evaluate_code):
            os.system("pause")

    if (run_evaluate_code):
        # 7. Use Hardcoded Questions and Ground Truths (basic example)
        questions = set_of_questions
        ground_truths = set_of_ground_truths

        # # 7. Generate Questions and Ground Truths (basic example)
        # questions, ground_truths = rag_pipeline.generate_questions_and_ground_truths(document, num_questions=6)
        # print("\nGenerated Questions:", questions)
        # print("Generated Ground Truths:", ground_truths)

        # 8. Evaluate RAG Pipeline
        if rag_pipeline.retriever:
            questions_string = "".join(questions)

            print(f"Number of questions: {len(questions)}")
            print(f"Number of ground_truths: {len(ground_truths)}")
            # Assuming 'answer' is the key for 'result'
            print(f"Number of results (answers): {len(results)}")
            print(f"Number of contexts: {len(contexts)}")

            evaluation_df = rag_pipeline.evaluate_rag(
                questions, ground_truths, results, contexts)
            print("\nRAG Evaluation Results:")
            print(evaluation_df)
            # 9. Save the DataFrame to CSV
            rag_pipeline.save_evaluation_to_csv(evaluation_df)
        else:
            print("\nRetriever not initialized, skipping evaluation and CSV saving.")


if __name__ == "__main__":
    main()
