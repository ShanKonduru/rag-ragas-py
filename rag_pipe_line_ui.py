import os
from dotenv import load_dotenv
from rag_pipe_line import RAGPipeline
import streamlit as st
import tempfile 


def main():
    # Load environment variables
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    st.title("RAG Pipeline Application")

    # Initialize RAGPipeline instance
    rag_pipeline = RAGPipeline(openai_api_key=openai_api_key)

    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("Knowledge Base")
        uploaded_files = st.file_uploader("Upload your knowledge base documents (txt files)", accept_multiple_files=True)
        load_method = st.radio("Choose how to load data:", ["Files", "Directory"])
        if load_method == "Directory":
            directory_path = st.text_input("Enter the path to the directory containing your text files:")
        else:
            directory_path = None

        # Load documents
        if uploaded_files or directory_path:
            with st.spinner("Loading and processing documents..."):
                if load_method == "Files":
                    documents = []
                    for uploaded_file in uploaded_files:
                        # Save the uploaded file to a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_file:
                            temp_file.write(uploaded_file.read())
                            temp_file_path = temp_file.name  # Get the path to the temp file
                        # Load the document from the temporary file
                        doc = rag_pipeline.load_knowledge_base(temp_file_path)
                        documents.extend(doc)
                    rag_pipeline.document = documents  # directly assign the documents
                elif load_method == "Directory":
                    rag_pipeline.document = rag_pipeline.load_knowledge_base_directory(directory_path)

                rag_pipeline.chunks = rag_pipeline.chunk_data(rag_pipeline.document)
                rag_pipeline.create_vector_database(rag_pipeline.chunks)
            st.success("Documents loaded and processed!")

        # Load Vector Store
        st.header("Load/Save Vector Store")
        load_vectorstore = st.checkbox("Load Vector Store from disk")
        if load_vectorstore:
            load_path = st.text_input("Enter the path to the vector store:")
            if load_path:
                rag_pipeline.load_vector_store(load_path)
                if rag_pipeline.vector_db:
                    st.success("Vector store loaded!")
                else:
                    st.error("Failed to load vector store.")

        # Save Vector Store
        save_vectorstore = st.checkbox("Save Vector Store to disk")
        if save_vectorstore:
            save_path = st.text_input("Enter the path to save the vector store:")
            if save_path and rag_pipeline.vector_db:
                rag_pipeline.save_vector_store(save_path)
                st.success("Vector store saved!")
            elif save_path and not rag_pipeline.vector_db:
                st.warning("No vector store to save.  Create the knowledge base first.")
            else:
                st.warning("Please enter a path to save the vector store.")

        # RAG Pipeline Setup
        st.header("RAG Pipeline Setup")
        prompt_template = st.text_area("Enter the prompt template:",
                                       "Use the following context to answer the question: {context} Question: {question}")
        if rag_pipeline.retriever:  # Only allow setup if retriever is available
            rag_pipeline.setup_rag_pipeline(prompt_template)
            st.success("RAG pipeline is set up!")

    # Main area for conversation
    st.header("Conversation")
    question = st.text_input("Ask a question:", key="question_input") # Assign a key
    if question:
        if rag_pipeline.chain:
            answer = rag_pipeline.query(question)
            st.write("Answer:", answer)

            # Evaluation for each answer
            if st.checkbox("Evaluate this response"):
                questions, ground_truths = rag_pipeline.generate_questions_and_ground_truths([rag_pipeline.document[0]], num_questions=1) # Generate for the specific question
                if questions and ground_truths:
                    # Get relevant documents *before* calling evaluate_rag
                    relevant_docs = rag_pipeline.retriever.get_relevant_documents(question)
                    contexts = [[doc.page_content for doc in relevant_docs]]
                    evaluation_df = rag_pipeline.evaluate_rag(
                        questions=[question],
                        ground_truths=ground_truths,
                        answers=[answer],
                        contexts=contexts
                    )
                    st.write("Evaluation Results:")
                    st.dataframe(evaluation_df)
                    rag_pipeline.save_evaluation_to_csv(evaluation_df)
                else:
                    st.warning("Not enough content in the document to generate ground truths for evaluation.")
        else:
            st.error("Please set up the RAG pipeline first.")

    # Display conversation history
    if len(rag_pipeline.history) > 0:
        st.subheader("Conversation History")
        for interaction in rag_pipeline.history:
            st.write(f"Question: {interaction['question']}")
            st.write(f"Answer: {interaction['answer']}")
            st.write("-" * 50)

if __name__ == "__main__":
    main()
