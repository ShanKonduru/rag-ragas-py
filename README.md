# RAG Pipeline Application

## Overview
This Streamlit application implements a Retrieval-Augmented Generation (RAG) pipeline. It allows you to upload a knowledge base (text files), query it using natural language, and evaluate the quality of the answers.

## Features
Knowledge Base Management:
Upload text files as your knowledge base.
Load data from individual files or a directory.
Vector Store:
Automatically processes uploaded documents into chunks.
Creates and saves a FAISS vector store for efficient retrieval.
Load/Save vector store from/to disk
RAG Pipeline:
Set up a RAG pipeline with a customizable prompt template.
Query the knowledge base using natural language.
Displays the answer
Evaluation:
Option to evaluate the answer using Ragas metrics(Faithfulness, Context Relevance, Context Recall, Context Precision).
Conversation History:
Displays the history of your questions and the corresponding answers.

## How to Use
Upload Knowledge Base:
In the sidebar, upload your text files or provide a directory containing them.
Load/Save Vector Store (Optional):
You can save the vector store to disk for later use or load an existing one.
Set Up RAG Pipeline:
In the sidebar, enter a prompt template. The default is "Use the following context to answer the question: {context} Question: {question}".
Ask Questions:
In the "Conversation" section, type your question and press Enter. The application will retrieve relevant information from the knowledge base and generate an answer.
Evaluate Response:
Check the "Evaluate this response" box to evaluate the last generated response. The application uses Ragas metrics and displays them.
View Conversation History:
The conversation history is displayed below the question input area.

## RAG Pipeline Details
The application uses the following components to implement the RAG pipeline:
Langchain:
TextLoader and DirectoryLoader: For loading documents.
RecursiveCharacterTextSplitter: For chunking documents.
ChatOpenAI: For generating answers.
FAISS: For creating the vector database.
ChatPromptTemplate: For creating the prompt.
RunnablePassthrough and StrOutputParser: For creating the RAG chain.
OpenAI:
OpenAIEmbeddings: For generating embeddings.
Ragas:
For evaluating the quality of the generated answers.

## Installation
Clone this repository.
Install the required packages:
pip install -r requirements.txt


Set up your environment variables:
Create a .env file in the root directory.
Add your OpenAI API key to the .env file:
OPENAI_API_KEY=YOUR_OPENAI_API_KEY


Run the Streamlit application:
streamlit run your_app_file.py # Replace your_app_file.py with the name of your streamlit file.


## Important Notes
The application assumes that you have an OpenAI API key.
Ensure that your input text files are encoded in UTF-8.

## Improvements
Add more sophisticated question/answer generation.
Implement more robust evaluation methods.
Support more file types.
