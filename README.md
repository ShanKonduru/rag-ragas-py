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


## Metrics Notes

### 1. Faithfulness(): Telling the Truth!

What it means: Did the LLM answer stick to what was actually written in the story? Did it make stuff up?
Example: 
    Let's say the story said the dog was brown. If your LLM says the dog was blue, that's a big NO-NO!
Simple Example:
    Story: "The cat climbed the big oak tree."
    Good Answer (Faithful): "The cat climbed the big oak tree." (the LLM told it exactly like the story!)
    Bad Answer (Not Faithful): "The cat chased a mouse up a small maple tree." (LLM changed the tree and what the cat was doing!)

### 2. ContextRelevance(): Using the Right Hints!

What it means: When LLM answered a question, did LLM use the helpful parts of the story to find their answer? Or did it talk about something totally different?
Example: 
    Imagine the question is, "What color was the bird?". If the story says, "The yellow bird sang sweetly," then the word "yellow" is the important hint. If your classmate talks about the weather, that's not using the right hint!
Simple Example:
    Story: "The girl wore a red hat and blue shoes. She went to the park to play."
    Question: "What color was the girl's hat?"
    Good Answer (Relevant Context): "The girl's hat was red." (LLM used the part of the story about the hat.)
    Bad Answer (Irrelevant Context): "The girl went to the park." (This is true, but it doesn't answer the question about the hat's color!)

### 3. ContextRecall(): Remembering All the Important Stuff!

What it means: When we gave LLM some parts of the story to help it answer, did LLM remember to use all the important pieces of information?
Example: 
    Let's say the story said, "The boy had a red ball and a blue kite." If we ask, "What toys did the boy have?" and LLM only says "a red ball," LLM forgot the blue kite!
Simple Example:
    Story Snippet (Context): "The old house had a scary ghost and a creaky door."
    Question: "What were two things the old house had?"
    Good Answer (Good Recall): "The old house had a scary ghost and a creaky door." (LLM remembered both things!)
    Okay Answer (Not Great Recall): "The old house had a scary ghost." (LLM forgot about the door!)

### 4. ContextPrecision(): Just the Right Hints, Please!

What it means: When we gave your classmate parts of the story, were all those parts actually helpful for answering the question? Or did we give them some extra stuff that wasn't needed?
Example: 
    Imagine the question is, "What did the dog eat?". If the story says, "The big brown dog ate a tasty bone under the sunny sky," the important part is "dog ate a tasty bone." The words "big brown" and "under the sunny sky" are extra details that don't really help answer what the dog ate. We want just the "tasty bone" part to be precise.
Example:
    Story Snippet (Context): "The friendly baker made delicious cookies with chocolate chips and nuts. He sold them at the market."
    Question: "What did the baker make?"
    Good Context (Precise): "The friendly baker made delicious cookies." (Just the important part!)
    Okay Context (Not as Precise): "The friendly baker made delicious cookies with chocolate chips and nuts. He sold them at the market." (It has the right answer, but also extra stuff about the chips, nuts, and market that isn't needed for this question.)


## Important Notes
The application assumes that you have an OpenAI API key.
Ensure that your input text files are encoded in UTF-8.

## Improvements
Add more sophisticated question/answer generation.
Implement more robust evaluation methods.
Support more file types.
