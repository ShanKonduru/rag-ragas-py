import os
from dotenv import load_dotenv

from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

from langchain_openai import OpenAIEmbeddings
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


def main():
    print("Hello World!!!")
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # 1. Load the input file to KnowledgeBase
    loader = TextLoader(f"inputs\\state_of_the_union.txt", encoding="utf-8")
    document = loader.load()
    print(document)

    # 2. Chunking the Data
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(document)
    print(chunks)

    # 3. Convert these chunks to embeddings and store them in Vector
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_documents(chunks, embeddings)
    retriever = vector_db.as_retriever()

    # 4. Define LLM
    llm = ChatOpenAI(temperature=0.6)

    # 5. Define Prompt Template
    template = """You are an assistant for question-answering tasks. 
    Use the following piece of retrieved context to answer the question.
    if you don't known the answer, just say that you don't know.
    Use two sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template=template)

    # 6. Setting up RAG Pipeline
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 7. Test One Question
    result = chain.invoke("what did president say about Justic Breyer?")
    print(result)

    # 8. Document Questions and ground truths
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

    answer = []
    content = []

    # 9. Inferences
    for query in questions:
        answer.append(chain.invoke(query))
        content.append(
            [doc.page_content for doc in retriever.get_relevant_documents(query)])

    print(answer)
    print(content)

    # 10. arrange meaningful output
    data = {
        "user_input": questions,
        "ground_truth": ground_truths,
        "answer": answer,
        "retrieved_contexts": content
    }

    dataset = Dataset.from_dict(data)
    print(dataset)

    result = evaluate(dataset=dataset, metrics=[
        Faithfulness(),
        ContextRelevance(),
        ContextRecall(),
        ContextPrecision()
    ],
        llm=llm,
        embeddings=embeddings
    )
    
    print(result)
    
    df = result.to_pandas()
    print(df)


if __name__ == "__main__":
    main()
