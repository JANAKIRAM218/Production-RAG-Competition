import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(".env", override=True)

CONFIG = {
"similarity_threshold": 0.95,
"k": 5,
"fetch_k": 50,
"embedding_model": "BAAI/bge-large-en-v1.5",
"llm_model": "llama-3.3-70b-versatile"
}

# Embeddings

embeddings = HuggingFaceEmbeddings(
model_name=CONFIG["embedding_model"]
)

print("[OK] BGE Large Embeddings Loaded")

# Load Saved FAISS Index

vectorstore = FAISS.load_local(
"zyro_hr_faiss",
embeddings,
allow_dangerous_deserialization=True
)

print("[OK] FAISS Loaded")

# Retriever

retriever = vectorstore.as_retriever(
search_type="mmr",
search_kwargs={
"k": CONFIG["k"],
"fetch_k": CONFIG["fetch_k"]
}
)

print("[OK] Retriever Ready")

# Out Of Scope Detection

def is_in_scope(query):

    result = vectorstore.similarity_search_with_score(
        query,
        k=1
    )

    _, score = result[0]

    return score < CONFIG["similarity_threshold"]

# LLM

llm = ChatGroq(
model=CONFIG["llm_model"],
temperature=0
)

print("[OK] Groq Loaded")

# Prompt

prompt = ChatPromptTemplate.from_template("""
You are a production-grade RAG assistant.

You must answer questions ONLY from the supplied context.

Instructions:

* Never use external knowledge.
* Never hallucinate information.
* If the answer is not explicitly present in the context, return:

I could not find that information in the provided documents.

Context:
{context}

Question:
{question}
""")

# Format Docs

def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

# RAG Chain

rag_chain = (
{
"context": retriever | format_docs,
"question": lambda x: x
}
| prompt
| llm
| StrOutputParser()
)

# Sources

def get_sources(docs):

    sources = []

    for doc in docs:

        source = (
            f"{doc.metadata['source_file']} "
            f"(Page {doc.metadata['page'] + 1})"
        )

        sources.append(source)

    return list(set(sources))

# Main Function

def ask_rag(question):

    if not is_in_scope(question):
        return (
            "I could not find that information "
            "in the provided documents."
        )

    docs = retriever.invoke(question)

    answer = rag_chain.invoke(question)

    sources = get_sources(docs)

    return (
        f"{answer}\n\n"
        f"Sources:\n- "
        + "\n- ".join(sources)
    )
