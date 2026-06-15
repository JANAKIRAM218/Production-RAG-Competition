from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from sentence_transformers import CrossEncoder

# ============================================================

# Configuration

# ============================================================

load_dotenv(".env", override=True)

CONFIG = {
    "similarity_threshold": 0.80,
    "k": 10,
    "fetch_k": 30,
    "rerank_top_k": 3,
    "embedding_model": "BAAI/bge-large-en-v1.5",
    "llm_model": "llama-3.3-70b-versatile"
}

# ============================================================

# Embeddings

# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name=CONFIG["embedding_model"]
)

print("[OK] BGE Large Embeddings Loaded")

# ============================================================

# FAISS Vector Store

# ============================================================

vectorstore = FAISS.load_local(
    "zyro_hr_faiss",
    embeddings,
    allow_dangerous_deserialization=True
)

print("[OK] FAISS Loaded")

# ============================================================

# Retriever

# ============================================================

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": CONFIG["k"],
        "fetch_k": CONFIG["fetch_k"]
    }
)

print("[OK] Retriever Ready")

# ============================================================

# Cross Encoder Reranker

# ============================================================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("[OK] Reranker Ready")

# ============================================================

# LLM

# ============================================================

llm = ChatGroq(
    model=CONFIG["llm_model"],
    temperature=0
)

print("[OK] Groq Loaded")

# ============================================================

# Prompt

# ============================================================

prompt = ChatPromptTemplate.from_template("""
You are Zyro Dynamics HR Help Desk Assistant.

Answer employee questions ONLY using the supplied HR policy context.

Rules:

1. Use ONLY the provided context.
2. Never use outside knowledge.
3. Never invent policies.
4. Never invent dates, numbers, benefits, eligibility criteria, approval processes, or procedures.
5. If the answer is not clearly present in the context, respond exactly:

I could not find this information in the Zyro Dynamics HR policies.

6. Give the direct answer in the first sentence.
7. Include important limits, eligibility requirements, and exceptions if present.
8. Keep the answer concise.
9. Summarize policies instead of copying text verbatim.

Context:
{context}

Question:
{question}

Answer:
""")

# ============================================================

# Out-of-Scope Detection

# ============================================================

def is_in_scope(query):
    # Normalize company names before retrieval
    normalized_query = query.replace("Acrux Dynamics", "Zyro Dynamics").replace("Acrux", "Zyro")
    results = vectorstore.similarity_search_with_score(
        normalized_query,
        k=3
    )

    if not results:
        return False

    top_score = results[0][1]
    return top_score < CONFIG["similarity_threshold"]

# ============================================================

# Reranking

# ============================================================

def rerank_documents(
    query,
    docs,
    top_k=None
):
    if top_k is None:
        top_k = CONFIG["rerank_top_k"]

    if not docs:
        return []

    pairs = [
        (query, doc.page_content)
        for doc in docs
    ]

    scores = reranker.predict(pairs)

    ranked_docs = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        doc
        for doc, _ in ranked_docs[:top_k]
    ]

# ============================================================

# Context Formatting

# ============================================================

def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

# ============================================================

# Source Formatting

# ============================================================

def get_sources(docs):
    seen = set()
    sources = []

    for doc in docs:
        source = doc.metadata.get('source_file', 'Unknown')

        if source not in seen:
            seen.add(source)
            sources.append(source)

    return sources

# ============================================================

# Main RAG Function

# ============================================================

def ask_rag(question):
    # Normalize company names before retrieval
    normalized_question = question.replace("Acrux Dynamics", "Zyro Dynamics").replace("Acrux", "Zyro")

    if not is_in_scope(normalized_question):
        return {
            "answer": "I could not find this information in the Zyro Dynamics HR policies.",
            "sources": [],
            "docs": []
        }

    docs = retriever.invoke(normalized_question)

    docs = rerank_documents(
        normalized_question,
        docs
    )

    if not docs:
        return {
            "answer": "I could not find this information in the Zyro Dynamics HR policies.",
            "sources": [],
            "docs": []
        }

    # Deduplicate retrieved chunks before sending context to the LLM
    seen_content = set()
    deduped_docs = []
    for doc in docs:
        content = doc.page_content.strip()
        if content not in seen_content:
            seen_content.add(content)
            deduped_docs.append(doc)
    docs = deduped_docs

    context = format_docs(docs)

    answer = (
        prompt
        | llm
        | StrOutputParser()
    ).invoke(
        {
            "context": context,
            "question": normalized_question
        }
    )

    refusal_text = "I could not find this information in the Zyro Dynamics HR policies."
    if refusal_text in answer:
        return {
            "answer": refusal_text,
            "sources": [],
            "docs": []
        }

    sources = get_sources(docs)

    return {
        "answer": answer,
        "sources": sources,
        "docs": docs
    }
