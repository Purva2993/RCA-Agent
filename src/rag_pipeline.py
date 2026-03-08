"""
2_rag_pipeline.py

WHAT THIS FILE DOES:
Loads ChromaDB → finds relevant log chunks → sends to LLM → returns RCA answer

This is the Simple RAG mode:
  Question → embed → search ChromaDB → build prompt → LLM → answer

REQUIRES:
  - data/chroma_db/ must exist (run 1_ingest_logs.py first)
  - GROQ_API_KEY in .env file
"""

import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

# ── Configuration ──────────────────────────────
CHROMA_DIR  = "data/chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K       = 5        # number of chunks to retrieve per query
LLM_MODEL   = "llama-3.3-70b-versatile"  # free and fast on Groq

RCA_PROMPT = ChatPromptTemplate.from_template("""
You are a senior Site Reliability Engineer (SRE) performing Root Cause Analysis.

Analyze the log entries below and answer the question with this exact structure:

1. 🔍 ROOT CAUSE
   The single deepest underlying reason for the failure.

2. 🔗 CHAIN OF EVENTS
   A numbered timeline of what triggered what.

3. 🚨 IMMEDIATE FIX
   What the on-call engineer should do right now.

4. 🛡️ LONG TERM FIX
   Code or config changes to prevent this recurring.

Be specific. Reference actual log lines and timestamps.
If the logs don't contain enough information, say so clearly.

════════════════════════════════
RELEVANT LOG ENTRIES:
════════════════════════════════
{context}

════════════════════════════════
QUESTION:
════════════════════════════════
{question}

════════════════════════════════
ROOT CAUSE ANALYSIS:
════════════════════════════════
""")
def load_retriever():
    """
    Connects to ChromaDB and returns a retriever object.
    The retriever's job: take a question, return the TOP_K most relevant chunks.
    """
    print("🗃️  Connecting to ChromaDB...")

    embeddings = HuggingFaceEmbeddings(
        model_name    = EMBED_MODEL,
        model_kwargs  = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        persist_directory  = CHROMA_DIR,
        embedding_function = embeddings,
        collection_name    = "system_logs",
    )

    count = vectorstore._collection.count()
    print(f"  ✅ Connected. {count} vectors in index.")

    retriever = vectorstore.as_retriever(
        search_type   = "similarity",
        search_kwargs = {"k": TOP_K},
    )

    return retriever, vectorstore

def load_llm():
    """
    Loads Llama 3 via the Groq API.
    Groq is free, fast (~300 tokens/second), no credit card needed.
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "\\n❌ GROQ_API_KEY not found!\\n"
            "   Make sure your .env file contains:\\n"
            "   GROQ_API_KEY=gsk_xxxx\\n"
        )

    print(f"🤖 Loading LLM: {LLM_MODEL} via Groq...")

    llm = ChatGroq(
        model       = LLM_MODEL,
        temperature = 0,
        api_key     = api_key,
    )

    print(f"  ✅ LLM ready.")
    return llm

def format_docs(docs) -> str:
    """
    Converts a list of Document objects into a single formatted string.
    This string becomes the {context} in our prompt.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown").split("/")[-1]
        parts.append(f"[Source: {source} | Chunk {i}]\\n{doc.page_content}")
    return "\\n\\n".join(parts)

def build_rag_chain(retriever, llm):
    """
    Wires everything together using LangChain's pipe operator.

    The chain flow:
    question
      → retriever fetches TOP_K relevant chunks
      → format_docs converts them to a string
      → RCA_PROMPT fills {context} and {question}
      → llm generates the answer
      → StrOutputParser extracts plain text
    """
    chain = (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | RCA_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain

def query_rag(question: str, chain, retriever) -> dict:
    """
    Runs a question through the RAG chain.
    Returns both the answer AND the retrieved chunks.
    The chunks are returned so the UI can show the evidence.
    """
    # Get the retrieved chunks separately so we can display them
    retrieved_docs = retriever.invoke(question)

    # Run the full chain to get the answer
    answer = chain.invoke(question)

    return {
        "answer":           answer,
        "retrieved_chunks": [
            {
                "text":   doc.page_content,
                "source": doc.metadata.get("source", "unknown").split("/")[-1],
            }
            for doc in retrieved_docs
        ],
    }

def main():
    print("=" * 50)
    print("  RAG Pipeline — Querying Logs With LLM")
    print("=" * 50)

    # Load components
    retriever, _ = load_retriever()
    llm          = load_llm()
    chain        = build_rag_chain(retriever, llm)

    # Test question
    question = "Why did the payment service crash at 2am? What was the root cause?"

    print(f"\\n📋 Question: {question}")
    print("\\nSearching logs and generating answer...\\n")

    result = query_rag(question, chain, retriever)

    # Show retrieved chunks
    print(f"Retrieved {len(result['retrieved_chunks'])} chunks:")
    for i, chunk in enumerate(result["retrieved_chunks"], 1):
        print(f"\\n  [{i}] Source: {chunk['source']}")
        print(f"  {chunk['text'][:150].strip()}...")

    # Show the answer
    print("\\n" + "=" * 50)
    print("🤖 ROOT CAUSE ANALYSIS:")
    print("=" * 50)
    print(result["answer"])
    print("=" * 50)

    return chain, retriever


if __name__ == "__main__":
    main()