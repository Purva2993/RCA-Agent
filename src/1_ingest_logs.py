"""
1_ingest_logs.py

WHAT THIS FILE DOES:
Reads log files → cuts them into chunks → converts to embeddings → stores in ChromaDB

RUN THIS ONCE before using the app.
Results are saved to data/chroma_db/ on disk.
"""

import os
import sys
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ── Configuration ──────────────────────────────
LOG_DIR       = "logs"
CHROMA_DIR    = "data/chroma_db"
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80

def load_logs(log_dir: str):
    """
    Reads all .log files from the logs/ folder.
    Returns a list of LangChain Document objects.
    """
    print(f"\n📂 Loading log files from '{log_dir}/'")

    # Check the folder actually exists
    if not os.path.exists(log_dir):
        print(f"❌ Folder '{log_dir}' not found.")
        print("Make sure you're running this from the rca-agent/ folder.")
        sys.exit(1)

    documents = []

    # Loop through every file in the logs/ folder
    for filename in sorted(os.listdir(log_dir)):

        # Only process .log files
        if filename.endswith(".log"):
            filepath = os.path.join(log_dir, filename)

            # TextLoader reads the file and returns a Document object
            loader = TextLoader(filepath, encoding="utf-8")
            docs   = loader.load()

            documents.extend(docs)

            size_kb = os.path.getsize(filepath) / 1024
            print(f"  ✅ Loaded: {filename} ({size_kb:.1f} KB)")

    # If no log files were found, stop the program
    if not documents:
        print("❌ No .log files found in the logs/ folder.")
        sys.exit(1)

    print(f"\n  → Total files loaded: {len(documents)}")
    return documents

def chunk_documents(documents):
    """
    Splits documents into smaller overlapping chunks.
    Returns a list of chunk Documents.
    """
    print(f"\n✂️  Chunking documents")
    print(f"  chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size      = CHUNK_SIZE,
        chunk_overlap   = CHUNK_OVERLAP,
        separators      = ["\n\n", "\n", " ", ""],
        add_start_index = True,
    )

    chunks = splitter.split_documents(documents)

    print(f"  → {len(chunks)} chunks created")

    # Show one example chunk so we can verify it looks right
    print(f"\n  Example chunk:")
    print(f"  ──────────────────────────────────────────")
    print(f"  {chunks[0].page_content[:200]}")
    print(f"  ──────────────────────────────────────────")

    return chunks

def build_vector_store(chunks):
    """
    Converts chunks to embeddings and stores in ChromaDB.
    Saves to disk at data/chroma_db/
    """
    print(f"\n🧠 Loading embedding model: '{EMBED_MODEL}'")
    print(f"  (First run downloads ~80MB — cached after that)")

    embeddings = HuggingFaceEmbeddings(
        model_name    = EMBED_MODEL,
        model_kwargs  = {"device": "cpu"},
        encode_kwargs = {"normalize_embeddings": True},
    )

    # Quick check — embed one sentence to confirm the model works
    test_vector = embeddings.embed_query("test")
    print(f"  ✅ Model loaded. Embedding size: {len(test_vector)} dimensions")

    print(f"\n🗃️  Building ChromaDB at '{CHROMA_DIR}/'")
    print(f"  Embedding {len(chunks)} chunks...")
    print(f"  This takes 20–60 seconds on first run.")

    vectorstore = Chroma.from_documents(
        documents         = chunks,
        embedding         = embeddings,
        persist_directory = CHROMA_DIR,
        collection_name   = "system_logs",
    )

    count = vectorstore._collection.count()
    print(f"\n  ✅ Done! {count} vectors stored in ChromaDB")
    print(f"  Saved to: {CHROMA_DIR}/")

    return vectorstore

def main():
    print("=" * 50)
    print("  Log Ingestion — Building Vector Store")
    print("=" * 50)

    # Step 1: Load the log files
    documents = load_logs(LOG_DIR)

    # Step 2: Cut them into chunks
    chunks = chunk_documents(documents)

    # Step 3: Embed and store in ChromaDB
    vectorstore = build_vector_store(chunks)

    print("\n" + "=" * 50)
    print("✅ Ingestion complete!")
    print("   Vector store is ready.")
    print("   Next: run python src/2_rag_pipeline.py")
    print("=" * 50)

    return vectorstore


if __name__ == "__main__":
    main()