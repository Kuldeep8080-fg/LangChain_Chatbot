"""
Embeddings and Vector Store Module

This module handles:
1. Creating embeddings using Ollama's nomic-embed-text model
2. Storing documents in ChromaDB (persistent vector database)
3. Loading existing vector store for queries

Why ChromaDB?
- Easy to set up (no external server needed)
- Persists to disk (data survives restarts)
- Great for learning and prototyping
- Works seamlessly with LangChain

Why nomic-embed-text?
- Free, runs locally via Ollama
- Good quality embeddings (768 dimensions)
- Fast inference
"""

import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List, Optional

# Get the directory where this file is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

# Vector store will be saved here
VECTORSTORE_DIR = os.path.join(PARENT_DIR, "vectorstore")
COLLECTION_NAME = "langchain_docs"


def get_embeddings():
    """
    Get the HuggingFace embeddings model (runs locally on CPU).
    
    Returns:
        HuggingFaceEmbeddings configured with all-MiniLM-L6-v2
    """
    try:
        # Try loading locally first (Faster & fixes SSL errors if cached)
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu', 'local_files_only': True},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception:
        # Fallback to downloading (For new users/fresh clones)
        print("Local model not found. Downloading from HuggingFace...")
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu', 'local_files_only': False},
            encode_kwargs={'normalize_embeddings': True}
        )


def create_vectorstore(documents: List, show_progress: bool = True) -> Chroma:
    """
    Create a new vector store from documents.
    
    This process:
    1. Takes each document chunk
    2. Converts it to a 768-dimensional vector using Ollama
    3. Stores the vector + original text in ChromaDB
    
    Args:
        documents: List of document chunks from doc_loader
        show_progress: Whether to show progress messages
    
    Returns:
        Chroma vector store with embedded documents
    """
    if show_progress:
        print("\n" + "="*60)
        print("\n" + "="*60)
        print("Creating Vector Store")
        print("="*60)
        print(f"   Documents to embed: {len(documents)}")
        print(f"   Storage location: {VECTORSTORE_DIR}")
        print(f"   Collection name: {COLLECTION_NAME}")
        print("\n   This may take a few minutes...")
    
    # Get embeddings model
    embeddings = get_embeddings()
    
    # Create ChromaDB vector store
    # persist_directory saves to disk so we don't have to re-embed every time!
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR,
        collection_name=COLLECTION_NAME
    )
    
    if show_progress:
        print(f"\n   Vector store created successfully!")
        print(f"   Total vectors stored: {len(documents)}")
    
    return vectorstore


def load_vectorstore() -> Optional[Chroma]:
    """
    Load an existing vector store from disk.
    
    Use this to avoid re-embedding documents every time!
    
    Returns:
        Chroma vector store if exists, None otherwise
    """
    if not os.path.exists(VECTORSTORE_DIR):
        print("Vector store not found. Run create_vectorstore first!")
        return None
    
    print(f"Loading vector store from: {VECTORSTORE_DIR}")
    
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    # Get the count of documents in the store
    count = vectorstore._collection.count()
    print(f"   Loaded {count} vectors from existing store")
    
    return vectorstore


def similarity_search(query: str, k: int = 4) -> List:
    """
    Search for documents similar to the query.
    
    This is how RAG works:
    1. Convert query to vector (using same embedding model)
    2. Find k most similar vectors in the database
    3. Return the original text of those documents
    
    Args:
        query: The question or search query
        k: Number of similar documents to return (default: 4)
    
    Returns:
        List of most similar document chunks
    """
    vectorstore = load_vectorstore()
    if not vectorstore:
        return []
    
    print(f"\n🔍 Searching for: '{query[:50]}...'")
    results = vectorstore.similarity_search(query, k=k)
    print(f"   Found {len(results)} relevant documents")
    
    return results


# Test the module when run directly
if __name__ == "__main__":
    print("\n" + "#"*60)
    print("#  Vector Store Setup")
    print("#"*60)
    
    # Check if vector store already exists
    if os.path.exists(VECTORSTORE_DIR) and os.listdir(VECTORSTORE_DIR):
        print("\nExisting vector store found!")
        choice = input("   Do you want to (L)oad existing or (R)ecreate? [L/R]: ").strip().upper()
        
        if choice == "L":
            vectorstore = load_vectorstore()
        else:
            # Import doc_loader to get fresh documents
            from doc_loader import load_all_documentation
            docs, chunks = load_all_documentation()
            vectorstore = create_vectorstore(chunks)
    else:
        print("\nNo existing vector store. Creating new one...")
        # Import doc_loader to get documents
        from doc_loader import load_all_documentation
        docs, chunks = load_all_documentation()
        vectorstore = create_vectorstore(chunks)
    
    # Test a search
    if vectorstore:
        print("\n" + "="*60)
        print("Testing Vector Search")
        print("="*60)
        
        test_query = "What is RAG in LangChain?"
        results = vectorstore.similarity_search(test_query, k=3)
        
        print(f"\nQuery: '{test_query}'")
        print(f"Found {len(results)} relevant chunks:\n")
        
        for i, doc in enumerate(results, 1):
            print(f"--- Result {i} ---")
            print(f"Source: {doc.metadata.get('source_framework', 'unknown')}")
            print(f"URL: {doc.metadata.get('source_url', 'unknown')[:50]}...")
            print(f"Content: {doc.page_content[:200]}...")
            print()
