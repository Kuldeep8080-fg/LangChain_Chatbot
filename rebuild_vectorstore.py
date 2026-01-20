"""
Rebuild Vector Store with Clean Data

This script will:
1. Delete the old vector store
2. Reload documentation, filtering out redirect/bad pages
3. Create new embeddings only for good content
"""
import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.doc_loader import load_all_documentation
from src.embeddings import VECTORSTORE_DIR, COLLECTION_NAME, get_embeddings
from langchain_community.vectorstores import Chroma


def filter_good_documents(chunks):
    """Filter out redirect pages and pages with no real content."""
    good_chunks = []
    bad_count = 0
    
    for chunk in chunks:
        content = chunk.page_content
        
        # Skip redirect pages
        if 'Redirecting' in content:
            bad_count += 1
            continue
        
        # Skip very short content
        if len(content.strip()) < 100:
            bad_count += 1
            continue
        
        # Skip pages that are mostly navigation
        if content.count('Skip to') > 2:
            bad_count += 1
            continue
        
        good_chunks.append(chunk)
    
    print(f"Filtered out {bad_count} bad chunks")
    print(f"Keeping {len(good_chunks)} good chunks")
    
    return good_chunks


def rebuild_vectorstore():
    """Delete old vectorstore and create new one with clean data."""
    
    print("\n" + "="*60)
    print("REBUILDING VECTOR STORE")
    print("="*60)
    
    # Step 1: Delete old vector store
    if os.path.exists(VECTORSTORE_DIR):
        print(f"\n[1/4] Deleting old vector store at: {VECTORSTORE_DIR}")
        shutil.rmtree(VECTORSTORE_DIR)
        print("      Done!")
    else:
        print("\n[1/4] No existing vector store to delete")
    
    # Step 2: Load documentation
    print("\n[2/4] Loading documentation from URLs...")
    docs, chunks = load_all_documentation()
    
    # Step 3: Filter out bad documents
    print("\n[3/4] Filtering out redirect and bad pages...")
    good_chunks = filter_good_documents(chunks)
    
    if len(good_chunks) == 0:
        print("ERROR: No good documents found!")
        return False
    
    # Step 4: Create new embeddings
    print("\n[4/4] Creating embeddings for good documents...")
    print(f"      This may take a few minutes for {len(good_chunks)} chunks...")
    
    embeddings = get_embeddings()
    
    vectorstore = Chroma.from_documents(
        documents=good_chunks,
        embedding=embeddings,
        persist_directory=VECTORSTORE_DIR,
        collection_name=COLLECTION_NAME
    )
    
    print(f"\n      Vector store created with {len(good_chunks)} vectors!")
    
    # Test the new vector store
    print("\n" + "="*60)
    print("TESTING NEW VECTOR STORE")
    print("="*60)
    
    test_query = "What is RAG?"
    results = vectorstore.similarity_search(test_query, k=3)
    
    print(f"\nQuery: '{test_query}'")
    print(f"Found {len(results)} results:\n")
    
    for i, doc in enumerate(results, 1):
        content = doc.page_content[:150].replace('\n', ' ')
        source = doc.metadata.get('source_framework', 'unknown')
        is_redirect = 'Redirecting' in doc.page_content
        status = "BAD" if is_redirect else "GOOD"
        print(f"[{status}] Result {i} ({source}): {content}...")
    
    print("\n" + "="*60)
    print("REBUILD COMPLETE!")
    print("="*60)
    
    return True


if __name__ == "__main__":
    rebuild_vectorstore()
