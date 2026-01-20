"""
Document Loader for LangChain, LangGraph, and LangSmith Documentation

This module handles:
1. Loading documentation from URLs using LangChain's WebBaseLoader
2. Splitting documents into chunks for embedding
3. Saving processed documents for vector store creation

Why we use WebBaseLoader:
- Built into LangChain, no extra setup needed
- Handles HTML parsing automatically
- Works with most documentation websites
"""

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
import os


# Documentation URLs for each framework
# Comprehensive coverage for developers learning or troubleshooting!

LANGCHAIN_URLS = [
    # ----- Core Concepts (Must Know) -----
    "https://python.langchain.com/docs/introduction/",
    "https://python.langchain.com/docs/concepts/",
    "https://python.langchain.com/docs/concepts/architecture/",
    
    # ----- Chat Models & LLMs -----
    "https://python.langchain.com/docs/concepts/chat_models/",
    "https://python.langchain.com/docs/concepts/messages/",
    "https://python.langchain.com/docs/concepts/llms/",
    
    # ----- Prompts & Templates -----
    "https://python.langchain.com/docs/concepts/prompt_templates/",
    "https://python.langchain.com/docs/concepts/few_shot_prompting/",
    "https://python.langchain.com/docs/concepts/example_selectors/",
    "https://python.langchain.com/docs/how_to/prompt_templates/",
    "https://python.langchain.com/docs/how_to/custom_prompt_templates/",
    "https://python.langchain.com/docs/how_to/prompts_composition/",
    
    # ----- Output Parsing -----
    "https://python.langchain.com/docs/concepts/output_parsers/",
    "https://python.langchain.com/docs/concepts/structured_outputs/",
    
    # ----- Chains & Runnables (Core Pattern) -----
    "https://python.langchain.com/docs/concepts/runnables/",
    "https://python.langchain.com/docs/concepts/lcel/",
    "https://python.langchain.com/docs/concepts/streaming/",
    
    # ----- RAG (Most Common Use Case) -----
    "https://python.langchain.com/docs/concepts/rag/",
    "https://python.langchain.com/docs/concepts/vectorstores/",
    "https://python.langchain.com/docs/concepts/retrievers/",
    "https://python.langchain.com/docs/concepts/text_splitters/",
    "https://python.langchain.com/docs/concepts/embedding_models/",
    
    # ----- Document Loaders -----
    "https://python.langchain.com/docs/concepts/document_loaders/",
    
    # ----- Agents & Tools -----
    "https://python.langchain.com/docs/concepts/agents/",
    "https://python.langchain.com/docs/concepts/tools/",
    "https://python.langchain.com/docs/concepts/tool_calling/",
    
    # ----- Memory & Chat History -----
    "https://python.langchain.com/docs/concepts/memory/",
    "https://python.langchain.com/docs/concepts/chat_history/",
    
    # ----- How-To Guides (Practical) -----
    "https://python.langchain.com/docs/how_to/sequence/",
    "https://python.langchain.com/docs/how_to/parallel/",
    "https://python.langchain.com/docs/how_to/binding/",
    "https://python.langchain.com/docs/how_to/fallbacks/",
    
    # ----- Tutorials -----
    "https://python.langchain.com/docs/tutorials/",
    "https://python.langchain.com/docs/tutorials/rag/",
    "https://python.langchain.com/docs/tutorials/chatbot/",
    "https://python.langchain.com/docs/tutorials/agents/",
]

LANGGRAPH_URLS = [
    # ----- Getting Started -----
    "https://langchain-ai.github.io/langgraph/",
    "https://langchain-ai.github.io/langgraph/tutorials/introduction/",
    
    # ----- Core Concepts -----
    "https://langchain-ai.github.io/langgraph/concepts/high_level/",
    "https://langchain-ai.github.io/langgraph/concepts/low_level/",
    "https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/",
    "https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/",
    "https://langchain-ai.github.io/langgraph/concepts/persistence/",
    "https://langchain-ai.github.io/langgraph/concepts/memory/",
    "https://langchain-ai.github.io/langgraph/concepts/streaming/",
    
    # ----- How-To Guides -----
    "https://langchain-ai.github.io/langgraph/how-tos/",
    "https://langchain-ai.github.io/langgraph/how-tos/state-model/",
    "https://langchain-ai.github.io/langgraph/how-tos/subgraph/",
    "https://langchain-ai.github.io/langgraph/how-tos/branching/",
    
    # ----- Tutorials -----
    "https://langchain-ai.github.io/langgraph/tutorials/",
    "https://langchain-ai.github.io/langgraph/tutorials/workflows/",
    "https://langchain-ai.github.io/langgraph/tutorials/multi-agent/",
]

LANGSMITH_URLS = [
    # ----- Getting Started -----
    "https://docs.smith.langchain.com/",
    "https://docs.smith.langchain.com/getting-started/quick-start",
    
    # ----- Observability (Debugging) -----
    "https://docs.smith.langchain.com/observability/",
    "https://docs.smith.langchain.com/observability/concepts",
    "https://docs.smith.langchain.com/observability/how_to_guides/tracing/",
    
    # ----- Evaluation (Testing) -----
    "https://docs.smith.langchain.com/evaluation/",
    "https://docs.smith.langchain.com/evaluation/concepts",
    "https://docs.smith.langchain.com/evaluation/how_to_guides/",
    
    # ----- Prompt Engineering -----
    "https://docs.smith.langchain.com/prompts/",
    "https://docs.smith.langchain.com/prompts/concepts",
]


def load_documents_from_urls(urls: List[str], source_name: str) -> List:
    """
    Load documents from a list of URLs.
    
    Args:
        urls: List of documentation URLs to load
        source_name: Name of the source (e.g., 'langchain', 'langgraph')
    
    Returns:
        List of loaded documents with metadata
    """
    print(f"\nLoading {source_name} documentation...")
    print(f"   URLs to load: {len(urls)}")
    
    all_docs = []
    
    for i, url in enumerate(urls, 1):
        try:
            print(f"   [{i}/{len(urls)}] Loading: {url[:50]}...")
            loader = WebBaseLoader(url)
            docs = loader.load()
            
            # Add source metadata to each document
            for doc in docs:
                doc.metadata["source_framework"] = source_name
                doc.metadata["source_url"] = url
            
            all_docs.extend(docs)
            all_docs.extend(docs)
            print(f"         Loaded {len(docs)} document(s)")
            
        except Exception as e:
            print(f"         Error loading {url}: {e}")
    
    print(f"   Total {source_name} documents loaded: {len(all_docs)}")
    return all_docs


def split_documents(documents: List, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
    """
    Split documents into smaller chunks for embedding.
    
    Why we split:
    - LLMs have context limits
    - Smaller chunks = more precise retrieval
    - Overlap ensures we don't lose context at chunk boundaries
    
    Args:
        documents: List of documents to split
        chunk_size: Maximum characters per chunk (default: 1000)
        chunk_overlap: Characters to overlap between chunks (default: 200)
    
    Returns:
        List of document chunks
    """
    print(f"\nSplitting {len(documents)} documents into chunks...")
    print(f"   Chunk size: {chunk_size} characters")
    print(f"   Chunk overlap: {chunk_overlap} characters")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  # Split by paragraphs first, then lines, etc.
    )
    
    chunks = text_splitter.split_documents(documents)
    
    chunks = text_splitter.split_documents(documents)
    
    print(f"   Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks


def load_all_documentation():
    """
    Load documentation from all three frameworks.
    
    Returns:
        Tuple of (all_documents, all_chunks)
    """
    print("\n" + "="*60)
    print("Starting Documentation Collection")
    print("="*60)
    
    all_docs = []
    
    # Load LangChain docs
    langchain_docs = load_documents_from_urls(LANGCHAIN_URLS, "langchain")
    all_docs.extend(langchain_docs)
    
    # Load LangGraph docs
    langgraph_docs = load_documents_from_urls(LANGGRAPH_URLS, "langgraph")
    all_docs.extend(langgraph_docs)
    
    # Load LangSmith docs
    langsmith_docs = load_documents_from_urls(LANGSMITH_URLS, "langsmith")
    all_docs.extend(langsmith_docs)
    
    print("\n" + "="*60)
    print(f"Total documents loaded: {len(all_docs)}")
    print("="*60)
    
    # Split into chunks
    chunks = split_documents(all_docs)
    
    print("\n" + "="*60)
    print("Documentation collection complete!")
    print(f"   Total chunks ready for embedding: {len(chunks)}")
    print("="*60)
    
    return all_docs, chunks


# This allows the file to be run directly for testing
if __name__ == "__main__":
    docs, chunks = load_all_documentation()
    
    # Show a sample chunk
    if chunks:
        print("\nSample chunk:")
        print("-"*40)
        print(f"Content: {chunks[0].page_content[:300]}...")
        print(f"Metadata: {chunks[0].metadata}")
