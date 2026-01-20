"""
RAG (Retrieval Augmented Generation) Chain Module

This is the BRAIN of your chatbot! Here's how it works:

1. User asks a question
2. We convert the question to a vector (embedding)
3. We find similar documents in the vector store
4. We send the question + relevant documents to the LLM
5. LLM generates an answer based on the documentation

Why RAG?
- LLMs don't know about specific documentation
- RAG gives the LLM "context" to answer accurately
- Reduces hallucination (making things up)
"""

import os
import sys

# Add parent directory to path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from src.embeddings import load_vectorstore


# The prompt template - this is CRITICAL for good answers!
RAG_PROMPT_TEMPLATE = """You are an expert assistant specialized in LangChain, LangGraph, and LangSmith frameworks.

Your job is to help developers understand and use these frameworks effectively.

INSTRUCTIONS:
1. First, try to answer based on the provided context from documentation
2. If the context is relevant, use it to provide a detailed answer with code examples
3. If the context doesn't directly answer the question but is related, use your knowledge of LangChain/LangGraph/LangSmith to help
4. Include code examples when relevant
5. Mention which framework (LangChain, LangGraph, or LangSmith) the answer relates to

CONTEXT FROM DOCUMENTATION:
{context}


CHAT HISTORY (Previous context - IGNORE if unrelated to new question):
{chat_history}

LATEST USER QUESTION (Focus your answer here):
{question}

    Provide a comprehensive, beginner-friendly explanation.
    - Break down complex concepts into simple terms.
    - Uses analogies if helpful.
    - Provide a detailed answer that teaches    - The user wants a detailed, comprehensive answer.
    - If the "Context" is just a code snippet (like a specific function), you MUST first define the general concept using your own knowledge, then explain how the code relates to it.
    - If the answer is completely missing from the context and you cannot infer it, say "I don't have information about that in my specific documentation, but generally speaking..." and provide a helpful answer based on your training.
    - Do NOT hallucinate specific library features that don't exist. Use your general knowledge only if you are 100% sure."""


def format_docs(docs):
    """
    Format retrieved documents into a single string for the prompt.
    
    Args:
        docs: List of retrieved documents
    
    Returns:
        Formatted string with document contents and sources
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get('source_framework', 'unknown')
        url = doc.metadata.get('source_url', '')
        content = doc.page_content
        
        # Skip redirect pages and pages with very little content
        if 'Redirecting' in content:
            continue
        if len(content.strip()) < 100:  # Skip very short content
            continue
        if 'Skip to main content' in content and len(content) < 200:
            continue
            
        # Clean up the content - remove navigation noise
        lines = content.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            # Skip navigation-like lines
            if line in ['', 'Docs', 'Search', 'Home', 'API Reference', 'Tutorials']:
                continue
            clean_lines.append(line)
        
        clean_content = ' '.join(clean_lines)
        
        if len(clean_content) > 50:  # Only add if there's meaningful content
            formatted.append(f"[Document {len(formatted)+1} - {source.upper()}]\n{clean_content}\n")
        
        # Limit to 5 good documents to avoid too much context
        if len(formatted) >= 5:
            break
    
    return "\n---\n".join(formatted) if formatted else "No relevant documentation found in the knowledge base."


def format_chat_history(history):
    """
    Format chat history for the prompt.
    
    Args:
        history: List of (user_message, ai_message) tuples
    
    Returns:
        Formatted string of chat history
    """
    if not history:
        return "No previous conversation."
    
    formatted = []
    for human, ai in history:
        formatted.append(f"User: {human}")
        formatted.append(f"Assistant: {ai}")
    
    return "\n".join(formatted)


def create_rag_chain(vectorstore=None):
    """
    Create the RAG chain that combines retrieval + generation.
    
    This is the core pipeline:
    Question → Retrieve docs → Format prompt → Generate answer
    
    Args:
        vectorstore: Optional vectorstore, will load from disk if not provided
    
    Returns:
        Configured RAG chain ready for queries
    """
    # Load vector store if not provided
    if vectorstore is None:
        vectorstore = load_vectorstore()
        if vectorstore is None:
            raise ValueError("No vector store found. Run embeddings.py first!")
    
    # Create retriever - fetches top 10 most relevant documents
    # We get more than we need, then filter out bad ones
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    # Create LLM connection to Groq
    # Use model from environment variable, default to llama3-8b-8192 if not set
    model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        print("ERROR: GROQ_API_KEY not found in .env file!")
        # Fallback to direct input if env not set (for testing)
        # api_key = "gsk_..."
    
    print(f"Loading Groq model: {model_name}")
    
    llm = ChatGroq(
        model=model_name,
        temperature=0.1,  # Reduced temperature for more factual answers
        api_key=api_key
    )
    
    # Helper function to get context from retriever
    def get_context(input_dict):
        """Retrieve relevant documents based on the question."""
        question = input_dict["question"]
        docs = retriever.invoke(question)
        
        # Debug output
        print(f"\n[DEBUG] Retrieved {len(docs)} documents for query: '{question}'")
        for i, doc in enumerate(docs[:3]):  # Show first 3
            content_preview = doc.page_content[:100].replace('\n', ' ')
            print(f"  Doc {i+1}: {content_preview}...")
        
        context = format_docs(docs)
        print(f"[DEBUG] Formatted context length: {len(context)} chars")
        print(f"[DEBUG] Context preview: {context[:200]}...")
        
        return context
    
    # Build the chain using proper input handling
    rag_chain = (
        {
            "context": lambda x: get_context(x),
            "chat_history": lambda x: format_chat_history(x.get("chat_history", [])),
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain, retriever


def ask_question(question: str, chat_history: list = None):
    """
    Ask a question to the RAG chatbot.
    
    Args:
        question: The user's question
        chat_history: List of (user_message, ai_message) tuples
    
    Returns:
        The AI's response
    """
    chat_history = chat_history or []
    
    print(f"\nProcessing: '{question[:50]}...'")
    
    # Create the chain
    rag_chain, retriever = create_rag_chain()
    
    # Get the answer
    response = rag_chain.invoke({
        "question": question,
        "chat_history": chat_history
    })
    
    return response


# Interactive test when run directly
if __name__ == "__main__":
    print("\n" + "#"*60)
    print("#  LangChain Expert Chatbot - Test Mode")
    print("#"*60)
    print("\nLoading RAG chain...")
    
    try:
        rag_chain, retriever = create_rag_chain()
        print("RAG chain ready!\n")
        
        chat_history = []
        
        print("Ask questions about LangChain, LangGraph, or LangSmith.")
        print("Type 'quit' to exit.\n")
        print("-"*60)
        
        while True:
            question = input("\n You: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not question:
                continue
            
            # Get answer
            response = rag_chain.invoke({
                "question": question,
                "chat_history": chat_history
            })
            
            print(f"\nAssistant: {response}")
            
            # Add to history
            chat_history.append((question, response))
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have run embeddings.py first to create the vector store!")
