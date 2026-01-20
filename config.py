"""
Configuration settings for the LangChain Expert Chatbot.
Loads environment variables from .env file.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# PostgreSQL Configuration
POSTGRES_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "database": os.getenv("POSTGRES_DB", "langchain_chatbot"),
    "user": os.getenv("POSTGRES_USER", "postgres"),
    "password": os.getenv("POSTGRES_PASSWORD", ""),
}

# Build database URL for SQLAlchemy
DATABASE_URL = f"postgresql://{POSTGRES_CONFIG['user']}:{POSTGRES_CONFIG['password']}@{POSTGRES_CONFIG['host']}:{POSTGRES_CONFIG['port']}/{POSTGRES_CONFIG['database']}"

# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "change-this-secret-key")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# ChromaDB Configuration
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")
CHROMA_COLLECTION_NAME = "langchain_docs"

# Documentation URLs to scrape
DOC_URLS = {
    "langchain": [
        "https://python.langchain.com/docs/introduction/",
        "https://python.langchain.com/docs/concepts/",
        "https://python.langchain.com/docs/tutorials/",
    ],
    "langgraph": [
        "https://langchain-ai.github.io/langgraph/",
        "https://langchain-ai.github.io/langgraph/concepts/",
        "https://langchain-ai.github.io/langgraph/tutorials/",
    ],
    "langsmith": [
        "https://docs.smith.langchain.com/",
        "https://docs.smith.langchain.com/concepts/",
    ],
}
