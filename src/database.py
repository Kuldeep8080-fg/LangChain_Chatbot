"""
Database Module - PostgreSQL Connection and Models

Models:
1. User - For authentication
2. Conversation - Groups messages into separate chats
3. ChatMessage - Individual messages within conversations

This allows ChatGPT-style conversation management!
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()


# ============================================
# DATABASE MODELS
# ============================================

class User(Base):
    """User model for authentication."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}')>"


class Conversation(Base):
    """
    Conversation model - groups messages into separate chats.
    Like ChatGPT, each "New Chat" creates a new conversation.
    """
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(100), default="New Chat")  # Auto-generated from first message
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("ChatMessage", back_populates="conversation", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, title='{self.title}')>"


class ChatMessage(Base):
    """Individual message within a conversation."""
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, index=True)
    role = Column(String(10), nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    def __repr__(self):
        return f"<ChatMessage(id={self.id}, role='{self.role}')>"


# ============================================
# DATABASE CONNECTION
# ============================================

def get_database_url():
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "langchain_chatbot")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def get_engine():
    return create_engine(get_database_url(), echo=False)


def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def init_database():
    """Initialize database - create all tables."""
    engine = get_engine()
    try:
        Base.metadata.create_all(engine)
        return True
    except Exception as e:
        print(f"Database error: {e}")
        return False


# ============================================
# CONVERSATION FUNCTIONS
# ============================================

def create_conversation(user_id: int, title: str = "New Chat") -> int:
    """Create a new conversation and return its ID."""
    session = get_session()
    try:
        conv = Conversation(user_id=user_id, title=title)
        session.add(conv)
        session.commit()
        return conv.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def get_user_conversations(user_id: int, limit: int = 20):
    """Get all conversations for a user, most recent first."""
    session = get_session()
    try:
        conversations = session.query(Conversation)\
            .filter(Conversation.user_id == user_id)\
            .order_by(Conversation.updated_at.desc())\
            .limit(limit)\
            .all()
        return conversations
    finally:
        session.close()


def get_conversation_messages(conversation_id: int):
    """Get all messages for a conversation."""
    session = get_session()
    try:
        messages = session.query(ChatMessage)\
            .filter(ChatMessage.conversation_id == conversation_id)\
            .order_by(ChatMessage.created_at.asc())\
            .all()
        return messages
    finally:
        session.close()


def add_message(conversation_id: int, role: str, content: str) -> int:
    """Add a message to a conversation."""
    session = get_session()
    try:
        msg = ChatMessage(conversation_id=conversation_id, role=role, content=content)
        session.add(msg)
        
        # Update conversation title if it's the first user message
        conv = session.query(Conversation).filter(Conversation.id == conversation_id).first()
        if conv and conv.title == "New Chat" and role == "user":
            # Set title to first 50 chars of first message
            conv.title = content[:50] + "..." if len(content) > 50 else content
        
        # Update conversation timestamp
        conv.updated_at = datetime.utcnow()
        
        session.commit()
        return msg.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def delete_conversation(conversation_id: int):
    """Delete a conversation and all its messages."""
    session = get_session()
    try:
        # Custom delete to handle foreign keys if cascade doesn't work
        # First delete all messages in this conversation
        session.query(ChatMessage)\
            .filter(ChatMessage.conversation_id == conversation_id)\
            .delete(synchronize_session=False)
            
        # Then delete the conversation
        session.query(Conversation)\
            .filter(Conversation.id == conversation_id)\
            .delete(synchronize_session=False)
            
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def delete_all_conversations(user_id: int):
    """Delete all conversations for a user."""
    session = get_session()
    try:
        # First get all conversation IDs for this user
        conv_ids = session.query(Conversation.id).filter(Conversation.user_id == user_id).all()
        conv_ids = [c[0] for c in conv_ids]
        
        if conv_ids:
            # Delete all messages for these conversations
            session.query(ChatMessage)\
                .filter(ChatMessage.conversation_id.in_(conv_ids))\
                .delete(synchronize_session=False)
        
        # Then delete the conversations
        session.query(Conversation)\
            .filter(Conversation.user_id == user_id)\
            .delete(synchronize_session=False)
            
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


# Legacy functions for backward compatibility
def add_chat_message(user_id: int, role: str, message: str):
    """Legacy function - now requires conversation_id in app.py"""
    pass  # Will be handled in app.py


def get_user_chat_history(user_id: int, limit: int = 50):
    """Legacy function - now use get_conversation_messages"""
    return []


def clear_user_chat_history(user_id: int):
    """Legacy - now use delete_all_conversations"""
    return delete_all_conversations(user_id)


if __name__ == "__main__":
    print("Initializing database...")
    success = init_database()
    if success:
        print("Database ready!")
    else:
        print("Database setup failed!")
