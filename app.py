"""
LangChain Expert Chatbot - Streamlit Web Application

Features:
- User registration and login
- ChatGPT-style conversation management
- Multiple active conversations
- Persistent history for each chat

Run with: streamlit run app.py
"""

import streamlit as st
import os
import sys
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database import init_database, create_conversation, get_user_conversations, get_conversation_messages, add_message, delete_conversation, delete_all_conversations
from src.auth import register_user, login_user
from src.rag_chain import create_rag_chain
from src.embeddings import load_vectorstore

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="LangChain Expert Chatbot",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main { padding: 1rem; }
    .header-title {
        color: #1976d2;
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Conversation buttons in sidebar */
    .stButton button {
        text-align: left;
        padding-left: 10px;
    }
    
    .current-chat {
        background-color: #e3f2fd;
        border-left: 3px solid #1976d2;
        padding: 5px;
        margin: 2px 0;
    }
    
    /* Timestamp styling */
    .msg-timestamp {
        font-size: 0.7rem;
        color: #888;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# SESSION STATE
# ============================================
def init_session_state():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "user_id" not in st.session_state:
        st.session_state.user_id = None
    if "username" not in st.session_state:
        st.session_state.username = None
    if "current_conversation_id" not in st.session_state:
        st.session_state.current_conversation_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "vectorstore_loaded" not in st.session_state:
        st.session_state.vectorstore_loaded = False
    if "show_register" not in st.session_state:
        st.session_state.show_register = False


# ============================================
# AUTHENTICATION
# ============================================
def show_login_page():
    st.markdown('<h1 class="header-title">🦜 LangChain Chatbot</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("🔐 Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login", use_container_width=True):
                success, msg, token, user_id = login_user(username, password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.user_id = user_id
                    st.session_state.username = username
                    # FORCE NEW CHAT ON LOGIN
                    st.session_state.current_conversation_id = None
                    st.session_state.messages = []
                    # Don't create chat yet - wait for first message
                    st.rerun()
                else:
                    st.error(msg)
        
        st.markdown("Don't have an account?")
        if st.button("📝 Register Here"):
            st.session_state.show_register = True
            st.rerun()


def show_register_page():
    st.markdown('<h1 class="header-title">🦜 Create Account</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("register_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            confirm = st.text_input("Confirm Password", type="password")
            if st.form_submit_button("Register", use_container_width=True):
                if password != confirm:
                    st.error("Passwords don't match!")
                else:
                    success, msg, user_id = register_user(username, password)
                    if success:
                        st.success("Registered! Login now.")
                        st.session_state.show_register = False
                        st.rerun()
                    else:
                        st.error(msg)
        
        if st.button("⬅️ Back"):
            st.session_state.show_register = False
            st.rerun()


# ============================================
# CHAT LOGIC
# ============================================
def load_current_chat():
    """Load messages for the selected conversation."""
    if st.session_state.current_conversation_id:
        msgs = get_conversation_messages(st.session_state.current_conversation_id)
        st.session_state.messages = [{
            "role": m.role,
            "content": m.content,
            "timestamp": m.created_at
        } for m in msgs]
    else:
        st.session_state.messages = []


def create_new_chat():
    """Create a new conversation."""
    conv_id = create_conversation(st.session_state.user_id, "New Chat")
    st.session_state.current_conversation_id = conv_id
    st.session_state.messages = []
    st.rerun()


def show_chat_page():
    # --- SIDEBAR: CONVERSATION LIST ---
    with st.sidebar:
        st.title(f"👤 {st.session_state.username}")
        
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            st.session_state.current_conversation_id = None
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("### Your Conversations")
        
        # Get all conversations for user
        conversations = get_user_conversations(st.session_state.user_id)
        
        if not conversations:
            st.info("No saved conversations.")
        else:
            for conv in conversations:
                # Highlight current chat
                if conv.id == st.session_state.current_conversation_id:
                    label = f"📂 {conv.title}"
                else:
                    label = conv.title
                
                # Create a clickable button for each chat
                col1, col2 = st.columns([4, 1])
                with col1:
                    if st.button(label, key=f"conv_{conv.id}", use_container_width=True):
                        st.session_state.current_conversation_id = conv.id
                        load_current_chat()
                        st.rerun()
                with col2:
                    if st.button("🗑️", key=f"del_{conv.id}"):
                        delete_conversation(conv.id)
                        if st.session_state.current_conversation_id == conv.id:
                            st.session_state.current_conversation_id = None
                            st.session_state.messages = []
                        st.rerun()
        
        st.markdown("---")
        if st.button("🗑️ Clear All History", use_container_width=True, type="secondary"):
            delete_all_conversations(st.session_state.user_id)
            st.session_state.current_conversation_id = None
            st.session_state.messages = []
            st.rerun()
            
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()
    
    # --- MAIN CHAT AREA ---
    st.markdown('<h1 class="header-title">🦜 LangChain Expert</h1>', unsafe_allow_html=True)
    
    # Ensure a conversation exists
    # If no conversation selected, we are in "New Chat" mode (Draft)
    if not st.session_state.current_conversation_id:
        pass  # Do nothing, waiting for user input to create chat

    # Load RAG Model
    if not st.session_state.vectorstore_loaded:
        with st.spinner("Initializing AI..."):
            try:
                st.session_state.rag_chain, _ = create_rag_chain()
                st.session_state.vectorstore_loaded = True
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

    # Display Messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            st.markdown(f"<div class='msg-timestamp'>{msg['timestamp'].strftime('%H:%M')}</div>", unsafe_allow_html=True)
    
    # Input
    if prompt := st.chat_input("Ask a question..."):
        # 1. Add User Message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now()
        })
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Lazy create chat if it doesn't exist (Draft Mode)
        if not st.session_state.current_conversation_id:
             st.session_state.current_conversation_id = create_conversation(st.session_state.user_id, "New Chat")
             
        add_message(st.session_state.current_conversation_id, "user", prompt)
        
        # 2. Get AI Response
        with st.chat_message("assistant"):
            # Prepare history for RAG
            history = [(m["content"], st.session_state.messages[i+1]["content"]) 
                      for i, m in enumerate(st.session_state.messages[:-1]) 
                      if m["role"] == "user" and i+1 < len(st.session_state.messages)]
            
            # Create a generator for streaming
            def stream_response():
                try:
                    # Stream the response chunk by chunk
                    stream = st.session_state.rag_chain.stream({
                        "question": prompt,
                        "chat_history": history[-3:] # Pass last 3 turns
                    })
                    
                    full_response = ""
                    for chunk in stream:
                        full_response += chunk
                        yield chunk
                        
                    # Save complete response to session state after streaming
                    st.session_state.temp_response = full_response
                    
                except Exception as e:
                    yield f"Error: {e}"

            # Stream output to UI
            response = st.write_stream(stream_response)
        
        # 3. Add AI Message (use the response returned by write_stream)
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now()
        })
        add_message(st.session_state.current_conversation_id, "assistant", response)
        
        # Rerun to update the conversation title in sidebar if it's the first message
        if len(st.session_state.messages) <= 2:
            st.rerun()


def main():
    init_session_state()
    init_database()
    
    if st.session_state.logged_in:
        show_chat_page()
    elif st.session_state.show_register:
        show_register_page()
    else:
        show_login_page()


if __name__ == "__main__":
    main()
