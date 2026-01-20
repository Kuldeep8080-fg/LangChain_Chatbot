"""
Script to inspect the PostgreSQL database contents.
Reflects the current schema with Conversations.
"""
import os
import sys
from tabulate import tabulate  # You might need to install this: pip install tabulate
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database import get_session, User, Conversation, ChatMessage

def inspect_database():
    session = get_session()
    
    print("\n" + "="*80)
    print("🐘 POSTGRESQL DATABASE INSPECTOR")
    print("="*80)
    
    # 1. Users
    users = session.query(User).all()
    print(f"\n👥 USERS ({len(users)})")
    if users:
        data = [[u.id, u.username, u.created_at.strftime("%Y-%m-%d %H:%M")] for u in users]
        print(tabulate(data, headers=["ID", "Username", "Joined"], tablefmt="simple"))
    else:
        print("   No users found.")

    # 2. Conversations
    conversations = session.query(Conversation).all()
    print(f"\n📂 CONVERSATIONS ({len(conversations)})")
    if conversations:
        data = [[c.id, c.user_id, c.title[:30] + "...", c.updated_at.strftime("%Y-%m-%d %H:%M")] for c in conversations]
        print(tabulate(data, headers=["ID", "User ID", "Title", "Last Active"], tablefmt="simple"))
    else:
        print("   No conversations found.")

    # 3. Messages (Last 10)
    print(f"\n💬 RECENT MESSAGES (Last 10)")
    messages = session.query(ChatMessage).order_by(ChatMessage.created_at.desc()).limit(10).all()
    if messages:
        # Reverse to show chronological order
        messages.reverse()
        data = []
        for m in messages:
            content = m.content[:50] + "..." if len(m.content) > 50 else m.content.replace('\n', ' ')
            data.append([m.id, m.conversation_id, m.role.upper(), content])
            
        print(tabulate(data, headers=["ID", "Conv ID", "Role", "Message"], tablefmt="simple"))
    else:
        print("   No messages found.")
        
    session.close()

if __name__ == "__main__":
    inspect_database()
