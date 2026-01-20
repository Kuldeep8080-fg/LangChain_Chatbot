"""
Authentication Module - User Registration, Login, and Session Management

This module handles:
1. Password hashing with bcrypt
2. User registration and login
3. JWT token generation and validation

Security Best Practices Implemented:
- Passwords are hashed before storage (never store plain text!)
- JWT tokens expire after 24 hours
- Tokens are signed with a secret key
"""

import os
import sys
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bcrypt
import jwt
from dotenv import load_dotenv
from src.database import get_session, User

# Load environment variables
load_dotenv()

# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "default-secret-change-this")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24


# ============================================
# PASSWORD HASHING
# ============================================

def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Why bcrypt?
    - Designed specifically for passwords
    - Intentionally slow (prevents brute force attacks)
    - Includes salt automatically (prevents rainbow table attacks)
    
    Args:
        password: Plain text password
    
    Returns:
        Hashed password string
    """
    # Generate salt and hash the password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        password: Plain text password to verify
        hashed: The stored hash to check against
    
    Returns:
        True if password matches, False otherwise
    """
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))


# ============================================
# JWT TOKEN MANAGEMENT
# ============================================

def create_token(user_id: int, username: str) -> str:
    """
    Create a JWT token for a user.
    
    The token contains:
    - user_id: To identify the user
    - username: For display purposes
    - exp: Expiration time
    
    Args:
        user_id: The user's database ID
        username: The user's username
    
    Returns:
        JWT token string
    """
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    }
    
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token


def verify_token(token: str) -> dict:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token string
    
    Returns:
        Decoded payload dict, or None if invalid/expired
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None  # Token has expired
    except jwt.InvalidTokenError:
        return None  # Invalid token


# ============================================
# USER MANAGEMENT
# ============================================

def register_user(username: str, password: str) -> tuple:
    """
    Register a new user.
    
    Args:
        username: Desired username (must be unique)
        password: Plain text password
    
    Returns:
        Tuple of (success: bool, message: str, user_id: int or None)
    """
    session = get_session()
    
    try:
        # Check if username already exists
        existing = session.query(User).filter(User.username == username).first()
        if existing:
            return False, "Username already exists", None
        
        # Validate username
        if len(username) < 3:
            return False, "Username must be at least 3 characters", None
        
        if len(password) < 6:
            return False, "Password must be at least 6 characters", None
        
        # Hash password and create user
        password_hash = hash_password(password)
        user = User(username=username, password_hash=password_hash)
        
        session.add(user)
        session.commit()
        
        return True, "Registration successful!", user.id
        
    except Exception as e:
        session.rollback()
        return False, f"Registration failed: {str(e)}", None
    finally:
        session.close()


def login_user(username: str, password: str) -> tuple:
    """
    Authenticate a user and return a token.
    
    Args:
        username: The username
        password: Plain text password
    
    Returns:
        Tuple of (success: bool, message: str, token: str or None, user_id: int or None)
    """
    session = get_session()
    
    try:
        # Find the user
        user = session.query(User).filter(User.username == username).first()
        
        if not user:
            return False, "Invalid username or password", None, None
        
        # Verify password
        if not verify_password(password, user.password_hash):
            return False, "Invalid username or password", None, None
        
        # Create token
        token = create_token(user.id, user.username)
        
        return True, "Login successful!", token, user.id
        
    except Exception as e:
        return False, f"Login failed: {str(e)}", None, None
    finally:
        session.close()


def get_user_by_id(user_id: int) -> User:
    """Get a user by their ID."""
    session = get_session()
    try:
        return session.query(User).filter(User.id == user_id).first()
    finally:
        session.close()


# Test the module when run directly
if __name__ == "__main__":
    print("\n" + "#"*50)
    print("#  Authentication Module Test")
    print("#"*50)
    
    # First, make sure database tables exist
    from src.database import init_database
    init_database()
    
    print("\n--- Testing Registration ---")
    success, msg, user_id = register_user("testuser", "password123")
    print(f"Register: {msg} (user_id: {user_id})")
    
    print("\n--- Testing Login ---")
    success, msg, token, user_id = login_user("testuser", "password123")
    print(f"Login: {msg}")
    if token:
        print(f"Token: {token[:50]}...")
        
        # Test token verification
        payload = verify_token(token)
        print(f"Token payload: {payload}")
    
    print("\n--- Testing Wrong Password ---")
    success, msg, token, user_id = login_user("testuser", "wrongpassword")
    print(f"Login with wrong password: {msg}")
    
    print("\n✅ Auth module is working!")
