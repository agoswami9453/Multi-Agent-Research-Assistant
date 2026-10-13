import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from config import DB_PATH
from utils import get_logger

logger = get_logger("Database")

def get_db_connection():
    """Establishes and returns a connection to the SQLite database."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema if tables do not exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Table for conversations
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            session_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            title TEXT
        )
    """)
    
    # Table for messages
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            route TEXT,
            confidence REAL,
            sources TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES conversations(session_id)
        )
    """)
    
    # Table for uploaded document metadata
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            filepath TEXT,
            file_type TEXT,
            num_chunks INTEGER,
            upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully.")

# Run init_db on module load
init_db()

def create_session(session_id: str, title: str = "New Research Session"):
    """Creates a new conversation session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO conversations (session_id, title) VALUES (?, ?)",
        (session_id, title)
    )
    conn.commit()
    conn.close()

def get_latest_chat_session_id() -> Optional[str]:
    """Returns the session ID containing the most recently saved message."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT session_id FROM messages ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    return row["session_id"] if row else None

def add_message(session_id: str, role: str, content: str, route: Optional[str] = None, confidence: Optional[float] = None, sources: Optional[List[Dict[str, Any]]] = None):
    """Saves a user or assistant message to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    create_session(session_id)
    
    sources_json = json.dumps(sources) if sources else None
    
    cursor.execute(
        """
        INSERT INTO messages (session_id, role, content, route, confidence, sources)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (session_id, role, content, route, confidence, sources_json)
    )
    conn.commit()
    conn.close()

def get_chat_history(session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Retrieves recent chat history for a session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT role, content, route, confidence, sources, timestamp 
        FROM messages 
        WHERE session_id = ? 
        ORDER BY id DESC 
        LIMIT ?
        """,
        (session_id, limit)
    )
    rows = cursor.fetchall()
    rows.reverse()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            "role": row["role"],
            "content": row["content"],
            "route": row["route"],
            "confidence": row["confidence"],
            "sources": json.loads(row["sources"]) if row["sources"] else [],
            "timestamp": row["timestamp"]
        })
    return history

def save_document_metadata(filename: str, filepath: str, file_type: str, num_chunks: int):
    """Saves metadata for an uploaded document."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO document_metadata (filename, filepath, file_type, num_chunks)
        VALUES (?, ?, ?, ?)
        """,
        (filename, filepath, file_type, num_chunks)
    )
    conn.commit()
    conn.close()

def get_all_document_metadata() -> List[Dict[str, Any]]:
    """Returns a list of all uploaded documents."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT filename, filepath, file_type, num_chunks, upload_timestamp FROM document_metadata")
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "filename": row["filename"],
            "filepath": row["filepath"],
            "file_type": row["file_type"],
            "num_chunks": row["num_chunks"],
            "upload_timestamp": row["upload_timestamp"]
        }
        for row in rows
    ]

def clear_documents_metadata():
    """Clears all stored document metadata."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM document_metadata")
    conn.commit()
    conn.close()

def clear_session_history(session_id: str):
    """Clears history for a specific session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()
