"""Database connection helper using psycopg 3 (modern psycopg) to align with backend virtualenv packages."""

import psycopg
from psycopg.rows import dict_row
from src.config import DATABASE_URL

def get_db_connection():
    """Return a new raw database connection returning dictionaries from rows with autocommit enabled."""
    conn = psycopg.connect(DATABASE_URL, row_factory=dict_row, autocommit=True)
    return conn

def get_healthy_db_connection(conn=None):
    """Ensure database connection is active and not in an aborted transaction state."""
    try:
        if conn is None or conn.closed:
            return get_db_connection()
        conn.rollback()
        return conn
    except Exception:
        return get_db_connection()

def init_db():
    """Verify database connection can be established."""
    try:
        conn = get_db_connection()
        conn.close()
        print("Database connection verified successfully.")
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        raise e
