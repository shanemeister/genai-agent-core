import psycopg2
from psycopg2.extras import RealDictCursor

# Uses .pgpass — do NOT include password in code
DB_CONFIG = {
    "host": "192.168.4.25",
    "port": 5433,
    "dbname": "chathist",
    "user": "rshane",  # Replace with your actual username
}

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def init_chat_table():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,  -- 'user' or 'assistant'
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()

def load_chat_history(session_id: str):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT role, content FROM chat_history
                WHERE session_id = %s
                ORDER BY created_at ASC;
            """, (session_id,))
            return [{"role": row["role"], "content": row["content"]} for row in cur.fetchall()]

def save_message(session_id: str, role: str, content: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO chat_history (session_id, role, content)
                VALUES (%s, %s, %s);
            """, (session_id, role, content))
            conn.commit()