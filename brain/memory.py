# brain/memory.py
import os, sqlite3, time, json, uuid
from typing import List, Tuple, Optional, Dict, Any

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "storage", "memory.sqlite")
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "storage"), exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  title TEXT DEFAULT '',
  created_at INTEGER,
  updated_at INTEGER
);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT,
  role TEXT,                   -- 'user' | 'assistant' | 'system'
  content TEXT,
  created_at INTEGER
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);

CREATE TABLE IF NOT EXISTS summaries (
  session_id TEXT PRIMARY KEY,
  rolling_summary TEXT,        -- compact running summary
  last_checkpoint INTEGER,     -- unix time
  progress_json TEXT           -- e.g. {"step": 6, "todo":["..."], "done":["..."]}
);
"""

def _con():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def init():
    con = _con()
    try:
        for stmt in SCHEMA.strip().split(");"):
            s = stmt.strip()
            if s:
                con.execute(s + ");")
        con.commit()
    finally:
        con.close()

def ensure_session(session_id: Optional[str]) -> str:
    sid = session_id or str(uuid.uuid4())
    now = int(time.time())
    con = _con()
    try:
        cur = con.execute("SELECT id FROM sessions WHERE id=?", (sid,))
        row = cur.fetchone()
        if not row:
            con.execute("INSERT INTO sessions(id, created_at, updated_at) VALUES(?,?,?)", (sid, now, now))
        else:
            con.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, sid))
        con.commit()
    finally:
        con.close()
    return sid

def append_message(session_id: str, role: str, content: str):
    con = _con()
    try:
        con.execute(
            "INSERT INTO messages(session_id, role, content, created_at) VALUES(?,?,?,?)",
            (session_id, role, content, int(time.time()))
        )
        con.execute("UPDATE sessions SET updated_at=? WHERE id=?", (int(time.time()), session_id))
        con.commit()
    finally:
        con.close()

def fetch_recent_context(session_id: str, k: int = 12) -> List[Tuple[str,str]]:
    con = _con()
    try:
        cur = con.execute(
            "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, k)
        )
        rows = cur.fetchall()[::-1]
        return rows
    finally:
        con.close()

def get_summary(session_id: str) -> Dict[str, Any]:
    con = _con()
    try:
        cur = con.execute("SELECT rolling_summary, progress_json FROM summaries WHERE session_id=?", (session_id,))
        row = cur.fetchone()
        if not row:
            return {"rolling_summary":"", "progress": {"step":0, "todo":[], "done":[]}}
        rs, pj = row
        return {"rolling_summary": rs or "", "progress": json.loads(pj or '{"step":0,"todo":[],"done":[]}')}
    finally:
        con.close()

def save_summary(session_id: str, rolling_summary: str, progress: Dict[str,Any]):
    con = _con()
    try:
        payload = (rolling_summary, json.dumps(progress), int(time.time()), session_id)
        cur = con.execute("SELECT 1 FROM summaries WHERE session_id=?", (session_id,))
        if cur.fetchone():
            con.execute("UPDATE summaries SET rolling_summary=?, progress_json=?, last_checkpoint=? WHERE session_id=?",
                        payload)
        else:
            con.execute("INSERT INTO summaries(session_id, rolling_summary, progress_json, last_checkpoint) VALUES(?,?,?,?)",
                        (session_id, rolling_summary, json.dumps(progress), int(time.time())))
        con.commit()
    finally:
        con.close()

def wipe_session(session_id: str):
    con = _con()
    try:
        con.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        con.execute("DELETE FROM summaries WHERE session_id=?", (session_id,))
        con.execute("UPDATE sessions SET title='', updated_at=? WHERE id=?", (int(time.time()), session_id))
        con.commit()
    finally:
        con.close()
