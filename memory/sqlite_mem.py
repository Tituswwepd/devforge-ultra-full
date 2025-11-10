import sqlite3
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DB = BASE / "storage" / "memory.sqlite"
DB.parent.mkdir(exist_ok=True)

def _init():
    with sqlite3.connect(DB) as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS convo (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP,
            role TEXT,
            text TEXT
        )
        """)

def mem_save(role: str, text: str):
    _init()
    with sqlite3.connect(DB) as c:
        c.execute("INSERT INTO convo(role,text) VALUES(?,?)", (role, text))

def mem_last(n:int=20):
    _init()
    with sqlite3.connect(DB) as c:
        cur = c.execute("SELECT ts, role, text FROM convo ORDER BY id DESC LIMIT ?", (n,))
        rows = cur.fetchall()
        return [{"ts":ts,"role":role,"text":text} for ts,role,text in rows]
