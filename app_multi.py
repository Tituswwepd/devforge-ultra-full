# app_multi.py — DevForge Ultra Chat (Multi-Provider ChatGPT-style App)
import os, json, sqlite3, time, hashlib, httpx, asyncio
from typing import AsyncIterator, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

# ---------------- CONFIG ----------------
APP_NAME = "DevForge Ultra Chat"
DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "openai").lower()

OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MISTRAL_API_KEY   = os.getenv("MISTRAL_API_KEY", "")
COHERE_API_KEY    = os.getenv("COHERE_API_KEY", "")

OPENAI_BASE   = "https://api.openai.com/v1"
MODEL_OPENAI  = "gpt-4o-mini"
MODEL_ANTHROPIC = "claude-3-5-sonnet-latest"
MODEL_MISTRAL   = "mistral-large-latest"
MODEL_COHERE    = "command-r-plus"

SYSTEM_PROMPT = (
    "You are DevForge Ultra, a helpful AI assistant similar to ChatGPT. "
    "Be concise, friendly, and safe. Include code examples when relevant."
)

DB_PATH = os.getenv("DB_PATH","/var/data/memory_multi.db")
HISTORY_LIMIT = 40

# ---------------- APP ----------------
app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- DATABASE ----------------
def _db():
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("""CREATE TABLE IF NOT EXISTS chats(
        chat_id TEXT PRIMARY KEY, user_id TEXT, title TEXT, created REAL, updated REAL
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS messages(
        id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id TEXT, role TEXT, content TEXT, ts REAL
    )""")
    return con

def new_chat(user_id, title="New chat"):
    cid = hashlib.sha1(f"{user_id}-{time.time()}".encode()).hexdigest()[:16]
    now = time.time()
    with _db() as c:
        c.execute("INSERT INTO chats VALUES(?,?,?,?,?)", (cid, user_id, title, now, now))
    return cid

def append_msg(cid, role, content):
    ts = time.time()
    with _db() as c:
        c.execute("INSERT INTO messages(chat_id,role,content,ts) VALUES(?,?,?,?)", (cid, role, content, ts))
        c.execute("UPDATE chats SET updated=? WHERE chat_id=?", (ts, cid))

def get_history(cid, limit=HISTORY_LIMIT):
    with _db() as c:
        cur = c.execute("SELECT role,content FROM messages WHERE chat_id=? ORDER BY ts ASC LIMIT ?", (cid, limit))
        return [{"role": r, "content": c} for r, c in cur.fetchall()]

def build_messages(history, msg):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-HISTORY_LIMIT:])
    messages.append({"role": "user", "content": msg})
    return messages

# ---------------- PROVIDERS ----------------
async def openai_stream(messages, model=None):
    if not OPENAI_API_KEY:
        raise HTTPException(500, "OPENAI_API_KEY not set")
    url = f"{OPENAI_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model or MODEL_OPENAI, "messages": messages, "stream": True}
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content")
                except Exception:
                    delta = None
                if delta:
                    yield delta

async def anthropic_complete(messages, model=None):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(500, "ANTHROPIC_API_KEY not set")
    text = [m["content"] for m in messages if m["role"] == "user"][-1]
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model or MODEL_ANTHROPIC,
        "system": SYSTEM_PROMPT,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": [{"type": "text", "text": text}]}],
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        j = r.json()
        return "".join([b.get("text", "") for b in j.get("content", [])])

async def mistral_complete(messages, model=None):
    if not MISTRAL_API_KEY:
        raise HTTPException(500, "MISTRAL_API_KEY not set")
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model or MODEL_MISTRAL, "messages": messages, "temperature": 0.7}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

async def cohere_complete(messages, model=None):
    if not COHERE_API_KEY:
        raise HTTPException(500, "COHERE_API_KEY not set")
    text = [m["content"] for m in messages if m["role"] == "user"][-1]
    url = "https://api.cohere.com/v1/chat"
    headers = {"Authorization": f"Bearer {COHERE_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model or MODEL_COHERE, "message": text}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        j = r.json()
        return j.get("text") or j.get("response", "")

async def provider_stream(provider, messages, model=None):
    p = (provider or DEFAULT_PROVIDER).lower()
    if p == "openai":
        async for chunk in openai_stream(messages, model):
            yield chunk
    else:
        if p == "anthropic":
            text = await anthropic_complete(messages, model)
        elif p == "mistral":
            text = await mistral_complete(messages, model)
        elif p == "cohere":
            text = await cohere_complete(messages, model)
        else:
            text = "[Unsupported provider]"
        for i in range(0, len(text), 100):
            yield text[i:i+100]

# ---------------- ROUTES ----------------
@app.get("/api/health")
async def health():
    return {"ok": True, "provider": DEFAULT_PROVIDER}

@app.post("/api/new_chat")
async def api_new_chat(payload: dict):
    uid = payload.get("user_id")
    if not uid:
        raise HTTPException(400, "user_id required")
    return {"chat_id": new_chat(uid)}

@app.post("/api/chat")
async def api_chat(payload: dict):
    uid = payload.get("user_id")
    msg = payload.get("message")
    cid = payload.get("chat_id")
    prov = payload.get("provider") or DEFAULT_PROVIDER
    if not uid or not msg:
        raise HTTPException(400, "user_id and message required")
    if not cid:
        cid = new_chat(uid)
    hist = get_history(cid)
    messages = build_messages(hist, msg)
    append_msg(cid, "user", msg)

    async def gen():
        acc = []
        try:
            async for chunk in provider_stream(prov, messages):
                acc.append(chunk)
                yield json.dumps({"delta": chunk}).encode() + b"\n"
        finally:
            if acc:
                append_msg(cid, "assistant", "".join(acc))
    return EventSourceResponse(gen(), media_type="text/event-stream")

# ---------------- SIMPLE UI ----------------
@app.get("/ui")
async def ui():
    html = """<!DOCTYPE html><html><head><meta charset="utf-8"/>
<title>DevForge Ultra Chat</title>
<style>
body{background:#0b1220;color:#e6ecff;font-family:Inter,system-ui,Segoe UI,Arial;margin:0}
header{padding:12px 16px;border-bottom:1px solid #1e2740}
main{max-width:980px;margin:0 auto;padding:16px}
#log{white-space:pre-wrap;border:1px solid #22304d;background:#0a1324;border-radius:10px;padding:12px;height:65vh;overflow:auto}
.controls{display:flex;gap:8px;margin:10px 0}
textarea{flex:1;background:#0a1324;color:#fff;border:1px solid #22304d;border-radius:10px;padding:10px;height:70px}
select,button,input{background:#0a1324;color:#9fb0d0;border:1px solid #22304d;border-radius:10px;padding:10px}
button{background:#0d5bd8;color:#fff;border:0}
</style></head>
<body>
<header><b>DevForge Ultra Chat</b> — Multi-Provider</header>
<main>
<div class="controls">
<select id="prov">
<option value="openai" selected>OpenAI (stream)</option>
<option value="anthropic">Anthropic</option>
<option value="mistral">Mistral</option>
<option value="cohere">Cohere</option>
</select>
<input id="uid" placeholder="user id (auto)" style="width:220px"/>
<button onclick="newChat()">+ New Chat</button>
</div>
<div id="log"></div>
<div class="controls">
<textarea id="msg" placeholder="Message…"></textarea>
<button onclick="send()">Send</button>
</div>
</main>
<script>
const API = location.origin.replace(/\\/$/,'');
let uid = localStorage.getItem('df_uid') || crypto.randomUUID();
localStorage.setItem('df_uid', uid);
document.getElementById('uid').value = uid;
let chat = null;
async function newChat(){
  const r = await fetch(API+'/api/new_chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({user_id:uid})});
  const j = await r.json(); chat = j.chat_id;
  document.getElementById('log').textContent='';
}
function add(role,txt){
  const log=document.getElementById('log');
  log.textContent += (role==='user'?'\\nYou: ':'')+txt;
  log.scrollTop=log.scrollHeight;
}
async function send(){
  const prov=document.getElementById('prov').value;
  uid=document.getElementById('uid').value||uid;
  localStorage.setItem('df_uid',uid);
  const msg=document.getElementById('msg').value.trim();
  if(!msg)return;
  document.getElementById('msg').value='';
  if(!chat){await newChat();}
  add('user',msg+'\\n');
  const res=await fetch(API+'/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({user_id:uid,chat_id:chat,message:msg,provider:prov})});
  if(!res.ok||!res.body){add('assistant','\\n[Error: '+res.statusText+']\\n');return;}
  const rd=res.body.getReader();const dec=new TextDecoder();
  while(true){const {value,done}=await rd.read();if(done)break;
    const chunk=dec.decode(value,{stream:true});
    for(const line of chunk.split('\\n')){if(!line)continue;
      try{const j=JSON.parse(line);if(j.delta){add('assistant',j.delta);}}catch(e){}}
  }
}
</script></body></html>"""
    return HTMLResponse(html)
