# apps/api/main.py
import os, pathlib, base64, time
from typing import Optional, Dict, Any
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ---- load .env ----
BASE = pathlib.Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=BASE / ".env")

# ---- internal modules ----
from brain.orchestrator import answer_question, memory_add
from tools.rag import RAG
from tools.crawler import crawl_domain
from tools.filegen import make_pdf, make_docx, make_csv, make_pptx, make_zip, make_txt
from brain.providers import ensemble_chat, openai_image_b64

WEB_DIR = BASE / "apps" / "web-min"
OUT_DIR = BASE / "out"
DATA_DIR = BASE / "data"
OUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

rag = RAG(data_dir=DATA_DIR, store_dir=BASE / "storage")

app = FastAPI(title="DevForge Ultra API", version="2.4.0")

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # keep wide open for dev; restrict for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- static mounts ----------
if WEB_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(WEB_DIR), html=True), name="ui")
app.mount("/out", StaticFiles(directory=str(OUT_DIR), html=False), name="out")

# ---------- models ----------
class AskPayload(BaseModel):
    message: str

class VisionPayload(BaseModel):
    message: str
    image_data_url: str

class ImageGenPayload(BaseModel):
    prompt: str
    size: Optional[str] = "1024x1024"

class ImaginePayload(BaseModel):
    prompt: str
    n_seeds: int = 3  # number of creative alternates

class VideoStartPayload(BaseModel):
    prompt: str
    aspect_ratio: Optional[str] = "1280:720"
    duration: Optional[int] = 5
    image_url: Optional[str] = None

class CrawlPayload(BaseModel):
    domain: str
    max_pages: int = 10

class FilePayload(BaseModel):
    name: str
    content: str

class ZipPayload(BaseModel):
    name: str
    files: Dict[str, str] | None = None   # {"path.txt":"content"} optional

class CodeTestPayload(BaseModel):
    language: str = "python"
    code: str

# ---------- health / diag ----------
@app.get("/api/health")
def health():
    return {"ok": True}

@app.get("/api/diag")
def diag():
    return {
        "providers": {
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "google": bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
            "cohere": bool(os.getenv("COHERE_API_KEY")),
            "mistral": bool(os.getenv("MISTRAL_API_KEY")),
            "runway": bool(os.getenv("RUNWAY_API_KEY")),
        },
        "model_pref": os.getenv("MODEL_PREF", "auto")
    }

# ---------- Q&A: tools -> RAG -> ensemble ----------
@app.post("/api/ask")
def api_ask(p: AskPayload):
    q = (p.message or "").strip()
    if not q:
        return {"answer": "Please type a question.", "provider": "none"}

    local = answer_question(q)
    if local and isinstance(local, dict) and local.get("answer"):
        memory_add(q, local["answer"])
        return {"answer": local["answer"], "provider": local.get("provider", "local")}

    ctx_snips = rag.query(q, k=4)
    ctx_text = "\n\n".join([s["text"] for s in ctx_snips]) if ctx_snips else ""

    system = "Answer directly and correctly. Be concise and helpful. If asked for code, provide runnable code blocks."
    prompt = f"Question: {q}\n\nContext (optional):\n{ctx_text}\n\nAnswer:"
    ans = (ensemble_chat(prompt, system=system) or "").strip() or "I don't have an exact answer yet for that."
    memory_add(q, ans)
    return {"answer": ans, "provider": "ensemble"}

# ---------- Vision (OpenAI chat/vision via base64) ----------
@app.post("/api/vision")
def api_vision(p: VisionPayload):
    prefix = p.image_data_url.split(",")[0]
    if not prefix.startswith("data:image"):
        return {"answer": "Please attach an image."}
    img_b64 = p.image_data_url.split(",", 1)[1]
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return {"answer": "Vision requires OPENAI_API_KEY."}

    import requests
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": os.getenv("VISION_MODEL", "gpt-4o-mini"),
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": p.message or "Describe this image."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ]
        }]
    }
    r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body, timeout=90)
    try:
        r.raise_for_status()
        return {"answer": r.json()["choices"][0]["message"]["content"]}
    except Exception as e:
        return {"answer": f"Vision error: {e} | {r.text[:180]}"}

# ---------- Image generation (OpenAI) ----------
@app.post("/api/image/generate")
def api_image_gen(p: ImageGenPayload):
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(400, "Set OPENAI_API_KEY in your .env to enable image generation.")
    try:
        b64 = openai_image_b64(p.prompt, size=p.size or "1024x1024")
    except Exception as e:
        # show underlying OpenAI error (e.g. billing) for fast debugging
        raise HTTPException(500, f"OpenAI image error: {e}")

    img_path = OUT_DIR / f"image_{int(time.time())}.png"
    with open(img_path, "wb") as f:
        f.write(base64.b64decode(b64))
    return {"b64": b64, "url": f"/out/{img_path.name}"}

# ---------- Imagine (creative: text fusion OR tiny game ZIP) ----------
def _game_files(prompt: str) -> Dict[str, str]:
    title = "Neon Runner"
    idx = f"""<!doctype html><html><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{title}</title>
<link rel="stylesheet" href="style.css"/></head>
<body><canvas id="game"></canvas><script src="game.js"></script></body></html>"""

    css = """html,body{margin:0;height:100%;background:#0b0f19}
#game{display:block;margin:0 auto;background:linear-gradient(#0f172a,#0b1324)}"""

    js = r"""const C=document.getElementById('game'),X=C.getContext('2d');let W,H,t=0,score=0,alive=true;
function R(){W=C.width=innerWidth;H=C.height=innerHeight}addEventListener('resize',R);R();
let g=H*.8,p={x:120,y:0,vy:0,s:28,on:false},coins=[],spikes=[];
function spawn(){if(Math.random()<.06)coins.push({x:W+20,y:g-60-60*Math.random(),r:8,got:false});
if(Math.random()<.04)spikes.push({x:W+20,y:g-16,w:22,h:18})}
function input(d){p.on=d;if(d&&p.y>=g-p.s-1)p.vy=-15}
addEventListener('pointerdown',e=>input(true));addEventListener('pointerup',e=>input(false));
addEventListener('keydown',e=>{if(e.code==='Space'||e.code==='ArrowUp')input(true)});addEventListener('keyup',e=>{if(e.code==='Space'||e.code==='ArrowUp')input(false)});
function reset(){score=0;coins=[];spikes=[];t=0;alive=true;p.y=g-p.s;p.vy=0}reset();
function HUD(){X.fillStyle='#bde0fe';X.font='bold 18px system-ui';X.fillText('Score: '+score,20,30);
if(!alive){X.textAlign='center';X.font='bold 36px system-ui';X.fillText('Game Over — click to retry',W/2,H/2);X.textAlign='left'}}
function loop(){requestAnimationFrame(loop);X.clearRect(0,0,W,H);
if(alive){t++;if(t%6===0)spawn();p.vy+=.8;p.y+=p.vy;if(p.y>g-p.s){p.y=g-p.s;p.vy=0}
X.fillStyle='#ff8fa3';for(let s of spikes){s.x-=6;X.beginPath();X.moveTo(s.x,s.y+s.h);X.lineTo(s.x+s.w/2,s.y);X.lineTo(s.x+s.w,s.y+s.h);X.closePath();X.fill();
if(Math.abs((s.x+s.w/2)-p.x)<(s.w/2+p.s*.6)&&Math.abs((s.y+5)-(p.y+p.s/2))<20){alive=false}}spikes=spikes.filter(s=>s.x>-60);
for(let c of coins){c.x-=5;X.fillStyle=c.got?'#ccc':'#ffd166';X.beginPath();X.arc(c.x,c.y,c.r,0,Math.PI*2);X.fill();
if(!c.got&&Math.hypot(c.x-p.x,c.y-(p.y+p.s/2))<c.r+p.s*.6){c.got=true;score+=10}}coins=coins.filter(c=>c.x>-40);
X.strokeStyle='#2a2a54';X.lineWidth=2;X.beginPath();X.moveTo(0,g);X.lineTo(W,g);X.stroke();X.fillStyle='#7cffc7';X.fillRect(p.x-p.s/2,p.y,p.s,p.s)}
else{if(p.on)reset()}HUD()}loop();"""
    return {
        "index.html": idx,
        "style.css": css,
        "game.js": js,
        "readme.txt": f"{title}\n\nPrompt seed: {prompt}\nRun: open index.html",
    }

@app.post("/api/imagine")
def api_imagine(p: ImaginePayload):
    q = (p.prompt or "").strip()
    if not q:
        raise HTTPException(400, "prompt required")

    # special: build a tiny game
    if "game" in q.lower():
        files_map = _game_files(q)
        fname = f"game_{int(time.time())}.zip"
        out = make_zip(fname, files_map, OUT_DIR)
        final = "I built a tiny HTML5 game (endless runner). Unzip and open **index.html**."
        return {"final": final, "zip_url": f"/out/{out.name}", "alternates": []}

    # otherwise: creative text fusion with alternates
    system = "Be imaginative, concrete, and helpful. Present crisp, original ideas."
    seeds = []
    for _ in range(max(1, min(8, p.n_seeds))):
        seeds.append(ensemble_chat(q, system=system) or "")
    seeds = [s for s in seeds if s.strip()]
    if not seeds:
        seeds = ["A speculative concept could combine new materials, responsive interfaces, and adaptive behavior."]
    final = seeds[0]
    alternates = seeds[1:]
    return {"final": final, "alternates": alternates}

# ---------- Video (Runway public API is api.dev.runwayml.com) ----------
@app.post("/api/video/runway/start")
def api_video_start(p: VideoStartPayload):
    if not os.getenv("RUNWAY_API_KEY"):
        raise HTTPException(400, "Set RUNWAY_API_KEY in your .env.")
    import requests
    hdr = {"Authorization": f"Bearer {os.getenv('RUNWAY_API_KEY')}", "Content-Type": "application/json"}
    body = {"prompt": p.prompt, "aspect_ratio": p.aspect_ratio, "duration": p.duration}
    if p.image_url:
        body["image_url"] = p.image_url
    r = requests.post("https://api.dev.runwayml.com/v1/tasks", headers=hdr, json=body, timeout=60)
    try:
        r.raise_for_status()
        j = r.json()
        return {"id": j.get("id") or j.get("task_id"), "status": j.get("status", "QUEUED")}
    except Exception as e:
        raise HTTPException(500, f"Runway start error: {e} | {r.text[:200]}")

@app.post("/api/video/runway/poll")
def api_video_poll(payload: Dict[str, Any]):
    if not os.getenv("RUNWAY_API_KEY"):
        raise HTTPException(400, "Set RUNWAY_API_KEY in your .env.")
    task_id = (payload or {}).get("task_id")
    if not task_id:
        raise HTTPException(400, "task_id required")
    import requests
    hdr = {"Authorization": f"Bearer {os.getenv('RUNWAY_API_KEY')}"}
    r = requests.get(f"https://api.dev.runwayml.com/v1/tasks/{task_id}", headers=hdr, timeout=60)
    try:
        r.raise_for_status()
        j = r.json()
        url = j.get("output") or j.get("result") or (j.get("assets") or {}).get("video") or ""
        return {"id": task_id, "status": j.get("status", "UNKNOWN"), "output": url}
    except Exception as e:
        raise HTTPException(500, f"Runway poll error: {e} | {r.text[:200]}")

# ---------- Upload → RAG ----------
@app.post("/api/upload")
def upload(file: UploadFile = File(...)):
    path = DATA_DIR / file.filename
    with open(path, "wb") as f:
        f.write(file.file.read())
    rag.reindex()
    return {"stored": str(path)}

# ---------- Crawl → RAG ----------
@app.post("/api/crawl")
def api_crawl(p: CrawlPayload):
    out = crawl_domain(p.domain, max_pages=p.max_pages, out_dir=DATA_DIR)
    rag.reindex()
    return {"ok": True, "saved": out}

# ---------- File creators (TXT/PDF/DOCX/CSV/PPTX/ZIP) ----------
@app.post("/api/file/txt")
def api_txt(p: FilePayload):
    fp = make_txt(p.name, p.content, OUT_DIR)
    return {"url": f"/out/{fp.name}"}

@app.post("/api/file/pdf")
def api_pdf(p: FilePayload):
    fp = make_pdf(p.name, p.content, OUT_DIR)
    return {"url": f"/out/{fp.name}"}

@app.post("/api/file/docx")
def api_docx(p: FilePayload):
    fp = make_docx(p.name, p.content, OUT_DIR)
    return {"url": f"/out/{fp.name}"}

@app.post("/api/file/csv")
def api_csv(p: FilePayload):
    fp = make_csv(p.name, p.content, OUT_DIR)
    return {"url": f"/out/{fp.name}"}

@app.post("/api/file/pptx")
def api_pptx(p: FilePayload):
    fp = make_pptx(p.name, p.content, OUT_DIR)
    return {"url": f"/out/{fp.name}"}

@app.post("/api/file/zip")
def api_zip(p: ZipPayload):
    files_map = p.files or {"readme.txt": "Generated by DevForge Ultra"}
    fp = make_zip(p.name, files_map, OUT_DIR)
    return {"url": f"/out/{fp.name}"}

# ---------- root ----------
@app.get("/")
def root():
    return {"message": "OK — open /ui for chat, or /api/health"}
