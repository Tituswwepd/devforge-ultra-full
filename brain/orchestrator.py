# brain/orchestrator.py
"""
Super-charged orchestrator:
- Fast math path (SymPy safe)
- Tiny facts + short-term memory (last 50 Q/A)
- Multi-provider fan-out with voting (OpenAI, Anthropic, Google/Gemini, Mistral, Cohere)
- Optional provider forcing via env or function arg
- Deliberate micro-pause (feels thoughtful, bounded by DELIBERATE_MS)
- Consistent, concise, runnable-code-friendly output

Public API (stable):
    answer_question(question: str, force_provider: Optional[str] = None, context: str = "") -> Dict[str, Any]
        Returns { "mode": "single", "answer": str, "provider": str, ... }
"""

import os
import re
import json
import time
from typing import List, Tuple, Optional, Dict, Any
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from sympy import sympify
from sympy.core.sympify import SympifyError

# ================== Config (env with sane defaults) ==================
DELIBERATE_MS = int(os.getenv("DELIBERATE_MS", "250"))      # small bounded delay (0..1500ms)
TIMEOUT_S     = float(os.getenv("PROVIDER_TIMEOUT", "22"))  # network timeout per call

OPENAI_MODEL   = os.getenv("OPENAI_CHAT_MODEL",   "gpt-4o")
ANTH_MODEL     = os.getenv("ANTHROPIC_MODEL",     "claude-3-5-sonnet-latest")
GOOGLE_MODEL   = os.getenv("GOOGLE_MODEL",        "gemini-1.5-flash")
MISTRAL_MODEL  = os.getenv("MISTRAL_MODEL",       "mistral-large-latest")
COHERE_MODEL   = os.getenv("COHERE_MODEL",        "command-r-plus")

OPENAI_TEMP    = float(os.getenv("OPENAI_TEMP",    "0.2"))
ANTH_TEMP      = float(os.getenv("ANTHROPIC_TEMP", "0.2"))
GOOGLE_TEMP    = float(os.getenv("GOOGLE_TEMP",    "0.2"))
MISTRAL_TEMP   = float(os.getenv("MISTRAL_TEMP",   "0.2"))
COHERE_TEMP    = float(os.getenv("COHERE_TEMP",    "0.2"))

OPENAI_MAXTOK  = int(os.getenv("OPENAI_MAX_TOKENS",    "700"))
ANTH_MAXTOK    = int(os.getenv("ANTHROPIC_MAX_TOKENS", "900"))
GOOGLE_MAXTOK  = int(os.getenv("GOOGLE_MAX_TOKENS",    "900"))
MISTRAL_MAXTOK = int(os.getenv("MISTRAL_MAX_TOKENS",   "700"))
COHERE_MAXTOK  = int(os.getenv("COHERE_MAX_TOKENS",    "800"))

# ================== Lightweight memory (last 50 turns) ==================
MEMORY: deque[Tuple[str, str]] = deque(maxlen=50)

def memory_add(q: str, a: str):
    if q and a:
        MEMORY.append((q, a))

def memory_tail(n: int = 5) -> List[Tuple[str, str]]:
    return list(MEMORY)[-n:]

# ================== Tiny internal facts ==================
FACTS = {
    "capital city of kenya": "Nairobi",
    "capital city of tanzania": "Dodoma",
    "president of china": "Xi Jinping",
    "mount kenya continent": "Africa",
    "largest mountain in the world": "Mount Everest",
}
# A few helper patterns (kept tiny to avoid false certainty)
_CAPITAL_OF = {
    "kenya": "Nairobi",
    "tanzania": "Dodoma",
    "south africa": "Pretoria (administrative), Cape Town (legislative), Bloemfontein (judicial)",
}

# ================== Math safe-path ==================
_ALLOWED_MATH = set("0123456789+-*/()% .")

def _math_try(expr: str) -> str:
    try:
        if expr and all(c in _ALLOWED_MATH for c in expr):
            # must contain at least one operator to avoid "numbers only"
            if any(op in expr for op in "+-*/%"):
                val = sympify(expr).evalf()
                s = str(val)
                if s.endswith(".0"): s = s[:-2]
                return s
    except (SympifyError, Exception):
        pass
    return ""

# ================== Normalization helpers ==================
def _strip_answer(text: str) -> str:
    if not text:
        return ""
    t = text.strip()
    t = re.sub(r"^\s*(answer\s*:\s*)", "", t, flags=re.I)
    return t.strip()

def _normalize_for_vote(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"`+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _vote_select(candidates: List[str]) -> str:
    """Simple majority vote on normalized strings; tie-break by length."""
    if not candidates:
        return ""
    counts: Dict[str, Tuple[int, str]] = {}
    for raw in candidates:
        n = _normalize_for_vote(raw)
        if not n:
            continue
        if n not in counts:
            counts[n] = (0, raw)
        counts[n] = (counts[n][0] + 1, counts[n][1])
    if not counts:
        return candidates[0]
    ranked = sorted(counts.items(), key=lambda kv: (kv[1][0], len(kv[1][1])), reverse=True)
    return ranked[0][1][1]

# ================== Provider wrappers ==================
def _openai_chat(prompt: str, system: str) -> Optional[str]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    url = "https://api.openai.com/v1/chat/completions"
    body = {
        "model": OPENAI_MODEL,
        "messages": [{"role":"system","content":system},{"role":"user","content":prompt}],
        "temperature": OPENAI_TEMP,
        "max_tokens": OPENAI_MAXTOK,
    }
    r = requests.post(url, headers={"Authorization": f"Bearer {key}","Content-Type":"application/json"}, json=body, timeout=TIMEOUT_S)
    if r.status_code >= 400: return None
    try:
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return None

def _anthropic_chat(prompt: str, system: str) -> Optional[str]:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    url = "https://api.anthropic.com/v1/messages"
    body = {
        "model": ANTH_MODEL,
        "max_tokens": ANTH_MAXTOK,
        "temperature": ANTH_TEMP,
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    r = requests.post(url, headers={"x-api-key": key, "anthropic-version":"2023-06-01", "Content-Type":"application/json"}, json=body, timeout=TIMEOUT_S)
    if r.status_code >= 400: return None
    try:
        parts = r.json().get("content", [])
        return "".join(p.get("text","") for p in parts if p.get("type") == "text") or None
    except Exception:
        return None

def _google_chat(prompt: str, system: str) -> Optional[str]:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    # Gemini text-only REST (v1beta also works; using v1 for stability)
    url = f"https://generativelanguage.googleapis.com/v1/models/{GOOGLE_MODEL}:generateContent?key={key}"
    body = {
        "contents": [{"role":"user","parts":[{"text": f"{system}\n\n{prompt}"}]}],
        "generationConfig": {"temperature": GOOGLE_TEMP, "maxOutputTokens": GOOGLE_MAXTOK},
    }
    r = requests.post(url, headers={"Content-Type":"application/json"}, json=body, timeout=TIMEOUT_S)
    if r.status_code >= 400: return None
    try:
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return None

def _mistral_chat(prompt: str, system: str) -> Optional[str]:
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        return None
    url = "https://api.mistral.ai/v1/chat/completions"
    body = {
        "model": MISTRAL_MODEL,
        "messages": [{"role":"system","content":system},{"role":"user","content":prompt}],
        "temperature": MISTRAL_TEMP,
        "max_tokens": MISTRAL_MAXTOK,
    }
    r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type":"application/json"}, json=body, timeout=TIMEOUT_S)
    if r.status_code >= 400: return None
    try:
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return None

def _cohere_chat(prompt: str, system: str) -> Optional[str]:
    key = os.getenv("COHERE_API_KEY")
    if not key:
        return None
    url = "https://api.cohere.com/v1/chat"
    # Cohere accepts "message" + "preamble" for system
    body = {"model": COHERE_MODEL, "message": prompt, "preamble": system, "temperature": COHERE_TEMP, "max_tokens": COHERE_MAXTOK}
    r = requests.post(url, headers={"Authorization": f"Bearer {key}", "Content-Type":"application/json"}, json=body, timeout=TIMEOUT_S)
    if r.status_code >= 400: return None
    try:
        j = r.json()
        if "text" in j:
            return j["text"]
        msg = j.get("message", {})
        if isinstance(msg, dict):
            parts = msg.get("content", [])
            return "".join(p.get("text","") for p in parts if p.get("type") == "text") or None
        return None
    except Exception:
        return None

_PROVIDER_FUNCS: Dict[str, Any] = {
    "openai":   _openai_chat,
    "anthropic":_anthropic_chat,
    "google":   _google_chat,
    "mistral":  _mistral_chat,
    "cohere":   _cohere_chat,
}

def _provider_order(force_provider: Optional[str] = None) -> List[str]:
    if force_provider and force_provider in _PROVIDER_FUNCS:
        return [force_provider]
    order_env = os.getenv("PROVIDER_ORDER", "")
    if order_env:
        order = [p.strip() for p in order_env.split(",") if p.strip() in _PROVIDER_FUNCS]
        if order:
            return order
    # Default preference (fast + strong; you can tweak via PROVIDER_ORDER=...)
    return ["openai", "anthropic", "google", "mistral", "cohere"]

def _active_providers(order: List[str]) -> List[str]:
    """Return providers that have keys configured."""
    act: List[str] = []
    if "openai" in order and os.getenv("OPENAI_API_KEY"):       act.append("openai")
    if "anthropic" in order and os.getenv("ANTHROPIC_API_KEY"): act.append("anthropic")
    if "google" in order and (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")): act.append("google")
    if "mistral" in order and os.getenv("MISTRAL_API_KEY"):     act.append("mistral")
    if "cohere" in order and os.getenv("COHERE_API_KEY"):       act.append("cohere")
    return act

# ================== Public entrypoint ==================
def answer_question(question: str, force_provider: Optional[str] = None, context: str = "") -> Dict[str, Any]:
    """
    Normalized dict:
        { "mode": "single", "answer": str, "provider": str, "providers_used": [...], "from_memory": bool }

    Pipeline:
      1) Safe math
      2) Tiny facts + capital lookup
      3) Memory (exact)
      4) Multi-provider fan-out + vote (or single provider if forced)
    """
    q = (question or "").strip()
    ql = q.lower()

    # (A) tiny deliberate pause for "thoughtful" feel (bounded)
    if DELIBERATE_MS > 0:
        time.sleep(min(DELIBERATE_MS, 1500) / 1000.0)

    # (1) math
    m = _math_try(ql)
    if m:
        memory_add(q, m)
        return {"mode":"single","answer":m,"provider":"local-math","providers_used":[], "from_memory": False}

    # (2) facts & light pattern ("capital of X")
    if "capital" in ql and " of " in ql:
        m_cap = re.search(r"capital\s+(city\s+)?of\s+([a-z\s\-]+)\??", ql)
        if m_cap:
            country = m_cap.group(2).strip()
            ans_cap = _CAPITAL_OF.get(country)
            if ans_cap:
                memory_add(q, ans_cap)
                return {"mode":"single","answer":ans_cap,"provider":"local-facts","providers_used":[], "from_memory": False}
    for k, v in FACTS.items():
        if k in ql:
            memory_add(q, v)
            return {"mode":"single","answer":v,"provider":"local-facts","providers_used":[], "from_memory": False}

    # (3) memory exact
    for (qq, aa) in reversed(MEMORY):
        if qq.lower() == ql and aa:
            return {"mode":"single","answer":aa,"provider":"local-memory","providers_used":[], "from_memory": True}

    # (4) model(s)
    system = "Answer directly and correctly. Be concise. If asked for code, provide runnable code blocks."
    prompt = f"{q}\n\nContext:\n{context}" if context else q

    order = _provider_order(force_provider)
    active = _active_providers(order)

    # If a provider is forced, do a single call (deterministic path).
    if force_provider and force_provider in active:
        fn = _PROVIDER_FUNCS[force_provider]
        try:
            text = fn(prompt, system)
        except Exception:
            text = None
        if text:
            ans = _strip_answer(text)
            memory_add(q, ans)
            return {"mode":"single","answer":ans,"provider":force_provider,"providers_used":[force_provider], "from_memory": False}
        # fall through to ensemble if the forced one failed

    # Ensemble fan-out to all active providers in parallel, then vote
    answers: List[str] = []
    used: List[str] = []
    with ThreadPoolExecutor(max_workers=len(active) or 1) as ex:
        futures = {ex.submit(_PROVIDER_FUNCS[p], prompt, system): p for p in active}
        for fut in as_completed(futures, timeout=TIMEOUT_S + 5):
            prov = futures[fut]
            try:
                resp = fut.result()
                if resp:
                    answers.append(_strip_answer(resp))
                    used.append(prov)
            except Exception:
                # ignore provider error; continue collecting others
                pass

    if answers:
        final = _vote_select(answers)
        memory_add(q, final)
        provider_label = "ensemble" if len(used) > 1 else (used[0] if used else "unknown")
        return {"mode":"single","answer":final,"provider":provider_label,"providers_used":used, "from_memory": False}

    # (5) last resort
    return {
        "mode": "single",
        "answer": "I don't have an exact answer yet for that.",
        "provider": "none",
        "providers_used": [],
        "from_memory": False,
    }
