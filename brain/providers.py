# brain/providers.py
import os, base64, requests
from typing import Optional, Dict, Any, List

# ---------- small helpers ----------
def _strip(s: Optional[str]) -> str:
    return (s or "").strip()

# ---------- Chat providers (simple ensemble in priority order) ----------
def _openai_chat(prompt: str, system: str) -> Optional[str]:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(os.getenv("OPENAI_TEMP", "0.2")),
        "max_tokens": int(os.getenv("OPENAI_MAX_TOKENS", "1024")),
    }
    r = requests.post(url, headers=headers, json=body, timeout=90)
    if r.status_code >= 400:
        return None
    try:
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return None

def _anthropic_chat(prompt: str, system: str) -> Optional[str]:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": int(os.getenv("ANTHROPIC_MAX_TOKENS", "1024")),
        "temperature": float(os.getenv("ANTHROPIC_TEMP", "0.2")),
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    r = requests.post(url, headers=headers, json=body, timeout=90)
    if r.status_code >= 400:
        return None
    try:
        parts = r.json().get("content", [])
        return "".join(p.get("text", "") for p in parts if p.get("type") == "text")
    except Exception:
        return None

def _google_chat(prompt: str, system: str) -> Optional[str]:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        return None
    model = os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={key}"
    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [{"role": "user", "parts": [{"text": f"{system}\n\n{prompt}"}]}],
        "generationConfig": {
            "temperature": float(os.getenv("GOOGLE_TEMP", "0.2")),
            "maxOutputTokens": int(os.getenv("GOOGLE_MAX_TOKENS", "1024")),
        },
    }
    r = requests.post(url, headers=headers, json=body, timeout=90)
    if r.status_code >= 400:
        return None
    try:
        j = r.json()
        return j["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return None

def _mistral_chat(prompt: str, system: str) -> Optional[str]:
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        return None
    model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": float(os.getenv("MISTRAL_TEMP", "0.2")),
        "max_tokens": int(os.getenv("MISTRAL_MAX_TOKENS", "1024")),
    }
    r = requests.post(url, headers=headers, json=body, timeout=90)
    if r.status_code >= 400:
        return None
    try:
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return None

def _cohere_chat(prompt: str, system: str) -> Optional[str]:
    key = os.getenv("COHERE_API_KEY")
    if not key:
        return None
    model = os.getenv("COHERE_MODEL", "command-r-plus")
    url = "https://api.cohere.com/v1/chat"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "messages": [{"role": "user", "content": f"{system}\n\n{prompt}"}],
        "temperature": float(os.getenv("COHERE_TEMP", "0.2")),
        "max_tokens": int(os.getenv("COHERE_MAX_TOKENS", "1024")),
    }
    r = requests.post(url, headers=headers, json=body, timeout=90)
    if r.status_code >= 400:
        return None
    try:
        j = r.json()
        if "text" in j:
            return j["text"]
        msg = j.get("message", {})
        parts = msg.get("content", []) if isinstance(msg, dict) else []
        return "".join(p.get("text", "") for p in parts if p.get("type") == "text")
    except Exception:
        return None

_FUNCS: Dict[str, Any] = {
    "openai": _openai_chat,
    "anthropic": _anthropic_chat,
    "google": _google_chat,
    "mistral": _mistral_chat,
    "cohere": _cohere_chat,
}

def _order() -> List[str]:
    env = os.getenv("PROVIDER_ORDER", "openai,anthropic,google,mistral,cohere")
    return [p.strip() for p in env.split(",") if p.strip() in _FUNCS]

def ensemble_chat(prompt: str, system: str = "") -> str:
    """Try providers in configured order, return first non-empty answer."""
    for name in _order():
        fn = _FUNCS.get(name)
        try:
            out = _strip(fn(prompt, system)) if fn else ""
        except Exception:
            out = ""
        if out:
            return out
    return ""

# ---------- Images (OpenAI) ----------
def openai_image_b64(prompt: str, size: str = "1024x1024") -> str:
    """
    Robust OpenAI image generation:
      1) Try with response_format=b64_json
      2) If API rejects that parameter, retry without it
      3) If only a URL is returned, download and convert to base64
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")

    model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
    url = "https://api.openai.com/v1/images/generations"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    body = {"model": model, "prompt": prompt, "size": size, "response_format": "b64_json"}
    r = requests.post(url, headers=headers, json=body, timeout=120)

    # Some accounts don’t accept response_format — retry cleanly.
    if r.status_code >= 400 and "response_format" in r.text:
        body = {"model": model, "prompt": prompt, "size": size}
        r = requests.post(url, headers=headers, json=body, timeout=120)

    if r.status_code >= 400:
        raise RuntimeError(r.text)

    j = r.json()
    data = (j.get("data") or [{}])[0]
    b64 = data.get("b64_json")
    if b64:
        return b64

    url_ret = data.get("url")
    if url_ret:
        img = requests.get(url_ret, timeout=120)
        img.raise_for_status()
        return base64.b64encode(img.content).decode("utf-8")

    raise RuntimeError(f"OpenAI image: no content returned ({j})")
