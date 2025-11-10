# apps/api/llm_clients.py
import os
from typing import Optional

# OpenAI
try:
    from openai import OpenAI as _OpenAI
except Exception:
    _OpenAI = None

# Anthropic
try:
    import anthropic as _anthropic
except Exception:
    _anthropic = None

# Google (Gemini)
try:
    from google import genai as _genai
except Exception:
    _genai = None

# Cohere
try:
    import cohere as _cohere_mod
except Exception:
    _cohere_mod = None

# Mistral
try:
    from mistralai import Mistral as _MistralClient
except Exception:
    _MistralClient = None


def ask_openai(prompt: str, system: str = "Be concise. Output only the answer.", model: Optional[str] = None) -> Optional[str]:
    key = os.getenv("OPENAI_API_KEY")
    if not key or _OpenAI is None:
        return None
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    try:
        cli = _OpenAI(api_key=key)
        resp = cli.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "256")),
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return None


def ask_anthropic(prompt: str, system: str = "Be concise. Output only the answer.", model: Optional[str] = None) -> Optional[str]:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key or _anthropic is None:
        return None
    model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
    try:
        cli = _anthropic.Anthropic(api_key=key)
        msg = cli.messages.create(
            model=model,
            max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "256")),
            temperature=float(os.getenv("ANTHROPIC_TEMPERATURE", "0.2")),
            system=system,
            messages=[{"role":"user","content":prompt}],
        )
        # Claude returns a list of content blocks
        out = "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")
        return out.strip() if out else None
    except Exception:
        return None


def ask_google(prompt: str, system: str = "Be concise. Output only the answer.", model: Optional[str] = None) -> Optional[str]:
    key = os.getenv("GOOGLE_API_KEY")
    if not key or _genai is None:
        return None
    model = model or os.getenv("GOOGLE_MODEL", "gemini-1.5-flash")
    try:
        cli = _genai.Client(api_key=key)
        # google-genai supports system instruction in generate_text params
        resp = cli.models.generate_text(
            model=model,
            contents=[{"role":"user","parts":[{"text": prompt}]}],
            config={"temperature": float(os.getenv("GOOGLE_TEMPERATURE","0.2")), "system_instruction": system, "max_output_tokens": int(os.getenv("GOOGLE_MAX_TOKENS","256"))},
        )
        text = getattr(resp, "text", None)
        return text.strip() if text else None
    except Exception:
        return None


def ask_cohere(prompt: str, system: str = "Be concise. Output only the answer.", model: Optional[str] = None) -> Optional[str]:
    key = os.getenv("COHERE_API_KEY")
    if not key or _cohere_mod is None:
        return None
    model = model or os.getenv("COHERE_MODEL", "command-r-plus")
    try:
        co = _cohere_mod.Client(api_key=key)
        resp = co.chat(model=model, message=prompt, temperature=float(os.getenv("COHERE_TEMPERATURE","0.2")), preamble=system)
        text = getattr(resp, "text", None)
        return text.strip() if text else None
    except Exception:
        return None


def ask_mistral(prompt: str, system: str = "Be concise. Output only the answer.", model: Optional[str] = None) -> Optional[str]:
    key = os.getenv("MISTRAL_API_KEY")
    if not key or _MistralClient is None:
        return None
    model = model or os.getenv("MISTRAL_MODEL", "mistral-small-latest")
    try:
        cli = _MistralClient(api_key=key)
        resp = cli.chat.complete(
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
            temperature=float(os.getenv("MISTRAL_TEMPERATURE","0.2")),
            max_tokens=int(os.getenv("MISTRAL_MAX_TOKENS","256")),
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return None
