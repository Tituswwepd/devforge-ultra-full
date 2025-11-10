# DevForge Ultra — Full Platform
Self-hosted super‑AI: multi‑provider reasoning (OpenAI/Anthropic/Gemini/Cohere/Mistral), RAG over your files,
code scaffolding, trading tools (Pine/MT5/Deriv), connectors (Binance/MT5), voice endpoints, and a minimal chat UI.

## Quick start (Windows, PowerShell)
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
notepad .env      # paste your keys
python -m uvicorn apps.api.main:app --reload --port 5055
```
Open `apps/web-min/index.html` in your browser.
