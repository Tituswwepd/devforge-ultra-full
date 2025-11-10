import os, base64, requests

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_HOST = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
IMG_MODEL = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
CHAT_MODEL = os.getenv("OPENAI_VISION_MODEL", os.getenv("MODEL_PREF","gpt-4o-mini"))

def _headers():
    if not OPENAI_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return {"Authorization": f"Bearer {OPENAI_KEY}"}

def openai_image_generate(prompt: str, size: str = "1024x1024") -> str:
    """
    Returns base64 PNG string.
    """
    url = f"{OPENAI_HOST}/v1/images/generations"
    body = {"model": IMG_MODEL, "prompt": prompt, "size": size}
    r = requests.post(url, json=body, headers={**_headers(), "Content-Type": "application/json"}, timeout=120)
    if not r.ok:
        raise RuntimeError(f"OpenAI image error: {r.status_code} {r.text[:200]}")
    data = r.json()
    b64 = data["data"][0]["b64_json"]
    return b64

def openai_vision_chat(message: str, image_data_url: str) -> str:
    """
    message (text) + image (data URL) -> text answer
    """
    url = f"{OPENAI_HOST}/v1/chat/completions"
    content = [
        {"type":"text","text": message or "Describe the image."},
        {"type":"image_url","image_url":{"url": image_data_url}}
    ]
    body = {
        "model": CHAT_MODEL,
        "messages": [{"role":"user","content": content}],
        "temperature": 0.2
    }
    r = requests.post(url, json=body, headers={**_headers(), "Content-Type": "application/json"}, timeout=120)
    if not r.ok:
        raise RuntimeError(f"OpenAI vision error: {r.status_code} {r.text[:200]}")
    return r.json()["choices"][0]["message"]["content"]
