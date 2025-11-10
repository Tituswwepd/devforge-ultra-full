import os, base64, requests

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

def _post_json(url, headers, body):
    r = requests.post(url, headers=headers, json=body, timeout=120)
    r.raise_for_status()
    return r.json()

def vision_describe(image_data_url: str, prompt: str = "Describe the image"):
    """
    Uses OpenAI Chat Completions with image content when OPENAI_API_KEY is present.
    Fallback: returns a local message if not configured.
    """
    if not OPENAI_KEY:
        return "Vision is not configured (no OPENAI_API_KEY)."

    # OpenAI v1 chat completions with image input (multimodal)
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
    body = {
        "model": os.getenv("MODEL_PREF","gpt-4o-mini"),
        "messages": [
            {"role": "user", "content": [
                {"type":"text", "text": prompt},
                {"type":"image_url", "image_url": {"url": image_data_url}}
            ]}
        ],
        "temperature": 0.2
    }
    j = _post_json(url, headers, body)
    return j["choices"][0]["message"]["content"]

def openai_image_generate(prompt: str, size: str = "1024x1024") -> str:
    """
    Uses OpenAI Images API to generate a PNG; returns base64 PNG string.
    """
    if not OPENAI_KEY:
        raise RuntimeError("No OPENAI_API_KEY configured.")
    url = "https://api.openai.com/v1/images/generations"
    headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
    body = {"model": "gpt-image-1", "prompt": prompt, "size": size, "response_format":"b64_json"}
    j = _post_json(url, headers, body)
    return j["data"][0]["b64_json"]
