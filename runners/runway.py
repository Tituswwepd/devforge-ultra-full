import os, requests, time

RUNWAY_KEY = os.getenv("RUNWAY_API_KEY")
API = "https://api.runwayml.com/v1"  # adjust if their base changes

def _hdr():
    if not RUNWAY_KEY:
        raise RuntimeError("RUNWAY_API_KEY not set")
    return {"Authorization": f"Bearer {RUNWAY_KEY}", "Content-Type":"application/json"}

def runway_start(prompt: str, aspect_ratio: str = "1280:720", duration: int = 5, image_url: str | None = None):
    """
    Starts a text-to-video (or image+text to video) job.
    NOTE: This is a representative wrapper; adapt endpoint/params to your Runway plan/model.
    """
    # Example payload; adjust to real Runway endpoint/model you use:
    payload = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "duration": duration
    }
    if image_url:
        payload["image_url"] = image_url

    r = requests.post(f"{API}/tasks", headers=_hdr(), json=payload, timeout=60)
    if not r.ok:
        raise RuntimeError(f"Runway start error: {r.status_code} {r.text[:200]}")
    return r.json()

def runway_poll(task_id: str):
    r = requests.get(f"{API}/tasks/{task_id}", headers=_hdr(), timeout=60)
    if not r.ok:
        raise RuntimeError(f"Runway poll error: {r.status_code} {r.text[:200]}")
    return r.json()
