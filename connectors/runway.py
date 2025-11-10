import os, time, requests

RUNWAY_API_KEY = os.getenv("RUNWAY_API_KEY")
BASE = "https://api.runwayml.com/v1"  # Runway Developer API

def _headers():
    if not RUNWAY_API_KEY:
        raise RuntimeError("RUNWAY_API_KEY not set in .env")
    return {"Authorization": f"Bearer {RUNWAY_API_KEY}", "Content-Type": "application/json"}

def image_to_video(prompt_text: str,
                   prompt_image_url: str | None = None,
                   aspect_ratio: str = "1280:720",
                   duration: int = 5,
                   model: str = "gen4_turbo"):
    """
    Starts an image->video generation task.
    Common AR: 1280:720, 720:1280, 960:960. Duration: 5 or 10 (Gen-4 Turbo).
    """
    body = {
        "model": model,
        "promptText": prompt_text,
        "duration": duration,
        "aspectRatio": aspect_ratio
    }
    if prompt_image_url:
        body["promptImage"] = prompt_image_url

    r = requests.post(f"{BASE}/image_to_video", headers=_headers(), json=body, timeout=60)
    r.raise_for_status()
    return r.json()   # returns { id, status, ... }

def get_task(task_id: str):
    r = requests.get(f"{BASE}/tasks/{task_id}", headers=_headers(), timeout=60)
    r.raise_for_status()
    return r.json()

def wait_for_result(task_id: str, timeout_s: int = 600, poll_s: float = 2.5):
    """
    Polls until status in {SUCCEEDED, FAILED, CANCELLED} or timeout.
    Returns final JSON; on success, expect a video URL in output/result/assets.
    """
    t0 = time.time()
    while True:
        j = get_task(task_id)
        st = j.get("status")
        if st in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            return j
        if time.time() - t0 > timeout_s:
            return {"status": "TIMEOUT", "id": task_id}
        time.sleep(poll_s)
