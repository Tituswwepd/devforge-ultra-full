# tools/crawler.py
import re, requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path
from typing import List

def _clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script","style","noscript"]): s.decompose()
    txt = soup.get_text("\n")
    txt = re.sub(r"\n{2,}", "\n", txt)
    return txt.strip()

def crawl_domain(domain: str, max_pages: int = 10, out_dir: Path = Path("data")) -> List[str]:
    seen, saved = set(), []
    base = f"https://{domain}" if not domain.startswith("http") else domain
    q = [base]
    out_dir = Path(out_dir); out_dir.mkdir(exist_ok=True)

    while q and len(saved) < max_pages:
        url = q.pop(0)
        if url in seen: continue
        seen.add(url)
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent":"DevForgeBot/1.0"})
            if r.status_code != 200: continue
            if "text/html" not in r.headers.get("Content-Type",""): continue
            txt = _clean_text(r.text)
            fn = out_dir / (re.sub(r"[^a-zA-Z0-9]+","_", urlparse(url).path or "index").strip("_") or "index")
            fn = fn.with_suffix(".txt")
            fn.write_text(txt, encoding="utf-8")
            saved.append(str(fn))
            # enqueue same-domain links
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                u = urljoin(url, a["href"])
                if urlparse(u).netloc == urlparse(base).netloc:
                    if u not in seen: q.append(u)
        except Exception:
            continue
    return saved
