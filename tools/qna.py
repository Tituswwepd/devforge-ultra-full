import os, json
from pathlib import Path
from typing import List, Dict, Any
from rapidfuzz import process, fuzz

class RagEngine:
    def __init__(self, data_dir: Path, storage_dir: Path):
        self.data_dir = data_dir
        self.index_file = storage_dir / "rag_index.json"
        self.docs = []
        self.reindex()

    def reindex(self):
        self.docs = []
        for fp in self.data_dir.glob("**/*"):
            if fp.is_file() and fp.suffix.lower() in {".txt", ".md", ".json"}:
                try:
                    self.docs.append({"path": str(fp), "text": fp.read_text(encoding='utf-8')})
                except: pass
        self.index_file.write_text(json.dumps({"count": len(self.docs)}))

    def query(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        corpus = {d["path"]: d["text"][:8000] for d in self.docs}
        matches = process.extract(question, corpus, scorer=fuzz.WRatio, limit=top_k)
        out = []
        for (path, score, _match) in matches:
            out.append({"path": path, "score": score, "snippet": corpus[path][:400]})
        return out
