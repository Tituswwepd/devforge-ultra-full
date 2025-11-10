# tools/rag.py
import os, json, hashlib
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

def _chunks(txt: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    i=0; n=len(txt)
    while i < n:
        yield txt[i:i+size]
        i += size - overlap

class RAG:
    def __init__(self, data_dir: Path, store_dir: Path):
        self.data_dir = Path(data_dir)
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(exist_ok=True)
        self.model = SentenceTransformer(os.getenv("EMBED_MODEL","all-MiniLM-L6-v2"))
        self.index_path = self.store_dir/"faiss.index"
        self.map_path = self.store_dir/"faiss_map.json"
        self.index = None
        self.map: List[Dict] = []
        if self.index_path.exists() and self.map_path.exists():
            self._load()
        else:
            self.reindex()

    def _load(self):
        self.index = faiss.read_index(str(self.index_path))
        self.map = json.loads(self.map_path.read_text(encoding="utf-8"))

    def reindex(self):
        docs = []
        for fp in self.data_dir.rglob("*"):
            if not fp.is_file(): continue
            if fp.suffix.lower() not in {".txt",".md",".json",".csv"}: continue
            try:
                txt = fp.read_text(encoding="utf-8", errors="ignore")
            except:
                continue
            for ch in _chunks(txt):
                docs.append({"path": str(fp), "text": ch})
        if not docs:
            self.index = faiss.IndexFlatIP(384)
            self.map = []
            faiss.write_index(self.index, str(self.index_path))
            self.map_path.write_text(json.dumps(self.map), encoding="utf-8")
            return
        embs = self.model.encode([d["text"] for d in docs], normalize_embeddings=True, convert_to_numpy=True)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)
        self.map = docs
        faiss.write_index(self.index, str(self.index_path))
        self.map_path.write_text(json.dumps(self.map), encoding="utf-8")

    def query(self, question: str, k: int = 4) -> List[Dict]:
        if self.index is None: self._load()
        q = self.model.encode([question], normalize_embeddings=True, convert_to_numpy=True)
        D, I = self.index.search(q, min(k, len(self.map)) if self.map else 1)
        out = []
        for idx in I[0]:
            if idx < 0 or idx >= len(self.map): continue
            out.append(self.map[idx])
        return out
