# brain/creative.py
import os, re, time, math, random, hashlib
from typing import List, Tuple, Dict, Optional
from collections import Counter

from .providers import (
    ensemble_chat,          # still available for fallback
    fanout_creative_sample, # new: parallel high-temp samples from all providers
)

# ---------- Helpers ----------

def _shingles(text: str, k: int = 5) -> Counter:
    t = re.sub(r"\s+", " ", (text or "")).strip().lower()
    toks = t.split()
    grams = [" ".join(toks[i:i+k]) for i in range(max(0, len(toks)-k+1))]
    return Counter(grams)

def novelty_score(text: str, corpus: List[str]) -> float:
    """Simple novelty: fewer shared shingles with other candidates = higher."""
    if not text: return 0.0
    s = _shingles(text, k=4)
    if not s: return 0.0
    overlap = 0
    total = sum(s.values())
    for other in corpus:
        if not other or other is text: 
            continue
        ss = _shingles(other, k=4)
        # sum of min counts
        common = sum(min(s[g], ss[g]) for g in s.keys() if g in ss)
        overlap += common
    # Normalize inversely to overlap; higher means more novel
    return 1.0 / (1.0 + overlap/float(max(total, 1)))

def _critique(prompt: str, draft: str) -> str:
    """
    Light critic: request concise critique & improvements from the best available model.
    Uses ensemble_chat with a strict critic instruction (low temperature).
    """
    system = (
        "You are a concise, tough critic and editor. "
        "Find any factual gaps, fuzzy claims, missing steps, or unclear parts. "
        "Propose precise improvements only; keep it under 160 words."
    )
    ask = f"Original Prompt:\n{prompt}\n\nDraft Answer:\n{draft}\n\nCritique and improved version:"
    out = ensemble_chat(ask, system=system)
    return out.strip()

def _refine(draft: str, critique: str) -> str:
    """
    Heuristic merge: if the critique contains 'Improved' section, prefer it,
    otherwise append concise corrections.
    """
    m = re.search(r"(improved(?: answer| version)?[:\n]+)(.*)$", critique, re.I | re.S)
    if m:
        improved = m.group(2).strip()
        return improved if len(improved) > 0 else draft
    # Fallback: stitch
    return f"{draft}\n\n— Improvements —\n{critique}"

# ---------- Public API ----------

def imagine(prompt: str, n_seeds: int = 5, max_time_s: float = 18.0) -> Dict:
    """
    High-creativity pipeline:
      1) Fan out across providers with higher temperature (diverse candidates)
      2) Rank with novelty + length + readability heuristic
      3) Critique & refine top candidate once
      4) Return final + shortlisted alternates
    """
    t0 = time.time()

    # 1) Diverse candidates
    seeds = max(3, min(n_seeds, 8))
    candidates = fanout_creative_sample(prompt, n=seeds, temperature=0.9, top_p=0.95, max_tokens=900)

    # If nothing from providers, fall back to ensemble (deterministic)
    if not candidates:
        base = ensemble_chat(prompt, system="Be boldly creative yet precise. Offer concrete details.")
        candidates = [base] if base else []

    # 2) Score & rank
    scored = []
    for c in candidates:
        nov = novelty_score(c, candidates)
        # crude readability/structure: more sentences and some Markdown code/headers/steps are good
        sent = max(1, len(re.split(r"[.!?]\s+", c)))
        has_struct = 1 if re.search(r"^#+\s|^[-*]\s|```|^\d+\.", c, re.M) else 0
        length_bonus = min(len(c) / 600.0, 1.0)     # enough depth but capped
        score = 0.55*nov + 0.25*length_bonus + 0.20*(sent/12.0) + 0.15*has_struct
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1] if scored else (candidates[0] if candidates else "I have an idea, but no models replied.")

    # 3) Critique & refine (once)
    if time.time() - t0 < max_time_s - 3.0:
        critique = _critique(prompt, best)
        if critique:
            best = _refine(best, critique)

    # 4) Return
    alts = [c for _, c in scored[1:4]]
    return {
        "final": best.strip(),
        "alternates": alts,
        "count": len(candidates)
    }
