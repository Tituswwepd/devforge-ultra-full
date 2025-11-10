from .providers import llm_complete

def _critique(q: str, a: str) -> int:
    fb = llm_complete(
        prompt=f"Question: {q}\n\nAnswer:\n{a}\n\nScore correctness 0-10. Reply only the number.",
        system="Evaluator", temperature=0.0, max_tokens=8
    )
    try:
        num = ''.join([c for c in fb if c.isdigit()])
        return int(num[:2]) if num else 0
    except:
        return 0

def vote(question: str, drafts):
    scored = [(a, _critique(question, a)) for a in drafts]
    scored.sort(key=lambda t: t[1], reverse=True)
    best = scored[0][0] if scored else (drafts[0] if drafts else "")
    return best, scored
