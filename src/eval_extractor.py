"""Score the Extractor against MedAESQA's expert nuggets (the Extractor gold set).

Run:
    python src/eval_extractor.py --mode sentences            # no-LLM reference, free
    python src/eval_extractor.py --mode gemini --limit 10    # 10 API calls
    python src/eval_extractor.py --mode gemini               # all 40 (two days of free quota)

Source text is each question's expert answer, the same frozen set the single-Gemini
baseline uses (data/processed/medaesqa_questions.csv), so the numbers line up.

Gemini replies are cached in results/extractor_cache.jsonl keyed by question id, model
and prompt hash, so a rerun after a scoring change costs nothing. Delete the cache, or
pass --no-cache, to re-query.

Metrics
-------
nugget_recall     share of the 498 nuggets covered by at least one atom. A nugget is
                  covered when every concept in it (the text before each [CUI] tag)
                  has >= 60% of its content words in a single atom. Deterministic,
                  no LLM judge, so it is cheap and reproducible but strict: a
                  paraphrased atom can miss.
hedge_kept        of the covered nuggets that carry a modality ([may] ...), the share
                  whose covering atom is assertion=uncertain. "may prevent" and
                  "prevents" are different claims.
atoms_per_q       mean atoms per question.
rejected_rate     share of the model's atoms the guards rejected. High means the
                  model is inventing numbers or facts.
warnings_per_q    mean guard warnings per question.

The `sentences` mode turns every source sentence into one atom with no model. It is
not a real extractor, it is a reference: recall near its value means the scorer is
working, and a real extractor should beat it on hedge_kept and atom granularity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import extractor_agent as X  # noqa: E402

DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
_CONCEPT = re.compile(r"([^,()\[\]]+?)\s*\[C\d{7}\]")
_HEDGE_WORDS = re.compile(r"\b(?:may|might|can|could|possibly|likely|suggest)", re.I)


def nugget_concepts(nugget: str) -> list[str]:
    return [c.strip() for c in _CONCEPT.findall(nugget or "") if c.strip()]


def _content(text: str) -> list[str]:
    return [X._stem(w) for w in X._words(text) if w not in X._STOP and len(w) > 2]


def concept_in(concept: str, atom_text: str, threshold: float = 0.6) -> bool:
    words = _content(concept)
    if not words:
        return False
    atom = set(_content(atom_text))
    return sum(w in atom for w in words) / len(words) >= threshold


def covering_atom(nugget: str, atoms: list[dict]) -> dict | None:
    concepts = nugget_concepts(nugget)
    if not concepts:
        return None
    for a in atoms:
        if all(concept_in(c, a["text"]) for c in concepts):
            return a
    return None


# ------------------------------------------------------------------ extractors

def sentence_atoms(source: str) -> list[dict]:
    """Reference extractor: one atom per sentence, hedge from a cue word. No LLM."""
    text = X.clean_source(source)
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 3]
    raw = [{"text": s, "type": "other",
            "assertion": "uncertain" if _HEDGE_WORDS.search(s) else "present"} for s in sents]
    return X.normalize_atoms(raw, source)[0]


def cached_generate_fn(generate_fn, cache_path: Path, use_cache: bool, model_tag: str):
    cache = {}
    if use_cache and cache_path.exists():
        for line in cache_path.read_text().splitlines():
            rec = json.loads(line)
            cache[rec["key"]] = rec["out"]

    def gen(prompt: str, qid) -> dict:
        key = f"{qid}|{model_tag}|{hashlib.sha256(prompt.encode()).hexdigest()[:12]}"
        if key in cache:
            return {**cache[key], "cached": True}
        out = generate_fn(prompt)
        if not out.get("error"):
            cache_path.parent.mkdir(exist_ok=True)
            with open(cache_path, "a") as f:
                f.write(json.dumps({"key": key, "out": out}) + "\n")
        return out
    return gen


# ------------------------------------------------------------------ scoring

def score(per_q: dict[int, dict], nuggets: pd.DataFrame) -> dict:
    covered = hedged = hedged_kept = 0
    misses = []
    for _, n in nuggets.iterrows():
        atoms = per_q.get(n.question_id, {}).get("atoms", [])
        a = covering_atom(n.nugget, atoms)
        is_hedged = isinstance(n.modality, str) and n.modality.strip()
        if a:
            covered += 1
            if is_hedged:
                hedged += 1
                hedged_kept += a["assertion"] == "uncertain"
        else:
            misses.append({"question_id": int(n.question_id), "nugget": n.nugget})
    runs = list(per_q.values())
    n_model_atoms = sum(r["n_raw"] for r in runs) or 1
    return {
        "questions": len(runs),
        "nuggets": len(nuggets),
        "nugget_recall": round(covered / max(len(nuggets), 1), 3),
        "hedge_kept": round(hedged_kept / hedged, 3) if hedged else None,
        "atoms_per_q": round(sum(len(r["atoms"]) for r in runs) / max(len(runs), 1), 1),
        "rejected_rate": round(sum(len(r["rejected"]) for r in runs) / n_model_atoms, 3),
        "warnings_per_q": round(sum(len(r["warnings"]) for r in runs) / max(len(runs), 1), 2),
        "errors": sum(1 for r in runs if r.get("error")),
        "sample_misses": misses[:15],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("sentences", "gemini"), default="sentences")
    p.add_argument("--limit", type=int, default=None, help="first N questions only")
    p.add_argument("--model", default=None, help="single model instead of the fallback chain")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--sleep", type=float, default=4.0, help="seconds between API calls")
    args = p.parse_args()

    q = pd.read_csv(DATA / "medaesqa_questions.csv")
    nug = pd.read_csv(DATA / "medaesqa_nuggets.csv")
    if args.limit:
        q = q.head(args.limit)
    nug = nug[nug.question_id.isin(q.question_id)]

    per_q, model_tag = {}, "sentences"
    if args.mode == "gemini":
        from google import genai
        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            sys.exit("Set GOOGLE_API_KEY or GEMINI_API_KEY.")
        models = (args.model,) if args.model else X.DEFAULT_MODELS
        model_tag = "+".join(models)
        gen = cached_generate_fn(X.make_generate_fn(genai.Client(api_key=key), models=models),
                                 RESULTS / "extractor_cache.jsonl", not args.no_cache, model_tag)

    for _, row in q.iterrows():
        src = row.expert_answer
        if args.mode == "sentences":
            atoms = sentence_atoms(src)
            per_q[row.question_id] = {"atoms": atoms, "rejected": [], "warnings": [],
                                      "n_raw": len(atoms)}
            continue
        out = {}
        r = X.run(src, lambda prompt: out.update(gen(prompt, row.question_id)) or out)
        n_raw = len(X.parse_atoms(r.raw) or [])
        per_q[row.question_id] = {"atoms": r.atoms, "rejected": r.rejected,
                                  "warnings": r.warnings, "n_raw": n_raw, "error": r.error,
                                  "model_used": r.model_used}
        print(f"  q{row.question_id}: {len(r.atoms)} atoms, {len(r.rejected)} rejected"
              + (f", ERROR {r.error}" if r.error else "") + (" (cached)" if out.get("cached") else ""))
        if not out.get("cached"):
            time.sleep(args.sleep)

    summary = {"mode": args.mode, "model": model_tag, **score(per_q, nug)}
    RESULTS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS / f"extractor_eval_{args.mode}_{stamp}.json"
    out_path.write_text(json.dumps({"summary": summary,
                                    "per_question": {str(k): v for k, v in per_q.items()}},
                                   indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "sample_misses"}, indent=2))
    print(f"\nwrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
