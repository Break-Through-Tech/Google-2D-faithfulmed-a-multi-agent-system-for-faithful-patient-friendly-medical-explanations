"""Glossary lookup: the retrieval contract from docs/architecture.md, as working code.

Why this file exists
--------------------
The Simplifier needs plain-language definitions. The architecture says retrieval happens
in exactly one place and returns one shape. This module is that shape, so the Simplifier
owner and the index owners build against the same thing instead of two guesses.

Two-step lookup, cheap step first:
  1. exact match against the 2,868 alternate names (a dict, no API call)
  2. vector search in ChromaDB, only if step 1 misses

Step 1 matters for cost: the free Gemini tier allows about 20 requests/day/key, so every
embedding call we avoid is one we can spend elsewhere.

The vector half is optional. Without ChromaDB wired in, exact lookup still works, which
means the Simplifier can be built and tested before the index is finished.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class LookupResult:
    """One lookup. Shape is fixed by docs/architecture.md."""
    term: str
    found: bool
    canonical_topic: str | None = None
    definition: str | None = None
    source_url: str | None = None
    match_type: str = "none"          # exact_synonym | vector | none
    score: float | None = None        # similarity for vector, None for exact

    def as_dict(self) -> dict:
        return asdict(self)


def _normalize(term: str) -> str:
    """Lowercase, collapse whitespace, drop surrounding punctuation.

    Deliberately conservative: we do NOT stem or strip plurals. "A1C" and "a1c" should
    match; "statins" and "statin" should not be silently merged, because a wrong match on
    a drug name is worse than no match at all.
    """
    t = (term or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t.strip(" .,;:!?()[]\"'")


class GlossaryLookup:
    """Exact synonym lookup, with optional vector fallback.

    Parameters
    ----------
    topics_csv, synonyms_csv
        Produced by notebooks/exploratory-data-analysis/medlineplus_glossary_eda.ipynb.
    collection
        Optional ChromaDB collection. If None, only exact lookup is available.
    embed_fn
        Optional callable(str) -> list[float]. Required for vector search.
    min_score
        Vector matches below this are reported as not found. A weak nearest neighbour is
        worse than an honest miss, because the Simplifier would define a term wrongly.
    """

    def __init__(self, topics_csv: str | Path, synonyms_csv: str | Path,
                 collection=None, embed_fn=None, min_score: float = 0.60):
        self.collection = collection
        self.embed_fn = embed_fn
        self.min_score = min_score

        self._definitions: dict[str, dict] = {}
        with open(topics_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self._definitions[row["title"]] = {
                    "definition": row.get("summary", ""),
                    "url": row.get("url", ""),
                }

        self._synonyms: dict[str, str] = {}
        with open(synonyms_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self._synonyms.setdefault(_normalize(row["surface"]), row["canonical"])

    def __len__(self) -> int:
        return len(self._synonyms)

    def lookup(self, term: str) -> LookupResult:
        key = _normalize(term)

        # 1. exact synonym match, no API call
        canonical = self._synonyms.get(key)
        if canonical and canonical in self._definitions:
            d = self._definitions[canonical]
            return LookupResult(term=term, found=True, canonical_topic=canonical,
                                definition=d["definition"], source_url=d["url"],
                                match_type="exact_synonym", score=None)

        # 2. vector fallback, only if the index is wired in
        if self.collection is not None and self.embed_fn is not None:
            try:
                res = self.collection.query(query_embeddings=[self.embed_fn(term)],
                                            n_results=1)
                metas = (res.get("metadatas") or [[]])[0]
                docs = (res.get("documents") or [[]])[0]
                dists = (res.get("distances") or [[]])[0]
                if metas and docs:
                    score = 1.0 - float(dists[0]) if dists else None
                    if score is None or score >= self.min_score:
                        topic = metas[0].get("title")
                        return LookupResult(term=term, found=True, canonical_topic=topic,
                                            definition=docs[0],
                                            source_url=metas[0].get("url"),
                                            match_type="vector", score=score)
            except Exception:
                # A retrieval failure must degrade to "not found", never to a guess.
                pass

        return LookupResult(term=term, found=False, match_type="none")

    def lookup_atoms(self, atoms: list[dict]) -> dict[str, LookupResult]:
        """Look up every atom the Extractor produced. Returns {atom text: result}."""
        return {a.get("text", ""): self.lookup(a.get("text", "")) for a in atoms}


def format_for_prompt(results: dict[str, LookupResult]) -> str:
    """Render lookups for the Simplifier's prompt.

    Misses are listed explicitly rather than omitted. The Simplifier has to know which
    terms have no definition so it keeps them as written instead of inventing one. A
    silently dropped miss is how a made-up drug definition reaches a patient.
    """
    found = [r for r in results.values() if r.found]
    missing = [r for r in results.values() if not r.found]

    lines = []
    if found:
        lines.append("PLAIN-LANGUAGE DEFINITIONS (use these):")
        for r in found:
            lines.append(f"- {r.canonical_topic}: {r.definition}")
    if missing:
        lines.append("")
        lines.append("NO DEFINITION FOUND for these terms. Keep them exactly as written "
                     "in the source. Do NOT invent an explanation:")
        for r in missing:
            lines.append(f"- {r.term}")
    return "\n".join(lines)
