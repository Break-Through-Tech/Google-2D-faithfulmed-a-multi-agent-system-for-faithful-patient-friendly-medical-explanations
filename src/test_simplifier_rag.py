"""Tests for the Simplifier's RAG step. Run: python src/test_simplifier_rag.py

No API key, no quota: the retriever is a stub, so these assert on what WOULD be sent.

They guard the two properties the retrieval step depends on:
  1. a lookup queries a TERM, never a whole atom sentence (docs/architecture.md)
  2. every vector lookup for one document costs ONE embedding request, not one per atom
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from simplifier import (derive_term, atom_term, load_atoms, gather_definitions,
                        _search_batch)

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond: passed += 1; print(f"  PASS  {name}")
    else:    failed += 1; print(f"  FAIL  {name}")


class StubRetriever:
    """Stands in for medlineplus_index.Retriever. Records what it was asked to embed."""
    TOPICS = {  # surface -> (topic_id, title)
        "high blood pressure": (1, "High Blood Pressure"),
        "hypertension": (1, "High Blood Pressure"),
        "asthma": (2, "Asthma"),
        "chest pain": (3, "Chest Pain"),
    }

    def __init__(self):
        self.surface_to_id = {s: str(t[0]) for s, t in self.TOPICS.items()}
        self.batches = []

    def _hit(self, tid, title, kind, score):
        return {"topic_id": tid, "title": title, "url": "u", "definition": "d",
                "score": score, "match_type": kind, "also_known_as": []}

    def _lookup(self, queries, use_synonyms):
        out, misses = [], []
        for q in queries:
            if use_synonyms and q.lower() in self.TOPICS:
                tid, title = self.TOPICS[q.lower()]
                out.append([self._hit(tid, title, "synonym", 1.0)])
            else:
                misses.append(q)
                out.append([self._hit(99, "Nearest Topic", "vector", 0.7)])
        if misses:
            self.batches.append(misses)   # one entry per embedding request
        return out

    def search(self, query, k=3, min_score=0.0, use_synonyms=True):
        return self._lookup([query], use_synonyms)[0]

    def search_many(self, queries, k=3, min_score=0.0, use_synonyms=True):
        return self._lookup(queries, use_synonyms)


class UnbatchedRetriever(StubRetriever):
    """An older/custom retriever exposing only .search - must still work."""
    search_many = None


print("derive_term: a key, not a sentence")
check("drops dose and frequency", derive_term("Metformin 500 mg twice daily") == "metformin")
check("drops report verb and severity", derive_term("ECG showed no acute ST changes") == "ecg st changes")
check("drops units and reference range", derive_term("Troponin I was 0.02 ng/mL (normal < 0.04)") == "troponin i")
check("drops the family subject", derive_term("Mother had breast cancer") == "breast cancer")
check("drops the negation", "no" not in derive_term("No shortness of breath").split())
check("empty text is an empty key", derive_term("") == "")

print("\nterm plumbing")
check("load_atoms carries the Extractor's term",
      load_atoms([{"text": "Metformin 500 mg", "term": "metformin"}])[0]["term"] == "metformin")
check("missing term defaults to empty, not absent",
      load_atoms([{"text": "Metformin 500 mg"}])[0]["term"] == "")
check("atom_term prefers the Extractor's term",
      atom_term({"text": "Metformin 500 mg twice daily", "term": "HbA1c"}) == "HbA1c")
check("atom_term falls back to the derived key",
      atom_term({"text": "Metformin 500 mg twice daily"}) == "metformin")

# spot_terms refuses a lone word inside a longer atom; an explicit term must still match.
r = StubRetriever()
atoms = load_atoms([{"id": "a1", "type": "diagnosis", "term": "asthma",
                     "text": "Patient has asthma and a cough"}])
g = gather_definitions(atoms, r, "exact")
check("explicit term matches where sentence-scanning cannot",
      [e["title"] for e in g] == ["Asthma"])
check("that match cost no embedding request", r.batches == [])

print("\nbatching")
r = StubRetriever()
atoms = load_atoms([
    {"id": "a1", "type": "diagnosis", "term": "hypertension", "text": "Hypertension"},
    {"id": "a2", "type": "lab_value", "term": "troponin", "text": "Troponin I was 0.02"},
    {"id": "a3", "type": "lab_value", "term": "ferritin", "text": "Ferritin was 12"},
    {"id": "a4", "type": "symptom", "term": "dysphagia", "text": "Dysphagia"},
])
gather_definitions(atoms, r, "exact+related")
check("three misses cost exactly one embedding request", len(r.batches) == 1)
check("the request carries all three terms", sorted(r.batches[0]) == ["dysphagia", "ferritin", "troponin"])
check("it queries terms, not atom sentences",
      not any("was 0.02" in q for q in r.batches[0]))
check("the exact hit was not embedded", "hypertension" not in r.batches[0])

r = StubRetriever()
gather_definitions(atoms, r, "exact")
check("exact mode never embeds anything", r.batches == [])

print("\nretriever without search_many still works")
r = UnbatchedRetriever()
check("stub offers no usable search_many", not callable(getattr(r, "search_many", None)))
# _search_batch must fall back to .search one query at a time
out = _search_batch(r, ["ferritin", "dysphagia"])
check("fallback returns one result list per query", len(out) == 2)
check("fallback still retrieves", out[0][0]["title"] == "Nearest Topic")
check("fallback costs one request per query, as expected", len(r.batches) == 2)
check("empty query list needs no retriever call", _search_batch(r, []) == [])

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
