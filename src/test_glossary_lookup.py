"""Tests for the retrieval contract. Run: python src/test_glossary_lookup.py

These guard the two behaviours the architecture depends on:
  1. a hit returns the canonical topic, not the surface form the source used
  2. a miss is reported as a miss, never as a guess
"""
import sys, csv, tempfile, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from glossary_lookup import GlossaryLookup, format_for_prompt, _normalize

def _fixtures():
    d = tempfile.mkdtemp()
    t = Path(d, "topics.csv"); s = Path(d, "syn.csv")
    with open(t, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["title", "summary", "url"])
        w.writerow(["A1C", "A blood test for diabetes.", "https://medlineplus.gov/a1c.html"])
        w.writerow(["Acne", "A skin condition with pimples.", "https://medlineplus.gov/acne.html"])
    with open(s, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["surface", "canonical", "kind", "canonical_url"])
        for row in [("A1C","A1C","title",""), ("HbA1C","A1C","also_called",""),
                    ("Hemoglobin A1c","A1C","see_reference",""),
                    ("Acne","Acne","title",""), ("Pimples","Acne","see_reference","")]:
            w.writerow(row)
    return str(t), str(s)

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond: passed += 1; print(f"  PASS  {name}")
    else:    failed += 1; print(f"  FAIL  {name}")

t, s = _fixtures()
g = GlossaryLookup(t, s)

print("exact lookup")
check("synonym resolves to canonical topic", g.lookup("HbA1C").canonical_topic == "A1C")
check("lay term resolves", g.lookup("Pimples").canonical_topic == "Acne")
check("match_type is exact_synonym", g.lookup("HbA1C").match_type == "exact_synonym")
check("score is None for exact", g.lookup("HbA1C").score is None)
check("definition comes back", "blood test" in (g.lookup("HbA1C").definition or ""))
check("url comes back", (g.lookup("HbA1C").source_url or "").startswith("https://"))

print("misses")
m = g.lookup("metformin")
check("miss reports found=False", m.found is False)
check("miss has no definition", m.definition is None)
check("miss match_type is none", m.match_type == "none")
check("empty string is a miss", g.lookup("").found is False)
check("None is a miss", g.lookup(None).found is False)

print("normalization")
check("case insensitive", g.lookup("hba1c").found)
check("whitespace tolerated", g.lookup("  HbA1C  ").found)
check("trailing period tolerated", g.lookup("Pimples.").found)
check("plurals NOT silently merged", _normalize("statins") != _normalize("statin"))

print("no vector index wired in")
check("degrades to exact-only without crashing", g.lookup("something unknown").found is False)

print("prompt formatting")
out = format_for_prompt(g.lookup_atoms([
    {"text": "HbA1C", "type": "lab_value"},
    {"text": "metformin", "type": "medication"},
]))
check("found terms listed", "A1C" in out)
check("missing terms listed explicitly", "metformin" in out)
check("instructs not to invent", "Do NOT invent" in out)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
