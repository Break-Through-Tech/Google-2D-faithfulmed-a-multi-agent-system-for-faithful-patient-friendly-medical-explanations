"""Tests for the Simplifier's faithfulness checks. Run: python src/test_simplifier_checks.py

No API key: the LLM is a stub returning a fixed draft, so these assert on what
check_explanation() catches.

Regression guards for two bugs found in review:
  1. the ADK Extractor's boolean `negated` was ignored, so "No chest pain" loaded as
     present and a flipped rewrite passed clean
  2. a cue inside a finding the Extractor called "present" ("ECG showed no acute ST
     changes") was invisible, so dropping the "no" passed clean
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from simplifier import load_atoms, simplify, _is_true

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond: passed += 1; print(f"  PASS  {name}")
    else:    failed += 1; print(f"  FAIL  {name}")


def draft(atoms, *pairs):
    """Run the real pipeline with a stubbed model that returns these sentences."""
    sents = [{"text": t, "atom_ids": ids, "glossary_topic_ids": []} for t, ids in pairs]
    return simplify(atoms, rag_mode="off",
                    llm_fn=lambda s, u: json.dumps({"sentences": sents}))


def flags_for(atoms, text, ids):
    return draft(atoms, (text, ids))["sentences"][0]["flags"]


print("both Extractor formats set `assertion`")
check("ADK boolean negated=True -> negated",
      load_atoms([{"text": "No chest pain", "negated": True}])[0]["assertion"] == "negated")
check("negated=False -> present",
      load_atoms([{"text": "Chest pain", "negated": False}])[0]["assertion"] == "present")
check("absent negated -> present",
      load_atoms([{"text": "Chest pain"}])[0]["assertion"] == "present")
check("explicit assertion wins over negated",
      load_atoms([{"text": "x", "assertion": "uncertain", "negated": True}])[0]["assertion"]
      == "uncertain")
check("string \"true\" counts (models emit it)", _is_true("true") and _is_true("True"))
check("string \"false\" does NOT count", not _is_true("false"))
check("None and empty do not count", not _is_true(None) and not _is_true(""))

print("\nflipped meaning is caught (the reported bugs)")
NEG_ADK = [{"id": "a1", "type": "diagnosis", "text": "No chest pain", "negated": True}]
check("ADK-format negation flip flagged",
      "negation_may_be_lost" in flags_for(NEG_ADK, "You have chest pain.", ["a1"]))
check("ADK-format negation kept is clean",
      flags_for(NEG_ADK, "You do not have chest pain.", ["a1"]) == [])

IN_TEXT = [{"id": "a7", "type": "lab_value", "text": "ECG showed no acute ST changes",
            "assertion": "present", "subject": "patient"}]
check("negation inside a 'present' finding, flip flagged",
      "negation_may_be_lost" in flags_for(IN_TEXT, "Your ECG showed acute ST changes.", ["a7"]))
check("negation inside a 'present' finding, kept is clean",
      flags_for(IN_TEXT, "Your ECG showed no sudden changes.", ["a7"]) == [])

print("\nsame defect for uncertainty and family subject")
UNC = [{"id": "a1", "type": "diagnosis", "text": "Possible pneumonia"}]
check("hedge in wording, flip flagged",
      "uncertainty_may_be_lost" in flags_for(UNC, "You have pneumonia.", ["a1"]))
check("hedge kept is clean",
      flags_for(UNC, "You may have pneumonia, an infection in the lungs.", ["a1"]) == [])

FAM = [{"id": "a1", "type": "diagnosis", "text": "Mother had breast cancer"}]
check("family in wording, flip flagged",
      "family_subject_may_be_lost" in flags_for(FAM, "You had breast cancer.", ["a1"]))
check("family kept is clean",
      flags_for(FAM, "Your mother had breast cancer.", ["a1"]) == [])

print("\nstructured fields still work on their own")
check("assertion=negated with no cue in wording still flags",
      "negation_may_be_lost" in flags_for(
          [{"id": "a1", "type": "symptom", "text": "Fever", "assertion": "negated"}],
          "You have a fever.", ["a1"]))
check("subject=family with no cue in wording still flags",
      "family_subject_may_be_lost" in flags_for(
          [{"id": "a1", "type": "diagnosis", "text": "Breast cancer", "subject": "family"}],
          "You had breast cancer.", ["a1"]))

print("\ncorrect drafts are not flagged (no false positives)")
OK_ATOMS = [
    {"id": "a1", "type": "symptom", "text": "No chest pain", "negated": True},
    {"id": "a2", "type": "lab_value", "text": "ECG showed no acute ST changes"},
    {"id": "a3", "type": "diagnosis", "text": "Possible pneumonia"},
    {"id": "a4", "type": "diagnosis", "text": "Mother had breast cancer"},
    {"id": "a5", "type": "lab_value", "text": "Stress test was negative for ischemia"},
]
r = draft(OK_ATOMS,
          ("You do not have chest pain.", ["a1"]),
          ("Your ECG showed no sudden changes in your heart's rhythm.", ["a2"]),
          ("Your doctor thinks you may have pneumonia, an infection in the lungs.", ["a3"]),
          ("Your mother had breast cancer.", ["a4"]),
          ("Your stress test did not show ischemia, or reduced blood flow.", ["a5"]))
check("no sentence flagged", r["checks"]["flagged_sentences"] == [])
check("every atom covered", r["checks"]["uncovered_atoms"] == [])
check("draft reported clean", r["checks"]["clean"] is True)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
