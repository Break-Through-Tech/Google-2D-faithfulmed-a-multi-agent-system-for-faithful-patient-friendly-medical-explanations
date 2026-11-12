"""Offline tests for src/extractor_agent.py. No API key, no quota.

Run: python src/test_extractor_agent.py

Each test hands run() a fake model reply and checks what the guards do with it. What
this does NOT prove: that Gemini extracts well. That is what eval_extractor.py measures.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extractor_agent as X  # noqa: E402

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}" + (f"\n        {detail}" if detail else ""))


def fake(reply, error=None):
    seen = {}

    def gen(prompt):
        seen["prompt"] = prompt
        return {"text": reply if isinstance(reply, str) else json.dumps(reply),
                "error": error, "model_used": "fake"}
    return gen, seen


NOTE = ("Patient with hypertension. Started metformin 500 mg twice daily. HbA1c 8.2%. "
        "Allergic to penicillin. No chest pain. Mother had breast cancer.")

MTS = ("HISTORY OF PRESENT ILLNESS: , A 56-year-old male seen by Dr. XXXX with chest "
       "pressure.,MEDICATIONS: , Lisinopril 10 mg daily.,ALLERGIES: , NKDA.,"
       "REVIEW OF SYSTEMS: , Denies fever or chills.,PLAN: , Follow up in 2 weeks.")

MEDAESQA = ("Lifestyle changes may prevent and treat sleep apnea [28646811, 33659106]. "
            "Sleep apnea can be prevented by losing weight [33659106, 12693795].")

# ---------------------------------------------------------------- schema
print("schema")
gen, _ = fake([
    {"text": "hypertension", "type": "diagnosis", "term": "hypertension", "assertion": "present"},
    {"text": "metformin 500 mg twice daily", "type": "medication", "term": "metformin"},
    {"text": "chest pain", "type": "diagnosis", "term": "", "assertion": "negated"},
    {"text": "Mother had breast cancer", "type": "diagnosis", "term": "breast cancer",
     "assertion": "present", "subject": "family"},
])
r = X.run(NOTE, gen)
keys = {"id", "text", "type", "term", "assertion", "negated", "subject", "section"}
check("every atom has the merged schema", all(keys <= set(a) for a in r.atoms))
check("ids are a1..an in order", [a["id"] for a in r.atoms] == [f"a{i}" for i in range(1, len(r.atoms) + 1)])
check("negated is derived from assertion", all(a["negated"] == (a["assertion"] == "negated") for a in r.atoms))
check("missing assertion defaults to present", r.atoms[1]["assertion"] == "present")
check("family subject kept", r.atoms[3]["subject"] == "family")

gen, _ = fake([{"text": "chest pain", "type": "diagnosis", "negated": True}])
r = X.run(NOTE, gen)
check("v2-style `negated: true` with no assertion becomes assertion negated",
      r.atoms[0]["assertion"] == "negated" and r.atoms[0]["negated"] is True)

gen, _ = fake([{"text": "metformin 500 mg twice daily", "type": "Drug"},
               {"text": "HbA1c 8.2%", "type": "lab test result"},
               {"text": "hypertension", "type": "condition", "assertion": "absent"}])
r = X.run(NOTE, gen)
check("loose type labels are mapped (Drug -> medication)", r.atoms[0]["type"] == "medication")
check("unknown type becomes other, with a warning",
      r.atoms[1]["type"] == "other" and any("unknown type" in w for w in r.warnings))
check("assertion alias 'absent' -> negated", r.atoms[2]["assertion"] == "negated")

# ---------------------------------------------------------------- guards
print("\nguards")
gen, _ = fake([
    {"text": "metformin 1000 mg twice daily", "type": "medication", "term": "metformin"},
    {"text": "metformin 500 mg twice daily", "type": "medication", "term": "metformin"},
])
r = X.run(NOTE, gen)
check("an atom with a number not in the source is rejected",
      len(r.atoms) == 1 and "1000" in r.rejected[0]["reason"], str(r.rejected))

gen, _ = fake([{"text": "patient should start aspirin to lower stroke risk", "type": "instruction"}])
r = X.run(NOTE, gen)
check("an invented fact is rejected as ungrounded",
      not r.atoms and "not grounded" in r.rejected[0]["reason"], str(r.rejected))

gen, _ = fake([{"text": "seen by Dr. XXXX", "type": "other"},
               {"text": "Lisinopril 10 mg daily", "type": "medication", "term": "Lisinopril"}])
r = X.run(MTS, gen)
check("a redaction placeholder is rejected", any("redaction" in x["reason"] for x in r.rejected))

gen, _ = fake([{"text": "HbA1c 8.2%", "type": "lab_value", "term": "glycated hemoglobin"}])
r = X.run(NOTE, gen)
check("a term not in the source is cleared (the Retriever would look up a word the "
      "source never used)", r.atoms[0]["term"] == "" and r.warnings)

gen, _ = fake([{"text": "chest pain", "type": "diagnosis", "term": "chest pain", "assertion": "present"}])
r = X.run(NOTE, gen)
check("an asserted finding the source negates is flagged",
      any("source negates" in w for w in r.warnings), str(r.warnings))

gen, _ = fake([{"text": "hypertension", "type": "diagnosis"},
               {"text": "Hypertension", "type": "diagnosis"}])
r = X.run(NOTE, gen)
check("duplicates are collapsed", len(r.atoms) == 1)

gen, _ = fake([{"text": "hypertension", "type": "diagnosis"}])
r = X.run(NOTE, gen)
check("allergy in source but no allergy atom -> warning",
      any("allerg" in w for w in r.warnings), str(r.warnings))

gen, _ = fake([{"text": "Lisinopril 10 mg daily", "type": "medication", "term": "Lisinopril"}])
r = X.run(MTS, gen)
nk = [a for a in r.atoms if a["type"] == "allergy"]
check("NKDA is added as a negated allergy when the model drops it",
      len(nk) == 1 and nk[0]["negated"] and nk[0].get("added_by") == "rule:nkda", str(r.atoms))

# ---------------------------------------------------------------- source shapes
print("\nsource shapes")
secs = X.split_sections(MTS)
heads = [h for h, _ in secs]
check("MTSamples comma-delimited headers are split",
      heads == ["HISTORY OF PRESENT ILLNESS", "MEDICATIONS", "ALLERGIES", "REVIEW OF SYSTEMS", "PLAN"],
      str(heads))
check("section bodies are clean", dict(secs)["MEDICATIONS"] == "Lisinopril 10 mg daily")
check("'BP: 120/80' is not taken as a header",
      [h for h, _ in X.split_sections("VITAL SIGNS: , BP: 120/80, pulse 72.")] == ["VITAL SIGNS"])
p = X.build_prompt(MTS)
check("prompt tags sections with a type hint", "[MEDICATIONS] (usually medication)" in p)
check("prose with no headers is passed through", X.split_sections(MEDAESQA)[0][0] == "")
p = X.build_prompt(MEDAESQA)
check("PubMed citation markers are stripped from the prompt", "28646811" not in p)

gen, _ = fake([{"text": "lifestyle changes may prevent sleep apnea [28646811]", "type": "other"}])
r = X.run(MEDAESQA, gen)
check("a PMID copied into an atom is rejected as a number not in the source",
      not r.atoms and "28646811" in r.rejected[0]["reason"])

gen, _ = fake([{"text": "Lifestyle changes may prevent sleep apnea", "type": "other",
                "term": "sleep apnea", "assertion": "uncertain"}])
r = X.run(MEDAESQA, gen)
check("hedged claim kept as uncertain", r.atoms and r.atoms[0]["assertion"] == "uncertain")

# ---------------------------------------------------------------- parsing and failure
print("\nparsing and failure")
check("parses ```json fences", X.parse_atoms('```json\n[{"text": "x"}]\n```') == [{"text": "x"}])
check("parses a {\"atoms\": [...]} wrapper", X.parse_atoms('{"atoms": [{"text": "x"}]}') == [{"text": "x"}])
check("parses a list with prose around it", X.parse_atoms('Here you go: [{"text": "x"}] done') == [{"text": "x"}])
check("empty list is [] not None", X.parse_atoms("[]") == [])
check("garbage is None", X.parse_atoms("I cannot help with that") is None)

gen, _ = fake("", error="503 UNAVAILABLE")
r = X.run(MTS, gen)
check("model failure -> error set, no guessed atoms except the NKDA rule",
      r.error and all(a.get("added_by") for a in r.atoms))
gen, _ = fake("not json at all")
r = X.run(NOTE, gen)
check("unparseable reply -> error 'unparseable reply'", r.error == "unparseable reply")

gen, seen = fake([])
X.run(NOTE, gen)
check("prompt starts with 'Extract every' (the offline pipeline test relies on it)",
      seen["prompt"].startswith("Extract every"))
check("prompt has no unresolved placeholders", "{" not in seen["prompt"])

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
