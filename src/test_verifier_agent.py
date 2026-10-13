"""Tests for the Verifier agent. Run: python src/test_verifier_agent.py

The important ones are the guard tests and the merge/polarity tests. The LLM half
is covered by stubbed-model cases that exercise each failure mode: good JSON,
fenced JSON, bad JSON, API error, and a model that disagrees with the deterministic
guards. The latter is the test the design hangs on: the LLM saying "faithful: true"
must not override a guard that found a dropped dose.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from verifier_agent import (
    check_negations, check_numbers, measure_reading_level,
    deterministic_check, parse_verdict, run,
)

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond: passed += 1; print(f"  PASS  {name}")
    else:    failed += 1; print(f"  FAIL  {name}")


SOURCE = ("Patient with hypertension. Started metformin 500 mg twice daily. "
          "HbA1c 8.2%. Allergic to penicillin. No chest pain.")

ATOMS = [
    {"text": "hypertension", "type": "diagnosis", "term": "hypertension", "negated": False},
    {"text": "metformin 500 mg twice daily", "type": "medication",
     "term": "metformin", "negated": False},
    {"text": "HbA1c 8.2%", "type": "lab_value", "term": "HbA1c", "negated": False},
    {"text": "allergic to penicillin", "type": "allergy",
     "term": "penicillin", "negated": False},
    {"text": "chest pain", "type": "diagnosis", "term": "chest pain", "negated": True},
]

GOOD_DRAFT = ("You have high blood pressure. Take metformin 500 mg twice daily. "
              "Your HbA1c was 8.2%. You are allergic to penicillin. "
              "You have no chest pain.")

# --------------------------------------------------------------- deterministic guards

print("check_negations")
check("passes a draft that keeps the negation",
      check_negations(ATOMS, GOOD_DRAFT) == [])
flipped = GOOD_DRAFT.replace("You have no chest pain.", "You have chest pain.")
check("catches a dropped negation (chest pain)",
      any("chest pain" in e for e in check_negations(ATOMS, flipped)))
check("ignores atoms where the term never appears in the draft",
      check_negations([{"text": "stroke", "term": "stroke", "negated": True}],
                      "Take two aspirin tablets.") == [])

print("\ncheck_numbers")
check("passes when all atom numbers appear in the draft",
      check_numbers(ATOMS, GOOD_DRAFT) == [])
dropped_dose = GOOD_DRAFT.replace("500 mg twice daily", "once a day")
missing = check_numbers(ATOMS, dropped_dose)
check("catches a dropped dose", any("500" in m for m in missing))
check("passes when the atom has no numeric content",
      check_numbers([{"text": "hypertension", "term": "hypertension"}],
                    "You have high blood pressure.") == [])

print("\nreading level")
easy = "You have high blood pressure. Take your pill. Drink water."
hard = ("Subsequent to the administration of the pharmacological agent, the patient "
        "should abstain from operating heavy machinery for a duration of 24 hours.")
check("easy text under grade 8", measure_reading_level(easy)["fk_grade"] <= 8.0)
check("hard text above grade 8", measure_reading_level(hard)["fk_grade"] > 8.0)
check("deterministic_check sets reading_level_ok true for easy text",
      deterministic_check([], easy)["reading_level_ok"] is True)

# --------------------------------------------------------------- JSON parsing

print("\nparse_verdict")
raw_good = json.dumps({"faithful": True, "unsupported_claims": [], "omissions": [],
                       "negation_errors": [], "unexplained_terms": ["metformin"],
                       "reading_level_ok": True})
v = parse_verdict(raw_good)
check("parses a clean JSON reply", v is not None and v["faithful"] is True)
check("coerces unexplained_terms to a list", v["unexplained_terms"] == ["metformin"])

v2 = parse_verdict("```json\n" + raw_good + "\n```")
check("parses a fenced JSON reply", v2 is not None and v2["faithful"] is True)

v3 = parse_verdict("Sure, here is the verdict:\n" + raw_good + "\nHope this helps.")
check("parses when the model adds prose around the JSON",
      v3 is not None and v3["faithful"] is True)

v4 = parse_verdict('{"faithful": true}')
check("fills missing keys with defaults when the model skips them",
      v4 is not None
      and v4["unsupported_claims"] == []
      and v4["omissions"] == []
      and v4["unexplained_terms"] == [])

check("returns None for a wholly non-JSON reply",
      parse_verdict("I don't know, sorry.") is None)
check("returns None for empty text", parse_verdict("") is None)

# a model that writes a plain string in unsupported_claims instead of a list
v5 = parse_verdict('{"unsupported_claims": "metformin invented"}')
check("coerces a lone string into a one-item list",
      v5 is not None and v5["unsupported_claims"] == ["metformin invented"])

# --------------------------------------------------------------- run() with stubbed models

print("\nrun() happy path")
def faithful_model(prompt):
    return {"text": raw_good, "error": None, "model_used": "stub"}

r = run(SOURCE, ATOMS, GOOD_DRAFT, faithful_model, definitions="")
check("marks a clean draft faithful", r.verdict["faithful"] is True)
check("passes unexplained_terms through",
      "metformin" in r.verdict["unexplained_terms"])
check("records which model was used", r.model_used == "stub")
check("no error", r.error is None)
check("deterministic reading check runs even on happy path",
      "fk_grade" in r.deterministic["reading_metrics"])

print("\nrun() detects an unsupported claim")
raw_unsupp = json.dumps({"faithful": False,
                         "unsupported_claims": ["Call a cardiologist within 48 hours"],
                         "omissions": [], "negation_errors": [],
                         "unexplained_terms": [], "reading_level_ok": True})
def unsupp_model(prompt):
    return {"text": raw_unsupp, "error": None}
r2 = run(SOURCE, ATOMS,
         GOOD_DRAFT + " Call a cardiologist within 48 hours.", unsupp_model)
check("faithful is false when an unsupported claim is reported",
      r2.verdict["faithful"] is False)
check("the claim is in the verdict",
      "Call a cardiologist within 48 hours" in r2.verdict["unsupported_claims"])

print("\nrun() handles an unexplained-term case (NOT unsupported)")
# The Simplifier correctly kept 'metformin' as written. The verdict should say
# unexplained_terms:['metformin'] and remain faithful. This is the v2 rule.
raw_unex = json.dumps({"faithful": True, "unsupported_claims": [], "omissions": [],
                       "negation_errors": [], "unexplained_terms": ["metformin"],
                       "reading_level_ok": True})
def unex_model(prompt):
    return {"text": raw_unex, "error": None}
r3 = run(SOURCE, ATOMS, GOOD_DRAFT, unex_model)
check("verbatim-kept term does NOT make the draft unfaithful",
      r3.verdict["faithful"] is True)
check("the term is reported as unexplained",
      "metformin" in r3.verdict["unexplained_terms"])
check("the term is NOT in unsupported_claims",
      not any("metformin" in c.lower() for c in r3.verdict["unsupported_claims"]))

print("\nrun() merges deterministic guards with the LLM verdict")
# The LLM says "faithful: true". The draft silently drops the dose. The number
# guard must catch it and the final verdict must flip to unfaithful.
weakened = GOOD_DRAFT.replace("500 mg twice daily", "as needed")
def overclaiming_model(prompt):
    return {"text": raw_good, "error": None}      # claims faithful anyway
r4 = run(SOURCE, ATOMS, weakened, overclaiming_model)
check("deterministic guard overrides an over-confident LLM",
      r4.verdict["faithful"] is False)
check("the omission is on the verdict",
      any("500" in o for o in r4.verdict["omissions"]))

# Flipped negation: a changed meaning the LLM might not notice. Guard must catch it.
flipped_draft = GOOD_DRAFT.replace("no chest pain", "chest pain")
r5 = run(SOURCE, ATOMS, flipped_draft, overclaiming_model)
check("flipped negation is reported in negation_errors",
      any("chest pain" in e for e in r5.verdict["negation_errors"]))
check("flipped negation makes the draft unfaithful",
      r5.verdict["faithful"] is False)

print("\nrun() degrades gracefully when the LLM fails")
def erroring_model(prompt):
    return {"text": "", "error": "429 RESOURCE_EXHAUSTED"}
r6 = run(SOURCE, ATOMS, weakened, erroring_model)
check("the error is surfaced", "429" in (r6.error or ""))
check("deterministic finding still lands on the verdict",
      any("500" in o for o in r6.verdict["omissions"]))
check("faithful stays false when the guard found something, even without the LLM",
      r6.verdict["faithful"] is False)

def garbage_model(prompt):
    return {"text": "I'm not sure, probably faithful.", "error": None}
r7 = run(SOURCE, ATOMS, GOOD_DRAFT, garbage_model)
check("unparseable reply is reported in error",
      r7.error == "unparseable verdict")
check("deterministic verdict survives an unparseable reply",
      r7.verdict["reading_level_ok"] is True
      or r7.verdict["reading_level_ok"] is False)   # just that it is set, not thrown
check("no LLM verdict is stored when parsing fails",
      r7.llm_verdict is None)

print("\nrun() recomputes faithful from the merged lists, not from the model")
# The model asserts faithful:true while listing an omission. The final verdict
# must flip to faithful:false regardless of what the model wrote.
contradictory = json.dumps({"faithful": True, "unsupported_claims": [],
                            "omissions": ["dose dropped"], "negation_errors": [],
                            "unexplained_terms": [], "reading_level_ok": True})
def contradictory_model(prompt):
    return {"text": contradictory, "error": None}
r8 = run(SOURCE, ATOMS, GOOD_DRAFT, contradictory_model)
check("faithful is derived from the lists, not copied from the model",
      r8.verdict["faithful"] is False
      and r8.llm_verdict["faithful"] is True)

# --------------------------------------------------------------- model fallback

print("\nmodel fallback")
from verifier_agent import make_generate_fn

class FakeModels:
    def __init__(self, fail_for): self.fail_for = fail_for; self.calls = []
    def generate_content(self, model, contents, config=None):
        self.calls.append(model)
        if model in self.fail_for:
            raise Exception("503 UNAVAILABLE high demand")
        class R: text = raw_good
        return R()

class FakeClient:
    def __init__(self, fail_for): self.models = FakeModels(fail_for)

c = FakeClient(fail_for={"m-bad"})
fn = make_generate_fn(c, models=("m-bad", "m-good"), retries=1, base_delay=0)
out = fn("prompt")
check("falls back to the next model on 503", out["error"] is None)
check("reports which model answered", out.get("model_used") == "m-good")
check("tried the bad model first", c.models.calls[0] == "m-bad")

c2 = FakeClient(fail_for={"m-bad", "m-also-bad"})
fn2 = make_generate_fn(c2, models=("m-bad", "m-also-bad"), retries=1, base_delay=0)
out2 = fn2("prompt")
check("reports an error when every model fails", out2["error"] is not None)
check("no text when every model fails", out2["text"] == "")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
