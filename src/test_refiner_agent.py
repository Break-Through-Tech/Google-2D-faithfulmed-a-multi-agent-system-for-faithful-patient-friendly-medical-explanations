"""Tests for the Refiner agent. Run: python src/test_refiner_agent.py

The important ones are the short-circuit, the unsafe-rewrite rejection, and the
bounded loop. This agent is allowed to edit the draft, so a bad "fix" is worse
than leaving the flagged text alone.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from refiner_agent import (
    parse_verdict, is_faithful, fact_guard, run, run_loop, MAX_ROUNDS,
)

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond: passed += 1; print(f"  PASS  {name}")
    else:    failed += 1; print(f"  FAIL  {name}")


SOURCE = ("Patient with hypertension. Started metformin 500 mg twice daily. "
          "HbA1c 8.2%. Allergic to penicillin. No chest pain.")

GOOD = ("You have high blood pressure. Take metformin 500 mg twice daily. "
        "Your HbA1c was 8.2%. You are allergic to penicillin. "
        "You have no chest pain.")

FAITHFUL = {"faithful": True, "unsupported_claims": [], "omissions": [],
            "negation_errors": [], "unexplained_terms": ["metformin"],
            "reading_level_ok": True}

def never_call(prompt):
    raise AssertionError("should not call the model")


# --------------------------------------------------------------- parse / faithful

print("parse_verdict / is_faithful")
check("parses a dict", parse_verdict(FAITHFUL)["unexplained_terms"] == ["metformin"])
check("parses a JSON string",
      parse_verdict(json.dumps(FAITHFUL))["unexplained_terms"] == ["metformin"])
fenced = "```json\n" + json.dumps(FAITHFUL) + "\n```"
check("parses fenced JSON", parse_verdict(fenced)["faithful"] is True)
check("returns None for prose", parse_verdict("I don't know") is None)
check("returns None for empty", parse_verdict("") is None)

class _VR:
    def __init__(self, verdict): self.verdict = verdict
check("accepts an object with .verdict",
      parse_verdict(_VR(FAITHFUL))["unexplained_terms"] == ["metformin"])

check("empty failure lists are faithful", is_faithful(FAITHFUL) is True)
check("unexplained_terms do not make it unfaithful", is_faithful(FAITHFUL) is True)
check("an omission is unfaithful",
      is_faithful({**FAITHFUL, "faithful": True, "omissions": ["dose dropped"]}) is False)
check("faithful boolean is ignored when lists are empty",
      is_faithful({**FAITHFUL, "faithful": False}) is True)
check("unparseable is not faithful", is_faithful("sorry") is False)

# --------------------------------------------------------------- fact_guard

print("\nfact_guard")
check("a clean restore from the source passes",
      fact_guard(SOURCE, GOOD + " Call a cardiologist.", GOOD)["ok"])
check("inventing a dose not in the source is caught",
      not fact_guard(SOURCE, GOOD, GOOD.replace("500", "800"))["ok"])
check("empty rewrite is caught", not fact_guard(SOURCE, GOOD, "   ")["ok"])
check("percent normalization: 8.2% in source matches 8.2 in rewrite",
      fact_guard("HbA1c 8.2%.", "Your HbA1c was 8.2%.", "Your HbA1c was 8.2")["ok"])

# --------------------------------------------------------------- run() short-circuit

print("\nrun() short-circuit")
r = run(SOURCE, GOOD, FAITHFUL, never_call)
check("faithful draft does not call the model", r.rounds_used == 0)
check("draft unchanged", r.final == GOOD)
check("stopped_reason is already_faithful", r.stopped_reason == "already_faithful")
check("records the verdict", r.verdicts and r.verdicts[0]["unexplained_terms"] == ["metformin"])

unexplained_only = {**FAITHFUL, "faithful": False}  # boolean lies; lists are empty
r_u = run(SOURCE, GOOD, unexplained_only, never_call)
check("does not rewrite unexplained_terms", r_u.rounds_used == 0 and r_u.final == GOOD)

# --------------------------------------------------------------- run() targeted fixes

print("\nrun() targeted fixes")
bad_extra = GOOD + " Call a cardiologist within 48 hours."
unsupp = {**FAITHFUL, "faithful": False,
          "unsupported_claims": ["Call a cardiologist within 48 hours"],
          "unexplained_terms": []}

def drop_extra(prompt):
    check("prompt contains the unsupported claim",
          "Call a cardiologist within 48 hours" in prompt)
    return {"text": GOOD, "error": None, "model_used": "stub"}

r2 = run(SOURCE, bad_extra, unsupp, drop_extra)
check("removes an unsupported claim", r2.final == GOOD)
check("one round used", r2.rounds_used == 1)
check("stopped_reason is refined", r2.stopped_reason == "refined")
check("records which model was used", r2.model_used == "stub")

omitted = GOOD.replace("500 mg twice daily", "as needed")
omission_v = {**FAITHFUL, "faithful": False,
              "omissions": ["metformin 500 mg twice daily"],
              "unexplained_terms": []}

def restore_dose(prompt):
    return {"text": GOOD, "error": None}

r3 = run(SOURCE, omitted, omission_v, restore_dose)
check("restores an omitted dose from the source", "500" in r3.final)

flipped = GOOD.replace("You have no chest pain.", "You have chest pain.")
neg_v = {**FAITHFUL, "faithful": False,
         "negation_errors": ["'chest pain' asserted without its negation"],
         "unexplained_terms": []}

def restore_neg(prompt):
    return {"text": GOOD, "error": None}

r4 = run(SOURCE, flipped, neg_v, restore_neg)
check("restores a dropped negation", "no chest pain" in r4.final.lower())

# The model wrote faithful:true while listing an omission. Must still refine.
contradictory = {**FAITHFUL, "faithful": True, "omissions": ["dose dropped"]}
r5 = run(SOURCE, omitted, contradictory, restore_dose)
check("refines when faithful:true but an omission is listed", r5.rounds_used == 1)

# --------------------------------------------------------------- run() rejects unsafe / errors

print("\nrun() rejects unsafe rewrites and degrades on errors")
def invent_dose(prompt):
    return {"text": GOOD.replace("500", "800"), "error": None}

r6 = run(SOURCE, omitted, omission_v, invent_dose)
check("invented dose is REJECTED", r6.final == omitted)
check("rejection is recorded", len(r6.rejected_rounds) == 1)
check("rejection says why", "invented" in str(r6.rejected_rounds).lower())
check("stopped_reason is rejected_unsafe", r6.stopped_reason == "rejected_unsafe")

def erroring(prompt):
    return {"text": "", "error": "429 RESOURCE_EXHAUSTED"}
r7 = run(SOURCE, omitted, omission_v, erroring)
check("API error keeps the original", r7.final == omitted)
check("error is recorded", "429" in (r7.error or ""))
check("stopped_reason is model_error", r7.stopped_reason == "model_error")

def empty(prompt):
    return {"text": "   ", "error": None}
r8 = run(SOURCE, omitted, omission_v, empty)
check("empty reply keeps the original", r8.final == omitted)
check("empty reply is recorded", r8.error == "empty response")

r9 = run(SOURCE, GOOD, "not json at all", never_call)
check("unparseable verdict does not call the model", r9.rounds_used == 0)
check("unparseable verdict is recorded", r9.stopped_reason == "unparseable_verdict")

# JSON string verdict, the way ADK stores {verdict}
r10 = run(SOURCE, bad_extra, json.dumps(unsupp), drop_extra)
check("accepts a JSON-string verdict from ADK state", r10.final == GOOD)

# --------------------------------------------------------------- run_loop()

print("\nrun_loop()")
verify_calls = []

def verify_until_clean(draft):
    verify_calls.append(draft)
    if "cardiologist" in draft:
        return unsupp
    return FAITHFUL

verify_calls.clear()
rl = run_loop(SOURCE, bad_extra, drop_extra, verify_until_clean, max_rounds=2)
check("stops once the draft becomes faithful", rl.stopped_reason == "became_faithful")
check("only one refine", rl.rounds_used == 1)
check("verified the original then the fix", len(verify_calls) == 2)
check("did not spend a second refine", rl.final == GOOD)

def already_good(draft):
    return FAITHFUL
rl2 = run_loop(SOURCE, GOOD, never_call, already_good, max_rounds=2)
check("loop skips the model when already faithful", rl2.rounds_used == 0)
check("loop stopped_reason already_faithful", rl2.stopped_reason == "already_faithful")

def always_unfaithful(draft):
    return unsupp

refine_calls = {"n": 0}
def never_quite(prompt):
    refine_calls["n"] += 1
    # Stay unfaithful (keep the extra claim) but do not invent numbers.
    return {"text": GOOD + " Please see a cardiologist.", "error": None}

rl3 = run_loop(SOURCE, bad_extra, never_quite, always_unfaithful, max_rounds=2)
check("bounded at max_rounds", rl3.rounds_used == 2)
check("default cap is 2", MAX_ROUNDS == 2)
check("model called at most twice", refine_calls["n"] == 2)
check("stopped_reason is max_rounds", rl3.stopped_reason == "max_rounds")

rl4 = run_loop(SOURCE, omitted, invent_dose, lambda d: omission_v, max_rounds=2)
check("loop rejects an unsafe rewrite and keeps the original", rl4.final == omitted)
check("loop does not continue after a rejection", rl4.rounds_used == 0)

# --------------------------------------------------------------- model fallback

print("\nmodel fallback")
from refiner_agent import make_generate_fn

class FakeModels:
    def __init__(self, fail_for): self.fail_for = fail_for; self.calls = []
    def generate_content(self, model, contents, config=None):
        self.calls.append(model)
        if model in self.fail_for:
            raise Exception("503 UNAVAILABLE high demand")
        class R: text = GOOD
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
