"""Tests for the Readability agent. Run: python src/test_readability_agent.py

The important ones are the rejection tests. This agent runs after the Verifier, so if it
silently breaks a fact nothing downstream catches it.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from readability_agent import measure, fact_check, run, meets_target

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond: passed += 1; print(f"  PASS  {name}")
    else:    failed += 1; print(f"  FAIL  {name}")

HARD = ("Subsequent to the administration of the aforementioned pharmacological agent, "
        "the patient should abstain from operating heavy machinery for a period of 24 "
        "hours, and should not consume alcohol concurrently with this medication.")

print("measurement")
m = measure(HARD)
check("grades a hard sentence above 8", m["fk_grade"] > 8)
check("reports jargon density", 0 <= m["jargon_density"] <= 1)
check("reports sentence length", m["avg_sentence_words"] > 0)
check("meets_target false for hard text", not meets_target(HARD))
check("meets_target true for easy text", meets_target("You should rest. Drink water."))

print("\nfact_check catches the dangerous edits")
check("number changed is caught",
      not fact_check("Take 500 mg daily.", "Take 800 mg daily.")["ok"])
check("number dropped is caught",
      not fact_check("Take 500 mg twice daily.", "Take the pill daily.")["ok"])
check("negation lost is caught",
      not fact_check("Do not drive for 24 hours.", "Drive after 24 hours.")["ok"])
check("heavy content loss is caught",
      not fact_check("A B C D E F G H I J K L M N O P", "A B C")["ok"])
check("a clean simplification passes",
      fact_check("Take 500 mg twice daily. Do not drink alcohol.",
                 "Take 500 mg two times a day. Do not drink alcohol.")["ok"])

print("\nrun() with a stubbed model")
def good_model(prompt):
    return {"text": "Do not use heavy machines for 24 hours after you take this medicine. "
                    "Do not drink alcohol with it.", "error": None}
r = run(HARD, good_model)
check("a good rewrite is accepted", r.passes_used == 1)
check("grade went down", r.grade_drop > 0)
check("no rejected passes", r.rejected_passes == [])

def unsafe_model(prompt):
    # flips the negation: dangerous
    return {"text": "Use heavy machines 24 hours after taking this. Drink alcohol with it.",
            "error": None}
r2 = run(HARD, unsafe_model)
check("an unsafe rewrite is REJECTED", r2.final == HARD)
check("rejection is recorded", len(r2.rejected_passes) == 1)
check("rejection says why", "negation" in str(r2.rejected_passes).lower())

def already_easy(prompt):
    raise AssertionError("should not be called")
r3 = run("You should rest. Drink water. Call us if it hurts.", already_easy)
check("skips the model when already at target", r3.passes_used == 0)
check("met_target true", r3.met_target)

def erroring_model(prompt):
    return {"text": "", "error": "429 RESOURCE_EXHAUSTED"}
r4 = run(HARD, erroring_model)
check("an API error keeps the original", r4.final == HARD)
check("error is recorded", "429" in (r4.error or ""))

def never_good_enough(prompt):
    return {"text": HARD + " Additionally, further considerations may apply henceforth.",
            "error": None}
r5 = run(HARD, never_good_enough, max_passes=2)
check("bounded at max_passes", r5.passes_used <= 2)

print("\nmodel fallback")
from readability_agent import make_generate_fn

class FakeModels:
    def __init__(self, fail_for): self.fail_for=fail_for; self.calls=[]
    def generate_content(self, model, contents, config=None):
        self.calls.append(model)
        if model in self.fail_for:
            raise Exception("503 UNAVAILABLE high demand")
        class R: text = "rewritten simply"
        return R()
class FakeClient:
    def __init__(self, fail_for): self.models=FakeModels(fail_for)

c = FakeClient(fail_for={"m-bad"})
fn = make_generate_fn(c, models=("m-bad","m-good"), retries=1, base_delay=0)
out = fn("prompt")
check("falls back to the next model on 503", out["error"] is None)
check("reports which model answered", out.get("model_used") == "m-good")
check("tried the bad model first", c.models.calls[0] == "m-bad")

c2 = FakeClient(fail_for={"m-bad","m-also-bad"})
fn2 = make_generate_fn(c2, models=("m-bad","m-also-bad"), retries=1, base_delay=0)
out2 = fn2("prompt")
check("reports an error when every model fails", out2["error"] is not None)
check("no text when every model fails", out2["text"] == "")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
