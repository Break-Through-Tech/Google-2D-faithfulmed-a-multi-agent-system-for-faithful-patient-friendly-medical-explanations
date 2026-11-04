"""Tests for the FinalCheck agent. Run: python src/test_final_check_agent.py

No API key, no quota. The model is a scripted fake.

The important tests are the safety ones: a check that cannot run must never pass the text,
a model that says "faithful: true" while listing a problem must not pass the text, and a
fix that makes the text worse must not be shipped.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from final_check_agent import (  # noqa: E402
    CHECK_PROMPT, FIX_PROMPT, check_negations, check_numbers, deterministic_check,
    make_generate_fn, merge, parse_verdict, problem_count, run,
)

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")


SOURCE = ("Patient with hypertension. Started metformin 500 mg twice daily. HbA1c 8.2%. "
          "Allergic to penicillin. No chest pain.")
ATOMS = [
    {"text": "hypertension", "type": "diagnosis", "term": "hypertension", "negated": False},
    {"text": "metformin 500 mg twice daily", "type": "medication", "term": "metformin",
     "negated": False},
    {"text": "HbA1c 8.2%", "type": "lab_value", "term": "HbA1c", "negated": False},
    {"text": "allergic to penicillin", "type": "allergy", "term": "penicillin",
     "negated": False},
    {"text": "chest pain", "type": "diagnosis", "term": "chest pain", "negated": True},
]
GOOD = ("You have high blood pressure. Take metformin 500 mg two times a day. "
        "Your HbA1c was 8.2%. You are allergic to penicillin. You have no chest pain.")
BAD_DOSE = GOOD.replace("500 mg", "800 mg")
BAD_NEG = GOOD.replace("You have no chest pain.", "You have chest pain.")


def verdict(ok=True, **lists):
    v = {"faithful": ok, "unsupported_claims": [], "omissions": [], "negation_errors": [],
         "unexplained_terms": [], "reading_level_ok": True}
    v.update(lists)
    return json.dumps(v)


class Script:
    """Fake model. Answers check prompts and fix prompts from two scripted queues."""

    def __init__(self, checks=(), fixes=()):
        self.checks, self.fixes = list(checks), list(fixes)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        queue = self.checks if prompt.startswith("You are checking") else self.fixes
        item = queue.pop(0) if queue else {"text": "", "error": "script ran out"}
        if isinstance(item, str):
            item = {"text": item, "error": None}
        item.setdefault("model_used", "fake")
        return item


print("number guard")
n = check_numbers(SOURCE, ATOMS, GOOD)
check("a faithful rewrite has no number problems", not n["omissions"] and
      not n["unsupported_claims"])
n = check_numbers(SOURCE, ATOMS, BAD_DOSE)
check("a changed dose is an omission (500 gone)", any("500" in o for o in n["omissions"]))
check("a changed dose is an invented number (800 new)",
      any("800" in u for u in n["unsupported_claims"]))
check("'twice' rewritten as '2 times' is not an invented number",
      not check_numbers(SOURCE, ATOMS, GOOD.replace("two times", "2 times"))
      ["unsupported_claims"])
check("'8.20%' matches '8.2%'",
      not check_numbers(SOURCE, ATOMS, GOOD.replace("8.2%", "8.20%"))["omissions"])
check("'1,000' matches '1000'",
      not check_numbers("Drink 1000 mL.", [], "Drink 1,000 mL of water.")["omissions"])
check("HbA1c's '1' is not read as a number",
      not check_numbers("Your HbA1c is high.", [], "Your HbA1c is high.")
      ["unsupported_claims"])
check("without atoms, falls back to the source's numbers",
      bool(check_numbers("Take 2 tablets at 9 pm.", [], "Take tablets at night.")
           ["omissions"]))

print("\nnegation guard")
check("'no chest pain' kept is fine", check_negations(ATOMS, GOOD) == [])
check("'no chest pain' flipped is caught", len(check_negations(ATOMS, BAD_NEG)) == 1)
check("a negated finding left out is not a negation error (it is the LLM's omission)",
      check_negations(ATOMS, "You have high blood pressure.") == [])
check("'denies chest pain' counts as negated",
      check_negations(ATOMS, "The patient denies chest pain.") == [])
check("negation a few words before still counts",
      check_negations(ATOMS, "There is no sign of any chest pain.") == [])

print("\ndeterministic_check")
d = deterministic_check(SOURCE, ATOMS, GOOD)
check("reports a reading grade", isinstance(d["fk_grade"], float))
check("easy text is reading_level_ok", d["reading_level_ok"])

print("\nparse_verdict")
check("parses plain JSON", parse_verdict(verdict(True))["faithful"] is True)
check("parses ```json fences", parse_verdict("```json\n" + verdict(True) + "\n```")
      ["faithful"] is True)
check("parses JSON wrapped in prose",
      parse_verdict("Here you go: " + verdict(False) + " thanks")["faithful"] is False)
check("garbage gives None, not a pass", parse_verdict("I think it is fine") is None)
check("empty gives None", parse_verdict("") is None)
check("a string 'true' is not faithful", parse_verdict('{"faithful": "true"}')
      ["faithful"] is False)
check("a single string is coerced to a list",
      parse_verdict('{"omissions": "dose"}')["omissions"] == ["dose"])

print("\nmerge recomputes faithful")
m = merge(parse_verdict(verdict(True, omissions=["the dose"])), {"reading_level_ok": True})
check("model says faithful but lists an omission: NOT faithful", m["faithful"] is False)
m = merge(parse_verdict(verdict(True)), {"omissions": ["500 missing"],
                                         "reading_level_ok": True})
check("model misses a dropped number, the guard catches it", m["faithful"] is False)
m = merge(parse_verdict(verdict(True, unexplained_terms=["metformin"])),
          {"reading_level_ok": True})
check("an unexplained term alone does not fail the check", m["faithful"] is True)
m = merge(parse_verdict(verdict(True, omissions=["Dose"])), {"omissions": ["dose"]})
check("duplicates are removed", m["omissions"] == ["Dose"])
mv = merge(parse_verdict('{"faithful": false}'), {"reading_level_ok": True})
check("model says NOT faithful but names no problem: nothing to fix, so it passes",
      mv["faithful"] is True)
check("...and that disagreement is recorded", mv["unexplained_fail"] is True)
check("a normal pass is not marked as a disagreement",
      merge(parse_verdict(verdict(True)), {})["unexplained_fail"] is False)
check("reading_level_ok comes from textstat, not the model",
      merge(parse_verdict(verdict(True)), {"reading_level_ok": False})
      ["reading_level_ok"] is False)
check("problem_count counts the three failure lists",
      problem_count({"unsupported_claims": ["a"], "omissions": ["b", "c"],
                     "negation_errors": [], "unexplained_terms": ["x"]}) == 3)

print("\nrun(): pass on the first check")
s = Script(checks=[verdict(True)])
r = run(SOURCE, GOOD, s, atoms=ATOMS)
check("passes", r.passed and not r.needs_review)
check("one call only", r.llm_calls == 1 and r.fix_rounds == 0)
check("text unchanged", r.final == GOOD and not r.changed)
check("the prompt contains the source and the final text",
      SOURCE in s.prompts[0] and GOOD in s.prompts[0])

print("\nrun(): fail, fix, pass")
s = Script(checks=[verdict(False, unsupported_claims=["800 mg"]), verdict(True)],
           fixes=[GOOD])
r = run(SOURCE, BAD_DOSE, s, atoms=ATOMS)
check("passes after one fix", r.passed and r.fix_rounds == 1)
check("ships the fixed text", r.final == GOOD and r.changed)
check("3 calls: check, fix, check", r.llm_calls == 3)
check("the fix prompt carries the verdict", "800 mg" in s.prompts[1] and
      s.prompts[1].startswith("Fix the final patient explanation"))
check("reason says it passed after a fix", r.reason == "passed after 1 fix(es)")

print("\nrun(): the guards override a model that says 'faithful'")
s = Script(checks=[verdict(True), verdict(True)], fixes=[GOOD])
r = run(SOURCE, BAD_DOSE, s, atoms=ATOMS)
check("a changed dose is not passed even though the model said faithful",
      r.history[0]["verdict"]["faithful"] is False)
check("so it was fixed and then passed", r.passed and r.final == GOOD)

print("\nrun(): still failing after the cap")
s = Script(checks=[verdict(False, omissions=["a"])] * 3, fixes=["try 1", "try 2"])
r = run(SOURCE, BAD_NEG, s, atoms=ATOMS, max_rounds=2)
check("flagged needs_review", r.needs_review and not r.passed)
check("stops at 2 fixes", r.fix_rounds == 2)
check("5 calls max: check, fix, check, fix, check", r.llm_calls == 5)
check("ends on a check, never an unchecked fix", len(r.history) == 3)
check("reason says it hit the cap", "still failing" in r.reason)

print("\nrun(): a fix that makes it worse is not shipped")
s = Script(checks=[verdict(False, omissions=["a"]),
                   verdict(False, omissions=["a", "b", "c"]),
                   verdict(False, omissions=["a", "b"])],
           fixes=[GOOD + " worse", GOOD + " still worse"])
r = run(SOURCE, GOOD, s, atoms=ATOMS)
check("keeps the version with the fewest problems (the original)", r.final == GOOD)
check("still flagged", r.needs_review)
check("verdict matches the shipped text", r.verdict["omissions"] == ["a"])

print("\nrun(): 'not faithful' with no reason given")
s = Script(checks=['{"faithful": false, "unexplained_terms": ["NSAIDs"]}'],
           fixes=[GOOD + " NSAIDs are pain relievers."])
r = run(SOURCE, GOOD, s, atoms=ATOMS)
check("passes without a fix: there is nothing to fix", r.passed and r.fix_rounds == 0)
check("the text is unchanged", r.final == GOOD)

print("\nrun(): the fix step only sees the problems")
s = Script(checks=[verdict(False, omissions=["the dose"], unexplained_terms=["NSAIDs"]),
                   verdict(True)], fixes=[GOOD])
r = run(SOURCE, GOOD, s, atoms=ATOMS)
check("the omission is sent to the fix step", "the dose" in s.prompts[1])
check("unexplained terms are not sent to the fix step", "NSAIDs" not in s.prompts[1])
check("the fix prompt forbids adding definitions",
      "Do not add explanations, definitions" in s.prompts[1])

print("\nrun(): the check itself fails")
s = Script(checks=[{"text": "", "error": "429 RESOURCE_EXHAUSTED"}])
r = run(SOURCE, GOOD, s, atoms=ATOMS)
check("an API error is NOT a pass", not r.passed and r.needs_review)
check("no fix is attempted without a verdict", r.fix_rounds == 0 and r.llm_calls == 1)
check("the error is reported", "429" in (r.error or ""))
check("text unchanged", r.final == GOOD)

s = Script(checks=["Looks fine to me!"])
r = run(SOURCE, GOOD, s, atoms=ATOMS)
check("an unreadable verdict is NOT a pass", not r.passed and r.needs_review)
check("reason says the check could not run", r.reason == "check could not run")

print("\nrun(): the recheck after a fix fails")
s = Script(checks=[verdict(False, omissions=["a"]),
                   {"text": "", "error": "503 UNAVAILABLE"}], fixes=[GOOD + " fixed"])
r = run(SOURCE, GOOD, s, atoms=ATOMS)
check("the unchecked fix is not shipped", r.final == GOOD)
check("flagged needs_review", r.needs_review)

print("\nrun(): the fix call fails")
s = Script(checks=[verdict(False, omissions=["a"])], fixes=[{"text": "", "error": "503"}])
r = run(SOURCE, GOOD, s, atoms=ATOMS)
check("keeps the checked text and flags it", r.final == GOOD and r.needs_review)
check("reason says the fix could not run", r.reason == "fix could not run")

print("\nrun(): max_rounds=0 means check only")
s = Script(checks=[verdict(False, omissions=["a"])])
r = run(SOURCE, GOOD, s, atoms=ATOMS, max_rounds=0)
check("one call, no fix, flagged", r.llm_calls == 1 and r.fix_rounds == 0 and
      r.needs_review)

print("\nfaithfulness report (the 'Outputs to user' box on the v2 diagram)")
s = Script(checks=[verdict(True, unexplained_terms=["metformin"])])
rep = run(SOURCE, GOOD, s, atoms=ATOMS).report()
check("pass first time: status passed", rep["status"] == "passed")
check("pass first time: nothing listed as fixed or remaining",
      rep["fixed"] == [] and rep["remaining"] == [])
check("unexplained terms are reported as a readability note",
      rep["terms_not_explained"] == ["metformin"])
check("reports a reading grade", isinstance(rep["reading_grade"], float))
check("explains the grade in words", "School grade" in rep["reading_level"] and
      "Goal: grade 8" in rep["reading_level"])
from final_check_agent import reading_level_text
check("grade 7.6 meets the goal", "meets the goal" in reading_level_text(7.6))
check("grade 11.2 is above the goal", "above the goal" in reading_level_text(11.2))
check("summary is plain words, no JSON", "{" not in rep["summary"])

s = Script(checks=[verdict(False, unsupported_claims=["800 mg"]), verdict(True)],
           fixes=[GOOD])
rep = run(SOURCE, BAD_DOSE, s, atoms=ATOMS).report()
check("fixed: status fixed", rep["status"] == "fixed")
check("fixed: lists what was fixed", "800 mg" in rep["fixed"])
check("fixed: nothing remaining", rep["remaining"] == [])

s = Script(checks=[verdict(False, omissions=["the dose"])] * 3, fixes=["a", "b"])
rep = run(SOURCE, GOOD, s, atoms=ATOMS).report()
check("still failing: status needs_review", rep["status"] == "needs_review")
check("still failing: lists what remains", "the dose" in rep["remaining"])
check("still failing: says how many fixes were tried", "2 fix attempt" in rep["summary"])

s = Script(checks=[{"text": "", "error": "429"}])
r = run(SOURCE, GOOD, s, atoms=ATOMS)
check("check could not run: needs_review, never passed",
      r.report()["status"] == "needs_review")
check("the report is included in as_dict()", r.as_dict()["report"]["status"] ==
      "needs_review")

print("\nprompts")
check("check prompt has every placeholder the code fills",
      all(k in CHECK_PROMPT for k in ("{source}", "{atoms}", "{definitions}", "{text}")))
check("fix prompt has every placeholder the code fills",
      all(k in FIX_PROMPT for k in ("{source}", "{text}", "{verdict}")))

print("\nmake_generate_fn")


class _Resp:
    def __init__(self, text):
        self.text = text


class _Models:
    def __init__(self, errors):
        self.errors, self.calls = errors, []

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        if model in self.errors:
            raise RuntimeError(self.errors[model])
        return _Resp("ok")


class _Client:
    def __init__(self, errors):
        self.models = _Models(errors)


c = _Client({"m-bad": "503 UNAVAILABLE"})
out = make_generate_fn(c, models=("m-bad", "m-good"), retries=2, base_delay=0)("p")
check("falls back after a 503", out["model_used"] == "m-good" and out["error"] is None)
check("retried the 503 before moving on", c.models.calls == ["m-bad", "m-bad", "m-good"])

c = _Client({"m-bad": "429 RESOURCE_EXHAUSTED"})
out = make_generate_fn(c, models=("m-bad", "m-good"), retries=3, base_delay=0)("p")
check("a 429 is not retried on the same model", c.models.calls == ["m-bad", "m-good"])

c = _Client({"a": "429", "b": "429"})
out = make_generate_fn(c, models=("a", "b"), retries=1, base_delay=0)("p")
check("every model failing returns an error and no text",
      out["error"] and out["text"] == "" and out["model_used"] is None)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
