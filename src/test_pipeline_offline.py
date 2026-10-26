"""End-to-end wiring test for the ADK pipeline, with a fake model. No API key, no quota.

Run: python src/test_pipeline_offline.py

What it proves: every {placeholder} in every agent's instruction resolves from session
state, state flows from agent to agent in the documented order, the Retriever writes what
the Simplifier and Verifier read, and the final text is checked, fixed at most twice when
it fails, and flagged for review if it still fails.

What it does NOT prove: that a real model follows the instructions. That needs a real run.
"""
import asyncio
import csv
import json
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

from google.adk.models import BaseLlm, LlmResponse  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402

import adk_pipeline as P  # noqa: E402
from glossary_lookup import GlossaryLookup  # noqa: E402

SOURCE = ("Patient with hypertension. Started metformin 500 mg twice daily. HbA1c 8.2%. "
          "Allergic to penicillin. No chest pain.")

ATOMS = [
    {"text": "hypertension", "type": "diagnosis", "term": "hypertension", "negated": False},
    {"text": "metformin 500 mg twice daily", "type": "medication", "term": "metformin",
     "negated": False},
    {"text": "HbA1c 8.2%", "type": "lab_value", "term": "HbA1c", "negated": False},
    {"text": "allergic to penicillin", "type": "allergy", "term": "penicillin",
     "negated": False},
    {"text": "chest pain", "type": "diagnosis", "term": "", "negated": True},
]

SEEN: dict[str, str] = {}   # agent marker -> rendered system instruction


def _which(instruction: str) -> str:
    if instruction.startswith("Extract every"):
        return "extractor"
    if instruction.startswith("Rewrite the SOURCE"):
        return "simplifier"
    if instruction.startswith("You are checking"):
        return "finalcheck" if "FINALTEXT" in instruction else "verifier"
    if instruction.startswith("Fix the final patient explanation"):
        return "finalrefiner"
    if instruction.startswith("Fix the explanation"):
        return "refiner"
    if instruction.startswith("Rewrite the explanation"):
        return "readability"
    return "unknown"


DRAFT = ("You have high blood pressure. You started metformin 500 mg twice daily. "
         "Your HbA1c was 8.2%. You are allergic to penicillin. You have no chest pain.")
FINAL = ("FINALTEXT You have high blood pressure. Take metformin 500 mg two times a day. "
         "Your HbA1c was 8.2%. You are allergic to penicillin. You have no chest pain.")
FIXED = FINAL + " FIXED"


def _verdict(ok: bool) -> str:
    return json.dumps({"faithful": ok, "unsupported_claims": [] if ok else ["invented"],
                       "omissions": [], "negation_errors": [],
                       "unexplained_terms": ["metformin"], "reading_level_ok": True})


VERDICT = _verdict(True)
FINAL_VERDICTS: list = []   # scripted outcomes for each final check, in order
CALLS: list = []            # every model call, in order
READABILITY_OUT = {"text": FINAL}   # what the fake Readability step returns
LAST: dict[str, str] = {}   # agent marker -> most recent rendered instruction


class FakeLlm(BaseLlm):
    model: str = "fake-model"

    async def generate_content_async(self, llm_request, stream=False):
        si = llm_request.config.system_instruction if llm_request.config else ""
        si = si if isinstance(si, str) else str(si)
        who = _which(si)
        SEEN.setdefault(who, si)
        LAST[who] = si
        CALLS.append(who)
        if who == "finalcheck":
            nxt = FINAL_VERDICTS.pop(0) if FINAL_VERDICTS else True
            reply = nxt if isinstance(nxt, str) else _verdict(nxt)
        else:
            reply = {"extractor": json.dumps(ATOMS), "simplifier": DRAFT, "verifier": VERDICT,
                     "refiner": DRAFT, "readability": READABILITY_OUT["text"],
                     "finalrefiner": FIXED}.get(who, "?")
        yield LlmResponse(content=types.Content(role="model",
                                                parts=[types.Part(text=reply)]))


def _glossary():
    d = Path(tempfile.mkdtemp())
    with open(d / "t.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["title", "summary", "url"])
        w.writerow(["High Blood Pressure", "Blood pressure that stays too high.", "u1"])
        w.writerow(["A1C", "A blood test for average blood sugar.", "u2"])
    with open(d / "s.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["surface", "canonical", "kind", "canonical_url"])
        for r in [("High Blood Pressure", "High Blood Pressure"),
                  ("hypertension", "High Blood Pressure"), ("A1C", "A1C"), ("HbA1c", "A1C")]:
            w.writerow([r[0], r[1], "x", ""])
    return GlossaryLookup(d / "t.csv", d / "s.csv")


async def _run(final_outcomes, readability_text=FINAL):
    FINAL_VERDICTS[:] = list(final_outcomes)
    READABILITY_OUT["text"] = readability_text
    CALLS.clear()
    agent = P.build_pipeline(model=FakeLlm(), lookup=_glossary())
    runner = InMemoryRunner(agent=agent, app_name="faithfulmed")
    session = await runner.session_service.create_session(
        app_name="faithfulmed", user_id="u", state=P.new_session_state(SOURCE))
    order = []
    async for ev in runner.run_async(
            user_id="u", session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part(text="go")])):
        if ev.author and (not order or order[-1] != ev.author):
            order.append(ev.author)
    final = await runner.session_service.get_session(
        app_name="faithfulmed", user_id="u", session_id=session.id)
    return order, final.state, list(CALLS)


passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond: passed += 1; print(f"  PASS  {name}")
    else:    failed += 1; print(f"  FAIL  {name}")


HEAD = ["Extractor", "Retriever", "Simplifier", "Verifier", "Refiner",
        "Verifier", "Refiner", "Readability", "FinalCheck", "FinalGuard"]

# Scenario A: the final check passes first time. The fix loop stops at once.
order, state, calls = asyncio.run(_run([True]))
print("scenario A: final check passes first time")
check("agents run in the documented order",
      order == HEAD + ["StopIfFaithful", "NeedsReviewFlag"])
if order != HEAD + ["StopIfFaithful", "NeedsReviewFlag"]:
    print("     got:", order)
check("no extra fix when it already passed", calls.count("finalrefiner") == 0)
check("not flagged for review", state.get("needs_review") is False)

print("\nstate written by each step")
for key in ("atoms", "definitions", "unexplained_terms", "draft", "verdict", "final",
            "final_verdict", "needs_review"):
    check(f"`{key}` is in state", key in state)

print("\nfaithfulness report")
rep = json.loads(state.get("faithfulness_report", "{}"))
check("the pipeline writes a faithfulness report", rep.get("status") == "passed")
check("the report has a plain summary", "original text" in rep.get("summary", ""))

print("\nretriever")
check("found hypertension via the synonym table", "High Blood Pressure" in state["definitions"])
check("found HbA1c via the synonym table", "A1C" in state["definitions"])
check("metformin listed as having no definition", "metformin" in state["unexplained_terms"])
check("penicillin listed as having no definition", "penicillin" in state["unexplained_terms"])
check("negated atom with no term is not looked up", '""' not in state["unexplained_terms"])

print("\nevery placeholder resolved")
for who, si in SEEN.items():
    leftovers = [k for k in ("{source}", "{atoms}", "{definitions}", "{draft}", "{verdict}",
                             "{final}") if k in si]
    check(f"{who}: no unresolved placeholders", not leftovers)

print("\nthe v2 fixes are actually wired")
check("Verifier sees the SOURCE, not only the atoms", SOURCE in SEEN.get("verifier", ""))
check("Verifier sees what was retrieved", "High Blood Pressure" in SEEN.get("verifier", ""))
check("Verifier is told an unexplained term is NOT unsupported",
      "is NOT an unsupported claim" in SEEN.get("verifier", ""))
check("FinalCheck reads the FINAL text, after Readability",
      FINAL in SEEN.get("finalcheck", ""))
check("FinalCheck also sees the SOURCE", SOURCE in SEEN.get("finalcheck", ""))
check("Simplifier sees the definitions", "High Blood Pressure" in SEEN.get("simplifier", ""))
check("Extractor asks for allergies and negation", "allergy" in SEEN.get("extractor", "")
      and "negated" in SEEN.get("extractor", ""))

print("\nscenario B: final check fails once, then passes")
order, state, calls = asyncio.run(_run([False, True]))
check("goes straight to the Refiner, not back to the Verifier",
      order[len(HEAD):len(HEAD) + 2] == ["StopIfFaithful", "FinalRefiner"])
check("the recheck is guarded too",
      order[len(HEAD) + 2:len(HEAD) + 4] == ["FinalRecheck", "FinalRecheckGuard"])
check("exactly one fix", calls.count("finalrefiner") == 1)
check("the fix is checked again", calls[-1] == "finalcheck")
check("the fixed text is what ships", state["final"] == FIXED)
check("not flagged for review", state.get("needs_review") is False)
check("main Verifier not re-run after Readability",
      "verifier" not in calls[calls.index("readability"):])
fr = SEEN.get("finalrefiner", "")
check("the final Refiner sees the final verdict", '"faithful"' in fr)
check("the final Refiner sees the SOURCE", SOURCE in fr)
check("final Refiner: no unresolved placeholders",
      not any(k in fr for k in ("{source}", "{final}", "{final_verdict}")))

print("\nscenario C: final check keeps failing")
order, state, calls = asyncio.run(_run([False, False, False, False]))
check("capped at 2 fixes", calls.count("finalrefiner") == 2)
check("3 final checks in total: 1 + 1 per fix", calls.count("finalcheck") == 3)
check("ends on a check, never on an unchecked fix", calls[-1] == "finalcheck")
check("flagged for review", state.get("needs_review") is True)
check("still finishes and returns text", bool(state.get("final")))
check("the report says it needs review",
      json.loads(state["faithfulness_report"])["status"] == "needs_review")

print("\nscenario D: unreadable final verdict counts as a fail, never a pass")
check("unparseable verdict is not faithful", P._is_faithful("not json at all") is False)
check("fenced JSON verdict is read", P._is_faithful('```json\n{"faithful": true}\n```'))

print("\nscenario E: the model misses a changed dose, the guard does not")
BAD = FINAL.replace("500 mg", "800 mg")
order, state, calls = asyncio.run(_run([True, True], readability_text=BAD))
check("the model said faithful, but the check still failed and a fix ran",
      calls.count("finalrefiner") == 1)
check("the fix prompt names the invented dose", "number 800 is not in the source"
      in LAST.get("finalrefiner", ""))
check("the fix prompt names the dropped dose", "500" in LAST.get("finalrefiner", "")
      and "missing" in LAST.get("finalrefiner", ""))
check("the fixed text ships and is not flagged",
      state["final"] == FIXED and state.get("needs_review") is False)

print("\nscenario F: an unreadable final verdict goes through the guard as a fail")
order, state, calls = asyncio.run(_run(["looks fine to me", True]))
check("unreadable verdict triggers a fix, not a pass", calls.count("finalrefiner") == 1)
order, state, calls = asyncio.run(_run(['{"faithful": false}', True]))
check("'faithful: false' that names no problem does not trigger a fix",
      calls.count("finalrefiner") == 0 and state.get("needs_review") is False)
check("the final Refiner is told not to add definitions",
      "Do not add explanations, definitions" in LAST.get("finalrefiner", ""))

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
