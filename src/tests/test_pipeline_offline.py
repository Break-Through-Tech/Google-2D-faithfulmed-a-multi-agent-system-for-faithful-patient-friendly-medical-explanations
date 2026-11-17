"""End-to-end wiring test for the ADK pipeline, with a fake model. No API key, no quota.

Run: python src/tests/test_pipeline_offline.py

What it proves: every {placeholder} in every agent's instruction resolves from session
state, state flows from agent to agent in the documented order, the Simplifier (with its RAG
lookup) writes the definitions the Verifier reads, Readability's own fact check runs inside
the pipeline, and the final text is checked, fixed at most twice when
it fails, and flagged for review if it still fails.

What it does NOT prove: that a real model follows the instructions. That needs a real run.
"""
import asyncio
import json
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "agents"))
sys.path.insert(0, str(ROOT / "notebooks"))

from google.adk.models import BaseLlm, LlmResponse  # noqa: E402
from google.adk.runners import InMemoryRunner  # noqa: E402
from google.genai import types  # noqa: E402

import adk_pipeline as P  # noqa: E402

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
    if instruction.startswith("You are checking"):
        return "finalcheck" if "FINALTEXT" in instruction else "verifier"
    if instruction.startswith("Fix the final patient explanation"):
        return "finalrefiner"
    if instruction.startswith("Fix the explanation"):
        return "refiner"
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
MAIN_VERDICTS: list = []    # scripted outcomes for each main-loop Verifier call, in order
CALLS: list = []            # every model call, in order
READABILITY_OUT = {"text": FINAL}   # what the fake Readability model returns
SIMPLIFIER_OUT = {"text": DRAFT}    # what the fake Simplifier model writes
REFINER_OUT = {"text": DRAFT}       # what the fake (main loop) Refiner returns
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
        elif who == "verifier":
            reply = _verdict(MAIN_VERDICTS.pop(0) if MAIN_VERDICTS else True)
        else:
            reply = {"extractor": json.dumps(ATOMS),
                     "refiner": REFINER_OUT["text"], "finalrefiner": FIXED}.get(who, "?")
        yield LlmResponse(content=types.Content(role="model",
                                                parts=[types.Part(text=reply)]))


class StubRetriever:
    """Stands in for medlineplus_index.Retriever: exact (synonym) matches only, no API."""
    TOPICS = {"high blood pressure": (1, "High Blood Pressure", "Blood pressure that stays too high."),
              "a1c": (2, "A1C", "A blood test for average blood sugar.")}
    ALIASES = {"hypertension": "high blood pressure", "hba1c": "a1c"}

    def __init__(self):
        self.surface_to_id = {k: str(v[0]) for k, v in self.TOPICS.items()}
        self.surface_to_id.update({a: self.surface_to_id[t] for a, t in self.ALIASES.items()})

    def search(self, query, k=3, min_score=0.0, use_synonyms=True):
        key = self.ALIASES.get(query.lower(), query.lower())
        if key not in self.TOPICS:
            return []
        tid, title, definition = self.TOPICS[key]
        return [{"topic_id": tid, "title": title, "url": "u", "definition": definition,
                 "score": 1.0, "match_type": "synonym", "also_known_as": []}]


def fake_simplifier_llm(system, user):
    """The Simplifier's model: one sentence per DRAFT sentence, citing every atom."""
    CALLS.append("simplifier")
    SEEN.setdefault("simplifier", system + user)
    ids = re.findall(r'"id": "([^"]+)"', user)
    sentences = [t.strip() + "." for t in SIMPLIFIER_OUT["text"].split(". ") if t.strip(". ")]
    return json.dumps({"sentences": [{"text": t.rstrip(".") + ".", "atom_ids": ids,
                                      "glossary_topic_ids": []} for t in sentences]})


def fake_readability_gen(prompt):
    """The Readability model: always returns READABILITY_OUT."""
    CALLS.append("readability")
    SEEN.setdefault("readability", prompt)
    return {"text": READABILITY_OUT["text"], "error": None}


async def _run(final_outcomes, readability_text=FINAL, simplifier_text=DRAFT,
               refiner_text=DRAFT, main_outcomes=()):
    FINAL_VERDICTS[:] = list(final_outcomes)
    MAIN_VERDICTS[:] = list(main_outcomes)
    READABILITY_OUT["text"] = readability_text
    SIMPLIFIER_OUT["text"] = simplifier_text
    REFINER_OUT["text"] = refiner_text
    CALLS.clear()
    # target_grade=0 so Readability always makes a pass: the test drafts are already easy.
    agent = P.build_pipeline(model=FakeLlm(), retriever=StubRetriever(), target_grade=0,
                             simplifier_llm_fn=fake_simplifier_llm,
                             readability_generate_fn=fake_readability_gen)
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


# The draft passes the Verifier first time, so the main loop stops before the Refiner.
HEAD = ["Extractor", "Simplifier", "Verifier", "StopIfDraftFaithful",
        "Readability", "FinalCheck", "FinalGuard"]

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
            "final_verdict", "needs_review", "simplifier_report", "readability_report"):
    check(f"`{key}` is in state", key in state)

print("\nfaithfulness report")
rep = json.loads(state.get("faithfulness_report", "{}"))
check("the pipeline writes a faithfulness report", rep.get("status") == "passed")
check("the report has a plain summary", "original text" in rep.get("summary", ""))

print("\nretrieval (inside the Simplifier)")
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
check("Simplifier sees the atoms", "metformin 500 mg twice daily" in SEEN.get("simplifier", ""))
check("Readability rewrites the Simplifier's draft", DRAFT in SEEN.get("readability", ""))
check("Extractor asks for allergies and negation", "allergy" in SEEN.get("extractor", "")
      and "negated" in SEEN.get("extractor", ""))

print("\nextractor guards ran inside the pipeline")
_atoms = json.loads(state["atoms"])
check("atoms carry the merged schema (id, assertion, subject)",
      all({"id", "assertion", "subject", "negated", "term"} <= set(a) for a in _atoms))
check("negated atom kept its negation", any(a["text"] == "chest pain" and a["negated"]
                                            and a["assertion"] == "negated" for a in _atoms))
check("extractor report written to state", "extractor_report" in state)
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

print("\nsimplifier and readability reports")
srep = json.loads(state["simplifier_report"])
check("simplifier report carries its checks and stats",
      "checks" in srep and srep["stats"]["n_glossary_exact"] == 2)
rrep = json.loads(state["readability_report"])
check("readability report carries before/after grades",
      "fk_grade" in rrep["before"] and "fk_grade" in rrep["after"])

print("\nscenario E1: Readability rejects a rewrite that changes a dose")
BAD = FINAL.replace("500 mg", "800 mg")
order, state, calls = asyncio.run(_run([True], readability_text=BAD))
check("the changed dose never reaches the final text", "800" not in state["final"])
check("the rejected pass is recorded",
      json.loads(state["readability_report"])["rejected_passes"] != [])

print("\nscenario E: the Simplifier invents a dose and every model misses it; the guard does not")
BAD_DRAFT = DRAFT.replace("500 mg", "800 mg")
order, state, calls = asyncio.run(_run([True, True], readability_text=BAD,
                                       simplifier_text=BAD_DRAFT, refiner_text=BAD_DRAFT))
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

print("\nmain loop: Verifier <-> Refiner stops as soon as the draft passes")
order, state, calls = asyncio.run(_run([True]))
check("faithful first time: no Refiner call at all", calls.count("refiner") == 0)
check("faithful first time: one Verifier call", calls.count("verifier") == 1)
order, state, calls = asyncio.run(_run([True], main_outcomes=[False, True]))
check("fails once: one fix, then the re-check passes and the loop stops",
      calls.count("refiner") == 1 and calls.count("verifier") == 2)
check("fails once: the Refiner's fix is what goes on", calls.index("refiner")
      < len(calls) - 1 - calls[::-1].index("verifier"))
order, state, calls = asyncio.run(_run([True], main_outcomes=[False, False, False]))
check("keeps failing: capped at 2 rounds", calls.count("refiner") == 2
      and calls.count("verifier") == 2)

print("\nmodel wiring")
try:
    P.build_pipeline(model=FakeLlm())
    check("a non-Gemini model without Simplifier/Readability functions is refused", False)
except ValueError:
    check("a non-Gemini model without Simplifier/Readability functions is refused", True)
p = P.build_pipeline(model="gemini-x")
check("a model name reaches the Simplifier and Readability",
      p.sub_agents[1].models == ("gemini-x",) and p.sub_agents[3].models == ("gemini-x",))
check("ADK-run agents get Gemini with 503 retries",
      p.sub_agents[0].model.model == "gemini-x"
      and 503 in p.sub_agents[0].model.retry_options.http_status_codes)

print("\nretries: a fake Gemini server answers 503 twice, then succeeds")
import http.server, threading  # noqa: E401,E402
HITS = []
class _Busy(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        HITS.append(self.path)
        if len(HITS) <= 2:
            body, code = b'{"error": {"code": 503, "message": "busy", "status": "UNAVAILABLE"}}', 503
        else:
            body, code = json.dumps({"candidates": [{"content": {"role": "model", "parts": [
                {"text": "ok"}]}, "finishReason": "STOP"}]}).encode(), 200
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.end_headers(); self.wfile.write(body)
    def log_message(self, *a):
        pass
srv = http.server.HTTPServer(("127.0.0.1", 0), _Busy)
threading.Thread(target=srv.serve_forever, daemon=True).start()
from google.adk.agents import LlmAgent  # noqa: E402
from google.adk.models.google_llm import Gemini  # noqa: E402
import os  # noqa: E402
os.environ.setdefault("GOOGLE_API_KEY", "offline-test-key")
fast = P._RETRY.model_copy(update={"initial_delay": 0.05, "max_delay": 0.1})
probe = LlmAgent(name="Probe", instruction="say ok", model=Gemini(
    model="gemini-x", retry_options=fast, base_url=f"http://127.0.0.1:{srv.server_port}"))
async def _probe():
    r = InMemoryRunner(agent=probe, app_name="p")
    sess = await r.session_service.create_session(app_name="p", user_id="u")
    out = []
    async for ev in r.run_async(user_id="u", session_id=sess.id, new_message=types.Content(
            role="user", parts=[types.Part(text="hi")])):
        if ev.content and ev.content.parts:
            out.append(ev.content.parts[0].text)
    return out
try:
    got = asyncio.run(_probe())
except Exception as e:  # noqa: BLE001
    got = [f"error: {e}"]
srv.shutdown()
check("two 503s are retried and the third attempt's answer comes through",
      len(HITS) == 3 and got == ["ok"])
if not (len(HITS) == 3 and got == ["ok"]):
    print("     hits:", len(HITS), "got:", got)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
