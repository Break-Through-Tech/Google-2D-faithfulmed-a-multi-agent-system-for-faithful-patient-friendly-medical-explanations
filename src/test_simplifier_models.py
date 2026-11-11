"""Tests for the Simplifier's model chain and request deadline.
Run: python src/test_simplifier_models.py

No API key: the Gemini client is replaced with a fake that fails on demand.

Regression guard for 2026-11-09: gemini-3.8-flash left the demo prompt unanswered for
over five minutes. A hang is not a 503, so no retry rule caught it. Two things fix it -
a deadline on every request, and a chain so the run still produces an answer.
"""
import json
import sys
import types as pytypes
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent))
import simplifier as S

passed = failed = 0
def check(name, cond):
    global passed, failed
    if cond: passed += 1; print(f"  PASS  {name}")
    else:    failed += 1; print(f"  FAIL  {name}")

GOOD = json.dumps({"sentences": [
    {"text": "You do not have chest pain.", "atom_ids": ["a1"], "glossary_topic_ids": []}]})
ATOMS = [{"id": "a1", "type": "symptom", "text": "No chest pain", "negated": True}]

_real_time = S.time


class FakeModels:
    """Behaviour per model: ok | hang | busy | ratelimit | quota | blocked."""
    def __init__(self, behaviour):
        self.behaviour, self.seen, self.cfgs = behaviour, [], []

    def generate_content(self, *, model, contents, config):
        self.seen.append(model)
        self.cfgs.append(config)
        b = self.behaviour.get(model, "ok")
        if b == "hang":      raise httpx.ReadTimeout("timed out")
        if b == "busy":      raise Exception("503 UNAVAILABLE: overloaded")
        if b == "ratelimit": raise Exception("429 RESOURCE_EXHAUSTED")
        if b == "quota":     raise Exception("RESOURCE_EXHAUSTED: GenerateRequestsPerDay")
        if b == "blocked":   return pytypes.SimpleNamespace(text="")
        return pytypes.SimpleNamespace(text=GOOD)


def run(behaviour, **kw):
    """Returns (result | exception, fake). Sleeps are stubbed out."""
    fake = FakeModels(behaviour)
    S._client = pytypes.SimpleNamespace(models=fake)
    S.time = pytypes.SimpleNamespace(sleep=lambda s: None)
    try:
        return S.simplify(ATOMS, rag_mode="off", **kw), fake
    except Exception as e:
        return e, fake
    finally:
        S.time = _real_time


print("the requested chain order")
check("default chain is 3.5 -> 3.6 -> 3.1-lite",
      S.DEFAULT_MODELS == ("gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite"))
check("3.8-flash is gone", "gemini-3.8-flash" not in S.DEFAULT_MODELS)
check("matches readability_agent's chain",
      __import__("readability_agent").DEFAULT_MODELS == S.DEFAULT_MODELS)

print("\nevery request carries a deadline")
r, f = run({})
check("http_options.timeout is set", f.cfgs[0].http_options.timeout is not None)
check("timeout is REQUEST_TIMEOUT_S in milliseconds",
      f.cfgs[0].http_options.timeout == int(S.REQUEST_TIMEOUT_S * 1000))
_, f = run({}, timeout_s=12.5)
check("timeout_s is honoured per call", f.cfgs[0].http_options.timeout == 12500)

print("\na hang falls through instead of stopping the run")
r, f = run({"gemini-3.5-flash": "hang"})
check("tried 3.5 then 3.6", f.seen == ["gemini-3.5-flash", "gemini-3.6-flash"])
check("a hang is not retried on the same model", f.seen.count("gemini-3.5-flash") == 1)
check("the run still produced an explanation", r["explanation"] != "")
check("stats name the model that answered", r["stats"]["model"] == "gemini-3.6-flash")

r, f = run({"gemini-3.5-flash": "hang", "gemini-3.6-flash": "hang"})
check("falls through twice to the last model", r["stats"]["model"] == "gemini-3.1-flash-lite")

r, f = run(dict.fromkeys(S.DEFAULT_MODELS, "hang"))
check("all hanging raises rather than hanging forever", isinstance(r, RuntimeError))
check("the error names every model tried", all(m in str(r) for m in S.DEFAULT_MODELS))

print("\nretry in place only for transient server errors")
r, f = run({"gemini-3.5-flash": "busy"})
check("503 is retried on the same model before moving on",
      f.seen.count("gemini-3.5-flash") == S.RETRIES_PER_MODEL + 1)
check("then falls through and answers", r["stats"]["model"] == "gemini-3.6-flash")
check("retry budget is small now (was 8)", S.RETRIES_PER_MODEL <= 3)

print("\ndaily quota is not waited out")
r, f = run({"gemini-3.5-flash": "quota"})
check("quota moves on immediately, no retry", f.seen.count("gemini-3.5-flash") == 1)
check("and the run still answers", r["stats"]["model"] == "gemini-3.6-flash")
r, f = run(dict.fromkeys(S.DEFAULT_MODELS, "quota"))
check("all out of quota raises the 'waiting will not help' error",
      isinstance(r, RuntimeError) and "waiting will not help" in str(r))

print("\nmodel selection")
r, f = run({}, model="gemini-3.6-flash")
check("a single model string is honoured", f.seen == ["gemini-3.6-flash"])
check("and no chain is used", r["stats"]["models_configured"] == ["gemini-3.6-flash"])
r, f = run({"a": "hang"}, model=["a", "b"])
check("an explicit list is used in order", f.seen == ["a", "b"])
r, f = run({})
check("stats report the configured chain",
      r["stats"]["models_configured"] == list(S.DEFAULT_MODELS))

print("\nan empty/blocked response is treated as a failure, not an answer")
r, f = run({"gemini-3.5-flash": "blocked"})
check("falls through to the next model", r["stats"]["model"] == "gemini-3.6-flash")

print("\n_is_timeout recognises what the SDK raises")
check("httpx.ReadTimeout", S._is_timeout(httpx.ReadTimeout("x")))
check("httpx.ConnectTimeout", S._is_timeout(httpx.ConnectTimeout("x")))
check("504 in the message", S._is_timeout(Exception("504 Gateway Timeout")))
check("DEADLINE_EXCEEDED", S._is_timeout(Exception("DEADLINE_EXCEEDED")))
try:
    raise httpx.ReadTimeout("inner")
except httpx.ReadTimeout as inner:
    wrapped = Exception("api call failed")
    wrapped.__cause__ = inner
    check("a wrapped timeout is still a timeout", S._is_timeout(wrapped))
check("a 503 is NOT a timeout", not S._is_timeout(Exception("503 UNAVAILABLE")))
check("a quota error is NOT a timeout", not S._is_timeout(Exception("RESOURCE_EXHAUSTED")))

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
