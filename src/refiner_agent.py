"""Refiner agent: targeted fixes to passages the Verifier flags.

Task #7, October milestone. Owner: Isabella.

Job
---
Read the SOURCE, the current DRAFT, and the Verifier's VERDICT. Fix only what
the verdict flagged, in at most two rounds, so the patient-friendly explanation
stays faithful to the source.

Why this is a standalone module and not just the LlmAgent in adk_pipeline.py
--------------------------------------------------------------------------
Same two reasons as src/verifier_agent.py and src/readability_agent.py.

1. The LLM rewrite is not trusted on its own. A "fix" that invents a dose or
   drops a "do not" is worse than the flagged draft. After every rewrite, a
   guard checks the candidate against the SOURCE. An unsafe rewrite is
   REJECTED and the previous draft is kept.

2. The ADK LoopAgent always spends both iterations unless something escalates.
   This module stops as soon as the verdict is faithful, so evaluation and
   demos do not burn a second round on a clean draft.

What the verdict looks like
---------------------------
Shape is pinned in docs/architecture.md, produced by src/verifier_agent.py:

    {
      "faithful": bool,
      "unsupported_claims": [str, ...],
      "omissions": [str, ...],
      "negation_errors": [str, ...],
      "unexplained_terms": [str, ...],
      "reading_level_ok": bool,
    }

Design rules this module encodes
--------------------------------
- A faithful draft is returned unchanged. The model is not called.
- unexplained_terms are a readability gap, not a faithfulness failure. The
  Refiner does not rewrite them; that is the Readability agent's job.
- `faithful` is derived from the three failure lists, not copied from the
  verdict's boolean. Same rule the Verifier uses: a model can write
  `faithful: true` while listing an omission.
- Restored omissions may only use facts that appear in the SOURCE.
- Numbers in the rewrite that do not appear in the SOURCE are invented; reject.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict

MAX_ROUNDS = 2

# Keep this identical to notebooks/adk_pipeline.py::REFINER_INSTRUCTION so the
# standalone eval path and the ADK LlmAgent ask for the same edit.
PROMPT = (
    "Fix the explanation using the verdict. Change nothing else.\n\n"
    "SOURCE:\n{source}\n\n"
    "EXPLANATION:\n{draft}\n\n"
    "VERDICT:\n{verdict}\n\n"
    "Rules:\n"
    "- Remove every unsupported claim.\n"
    "- Add back every omission, worded plainly, using only the SOURCE.\n"
    "- Fix every negation error.\n"
    "- Do not rewrite sentences the verdict did not flag.\n"
    "- If the verdict says faithful is true, return the explanation unchanged.\n"
    "- Return only the explanation text."
)

_NUM = re.compile(r"\b\d+(?:\.\d+)?\s?%?")
_FAILURE_KEYS = ("unsupported_claims", "omissions", "negation_errors")


def _as_list(v) -> list[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []


def _coerce(obj: dict) -> dict:
    """Fill missing keys with safe defaults and coerce types to the schema."""
    return {
        "faithful": bool(obj.get("faithful", False)),
        "unsupported_claims": _as_list(obj.get("unsupported_claims")),
        "omissions": _as_list(obj.get("omissions")),
        "negation_errors": _as_list(obj.get("negation_errors")),
        "unexplained_terms": _as_list(obj.get("unexplained_terms")),
        "reading_level_ok": bool(obj.get("reading_level_ok", False)),
    }


def parse_verdict(verdict) -> dict | None:
    """Accept a dict, JSON string, or an object with a `.verdict` attribute.

    ADK stores `{verdict}` as text; the standalone Verifier returns a
    `VerifierResult`. Both have to land on the same schema. Returns None only
    when nothing JSON-shaped is present.
    """
    if verdict is None:
        return None
    if hasattr(verdict, "verdict"):
        verdict = verdict.verdict
    if isinstance(verdict, dict):
        return _coerce(verdict)
    if not isinstance(verdict, str) or not verdict.strip():
        return None
    text = verdict.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        text = text[start:end + 1]
    try:
        obj = json.loads(text)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return _coerce(obj)


def is_faithful(verdict) -> bool:
    """True only when unsupported_claims, omissions, and negation_errors are empty.

    Do not trust the verdict's `faithful` boolean. unexplained_terms do not count:
    a term copied from the source is faithful (architecture v2).
    """
    v = parse_verdict(verdict)
    if v is None:
        return False
    return not any(v.get(k) for k in _FAILURE_KEYS)


def _nums(text: str) -> list[str]:
    return [n.strip().rstrip("%").strip() for n in _NUM.findall(text or "")]


def fact_guard(source: str, before: str, after: str) -> dict:
    """Reject a rewrite that invents facts the source does not support.

    Narrow on purpose, same idea as readability_agent.fact_check. Refiner is
    allowed to add omitted source facts and to delete unsupported sentences, so
    we do NOT compare before vs after for dropped numbers. We compare after vs
    SOURCE: a number that never appeared in the source is a hallucinated dose.
    """
    source_nums = set(_nums(source))
    after_nums = set(_nums(after))
    invented = sorted(n for n in after_nums if n not in source_nums)

    problems = []
    if invented:
        problems.append(f"numbers invented (not in source): {invented}")
    if not (after or "").strip():
        problems.append("empty rewrite")

    return {"ok": not problems, "problems": problems}


@dataclass
class RefinerResult:
    source: str
    original: str
    final: str
    rounds_used: int = 0
    stopped_reason: str = "unchanged"
    rejected_rounds: list = field(default_factory=list)
    verdicts: list = field(default_factory=list)
    model_used: str | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _refine_once(source: str, draft: str, verdict: dict, generate_fn,
                 res: RefinerResult, round_n: int) -> str | None:
    """Call the model once. Returns the new draft, or None if we must stop."""
    out = generate_fn(PROMPT.format(
        source=source, draft=draft,
        verdict=json.dumps(verdict, ensure_ascii=False),
    ))
    res.model_used = out.get("model_used")
    if out.get("error"):
        res.error = out["error"]
        res.stopped_reason = "model_error"
        return None

    candidate = (out.get("text") or "").strip()
    if not candidate:
        res.error = "empty response"
        res.stopped_reason = "empty_response"
        return None

    check = fact_guard(source, draft, candidate)
    if not check["ok"]:
        res.rejected_rounds.append({"round": round_n, "problems": check["problems"]})
        res.stopped_reason = "rejected_unsafe"
        return None
    return candidate


def run(source: str, draft: str, verdict, generate_fn) -> RefinerResult:
    """One targeted fix. The ADK LoopAgent calls this shape once per round.

    `generate_fn(prompt) -> {"text": str, "error": str|None, "model_used"?: str}`.
    Keeping the LLM behind a callable means tests do not need an API key, and the
    agent is model-agnostic by construction.
    """
    res = RefinerResult(source=source, original=draft, final=draft)
    parsed = parse_verdict(verdict)
    if parsed is None:
        res.error = "unparseable verdict"
        res.stopped_reason = "unparseable_verdict"
        return res
    res.verdicts.append(parsed)

    if is_faithful(parsed):
        res.stopped_reason = "already_faithful"
        return res

    candidate = _refine_once(source, draft, parsed, generate_fn, res, round_n=1)
    if candidate is None:
        return res

    res.final = candidate
    res.rounds_used = 1
    res.stopped_reason = "refined"
    return res


def run_loop(source: str, draft: str, generate_fn, verify_fn,
             max_rounds: int = MAX_ROUNDS) -> RefinerResult:
    """Bounded Verifier <-> Refiner loop.

    `verify_fn(draft)` returns a verdict dict, a JSON string, or a VerifierResult.
    Stops at faithful, at `max_rounds`, or if a rewrite is rejected / the model
    fails. The loop ends on a Refiner write, matching the ADK LoopAgent order
    (Verifier then Refiner). FinalCheck in the pipeline is what re-reads that
    last edit; this module does not add a third verify.
    """
    res = RefinerResult(source=source, original=draft, final=draft)

    for i in range(max(max_rounds, 0)):
        parsed = parse_verdict(verify_fn(res.final))
        if parsed is None:
            res.error = "unparseable verdict"
            res.stopped_reason = "unparseable_verdict"
            return res
        res.verdicts.append(parsed)
        if is_faithful(parsed):
            res.stopped_reason = "already_faithful" if i == 0 else "became_faithful"
            return res

        candidate = _refine_once(source, res.final, parsed, generate_fn, res,
                                 round_n=i + 1)
        if candidate is None:
            return res

        res.final = candidate
        res.rounds_used = i + 1

    res.stopped_reason = "max_rounds"
    return res


# --------------------------------------------------------------- model wiring

# Model availability has moved twice during this project: gemini-2.0-flash and
# gemini-2.5-flash are retired (404), and gemini-3.6-flash returned 503 on long
# prompts on 2026-10-13 while 3.5-flash handled the same prompts. So the model
# is a parameter with a fallback chain, not a constant. Same chain as
# src/verifier_agent.py so the whole pipeline reports under the same models.
DEFAULT_MODELS = ("gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite")


def make_generate_fn(client, models=DEFAULT_MODELS, temperature: float = 0.0,
                     retries: int = 3, base_delay: float = 5.0):
    """Build a generate_fn for run(). Mirrors verifier_agent.make_generate_fn.

    Retries transient failures with backoff, then falls back to the next model.
    Returns the usual {"text": str, "error": str|None, "model_used": str|None}.
    """
    import time
    try:
        from google.genai import types as _genai_types
    except ImportError:  # tests use a fake client and do not need the SDK
        _genai_types = None

    def generate_fn(prompt: str) -> dict:
        last = None
        config = (_genai_types.GenerateContentConfig(temperature=temperature)
                  if _genai_types is not None else None)
        for model in models:
            delay = base_delay
            for attempt in range(retries):
                try:
                    resp = client.models.generate_content(
                        model=model, contents=prompt, config=config)
                    return {"text": (resp.text or "").strip(), "error": None,
                            "model_used": model}
                except Exception as e:
                    msg = str(e)
                    last = msg[:200]
                    transient = any(k in msg for k in
                                    ("503", "UNAVAILABLE", "429",
                                     "RESOURCE_EXHAUSTED", "500"))
                    if transient and attempt < retries - 1:
                        time.sleep(delay)
                        delay *= 2
                        continue
                    break      # move on to the next model
        return {"text": "", "error": last or "all models failed",
                "model_used": None}

    return generate_fn
