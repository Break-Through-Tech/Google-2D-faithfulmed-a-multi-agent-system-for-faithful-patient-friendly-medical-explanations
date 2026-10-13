"""Verifier agent: the faithfulness check between Simplifier and Refiner.

Task #6, October milestone. Owner: Isha.

Job
---
Read the SOURCE, the extracted ATOMS, the retrieved DEFINITIONS, and the draft
EXPLANATION. Decide whether the explanation is faithful to the source and report
every claim that is not. This verdict is what the Refiner acts on, and it is the
output calibrated against MedAESQA's human labels.

Why this is a standalone module and not just the LlmAgent in adk_pipeline.py
--------------------------------------------------------------------------
Two reasons, both the same shape as src/readability_agent.py.

1. The LLM verdict is not trusted on its own for the parts a deterministic check
   can catch. A changed dose and a dropped negation are patient-safety failures,
   and asking a language model to notice them is reliable most of the time, not
   all of the time. So this module runs number and negation guards first and
   merges their findings INTO the verdict. The LLM adds judgment on wording and
   omissions; the guards add floor-level safety.

2. The LLM output needs to survive bad JSON, truncated responses, and 503s on
   long prompts. The ADK LlmAgent assumes a well-formed reply; the evaluation
   harness cannot. parse_verdict() recovers what it can and the run() path falls
   back to a deterministic-only verdict when the model fails outright.

What the verdict looks like
---------------------------
Shape is pinned in docs/architecture.md:

    {
      "faithful": bool,
      "unsupported_claims": [str, ...],   # supported by NEITHER source NOR definitions
      "omissions": [str, ...],            # facts in SOURCE missing from the draft
      "negation_errors": [str, ...],      # a "no X" rendered as "X", or vice versa
      "unexplained_terms": [str, ...],    # faithful but unreadable; readability gap
      "reading_level_ok": bool,
    }

Design rules this module encodes
--------------------------------
- A term copied verbatim from the source is faithful. It lands in `unexplained_terms`,
  never in `unsupported_claims`. This is the v2 fix for the contradiction documented
  in architecture.md section "Open problems in this design", item 1.
- A negation error IS an unsupported claim: "do not drive" rendered as "drive" is
  not a wording problem, it is a wrong fact. The verdict surfaces it in its own
  bucket so the Refiner can see what to fix, but it also counts toward `faithful`.
- `faithful` is a derived field: true only when `unsupported_claims`, `omissions`,
  and `negation_errors` are all empty. It is recomputed here regardless of what
  the LLM wrote, so the model cannot say "faithful: true" while listing failures.
- `reading_level_ok` is deterministic (textstat FK grade <= 8.0). The LLM's
  self-assessment is unreliable on this and the measurement is cheap.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict

import textstat

TARGET_GRADE = 8.0

PROMPT = """You are checking a patient-friendly explanation for faithfulness.

SOURCE (the ground truth):
{source}

EXTRACTED FACTS (a checklist, may be incomplete):
{atoms}

RETRIEVED DEFINITIONS (allowed background, not from the SOURCE):
{definitions}

EXPLANATION TO CHECK:
{draft}

Return a JSON object with these keys:
  faithful: true only if unsupported_claims and omissions are both empty
  unsupported_claims: statements in the explanation that are supported by NEITHER \
the SOURCE NOR the retrieved definitions
  omissions: facts in the SOURCE that the explanation leaves out
  negation_errors: findings whose negation was lost or flipped
  unexplained_terms: medical terms left without a plain explanation
  reading_level_ok: true if a typical 8th grader could read it

Important: a medical term copied from the SOURCE without explanation is NOT an \
unsupported claim. It is faithful. List it under unexplained_terms only.
Treat a negation error as an unsupported claim as well.
Return only the JSON object."""

_NUM = re.compile(r"\b\d+(?:\.\d+)?\s?%?")
_NEG = re.compile(r"\b(?:not|never|avoid|stop|don't|do not|cannot|without|no)\b", re.I)
_VERDICT_KEYS = ("faithful", "unsupported_claims", "omissions", "negation_errors",
                 "unexplained_terms", "reading_level_ok")


def _nums(text: str) -> list[str]:
    return [n.strip().rstrip("%").strip() for n in _NUM.findall(text or "")]


def _has_neg(text: str) -> bool:
    return bool(_NEG.search(text or ""))


def _atom_text(atom: dict) -> str:
    return (atom.get("text") or "").strip()


def check_negations(atoms: list[dict], draft: str) -> list[str]:
    """Flag atoms where the negation was dropped or inverted.

    For every atom with `negated: True`, the draft must either repeat the negated
    assertion ("no chest pain") or omit the finding entirely. If the draft asserts
    the finding WITHOUT a negation, meaning has been flipped. "No acute change"
    becoming "acute change" is the exact failure this guard exists for.

    The check uses the atom's `term` field as the anchor, falling back to `text`.
    It is approximate on purpose: a false positive wastes a Refiner round, which
    is cheap. A false negative lets a flipped finding reach the patient.
    """
    errors = []
    for a in atoms:
        if not a.get("negated"):
            continue
        anchor = (a.get("term") or _atom_text(a)).strip().lower()
        if not anchor or anchor not in draft.lower():
            continue
        # The anchor appears in the draft. Is a negation within ~8 words of it?
        pattern = re.compile(
            r"(?:" + _NEG.pattern + r")\W+(?:\w+\W+){0,8}" + re.escape(anchor),
            re.I,
        )
        if not pattern.search(draft):
            errors.append(f"'{anchor}' asserted without its negation")
    return errors


def check_numbers(atoms: list[dict], draft: str) -> list[str]:
    """Flag atoms whose numeric values are missing from the draft.

    A dose, a lab value, a frequency: these are the facts a patient cannot
    paraphrase around. If the atom says "500 mg twice daily" and the draft says
    "take this medicine", the dose has been dropped and the explanation is
    incomplete. The guard reports it as an omission.

    Percent signs and whitespace are normalized so "8.2%" and "8.2 %" match.
    """
    draft_nums = set(_nums(draft))
    missing = []
    for a in atoms:
        nums = _nums(_atom_text(a))
        for n in nums:
            if n not in draft_nums:
                missing.append(f"{_atom_text(a)} (missing number: {n})")
                break        # one report per atom, not one per missing digit
    return missing


def measure_reading_level(text: str) -> dict:
    """Readability snapshot. Same shape as readability_agent.measure()."""
    words = max(textstat.lexicon_count(text or "", removepunct=True), 1)
    return {
        "fk_grade": round(textstat.flesch_kincaid_grade(text or ""), 2),
        "smog": round(textstat.smog_index(text or ""), 2),
        "n_words": words,
    }


def deterministic_check(atoms: list[dict], draft: str,
                       target_grade: float = TARGET_GRADE) -> dict:
    """Everything we can decide without an API call."""
    reading = measure_reading_level(draft)
    return {
        "negation_errors": check_negations(atoms, draft),
        "omissions": check_numbers(atoms, draft),
        "reading_level_ok": reading["fk_grade"] <= target_grade,
        "reading_metrics": reading,
    }


def parse_verdict(raw: str) -> dict | None:
    """Pull a JSON verdict out of the model's reply.

    Tolerates ```json fences, surrounding prose, and keys the model forgot to
    include. Returns None only when nothing JSON-shaped is present; a verdict
    missing some keys is coerced to the full schema with empty defaults, since
    an incomplete verdict is still more useful than throwing it away.
    """
    if not raw:
        return None
    text = raw.strip()
    # strip ```json fences the model sometimes wraps its reply in
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        text = fenced.group(1)
    else:
        # find the outermost {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        text = text[start:end + 1]
    try:
        obj = json.loads(text)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return _coerce(obj)


def _coerce(obj: dict) -> dict:
    """Fill missing keys with safe defaults and coerce types to the schema."""
    def _as_list(v) -> list[str]:
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str) and v.strip():
            return [v.strip()]
        return []

    return {
        "faithful": bool(obj.get("faithful", False)),
        "unsupported_claims": _as_list(obj.get("unsupported_claims")),
        "omissions": _as_list(obj.get("omissions")),
        "negation_errors": _as_list(obj.get("negation_errors")),
        "unexplained_terms": _as_list(obj.get("unexplained_terms")),
        "reading_level_ok": bool(obj.get("reading_level_ok", False)),
    }


def _merge(llm: dict, deterministic: dict) -> dict:
    """Combine LLM judgment with deterministic findings.

    Union the two lists per bucket, de-duplicated. The deterministic signal is
    authoritative for reading_level_ok (textstat is cheap and the LLM is
    unreliable on this), and `faithful` is recomputed from the merged lists so
    the verdict is internally consistent.
    """
    def _dedup(xs: list[str]) -> list[str]:
        seen, out = set(), []
        for x in xs:
            k = x.strip().lower()
            if k and k not in seen:
                seen.add(k)
                out.append(x.strip())
        return out

    unsupported = _dedup(list(llm.get("unsupported_claims", [])))
    omissions = _dedup(list(llm.get("omissions", [])) + list(deterministic.get("omissions", [])))
    negations = _dedup(list(llm.get("negation_errors", [])) + list(deterministic.get("negation_errors", [])))
    unexplained = _dedup(list(llm.get("unexplained_terms", [])))

    faithful = not unsupported and not omissions and not negations

    return {
        "faithful": faithful,
        "unsupported_claims": unsupported,
        "omissions": omissions,
        "negation_errors": negations,
        "unexplained_terms": unexplained,
        "reading_level_ok": bool(deterministic.get("reading_level_ok", False)),
    }


_EMPTY_VERDICT = {
    "faithful": False, "unsupported_claims": [], "omissions": [],
    "negation_errors": [], "unexplained_terms": [], "reading_level_ok": False,
}


@dataclass
class VerifierResult:
    source: str
    draft: str
    verdict: dict = field(default_factory=lambda: dict(_EMPTY_VERDICT))
    llm_verdict: dict | None = None
    deterministic: dict = field(default_factory=dict)
    raw: str = ""
    model_used: str | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def run(source: str, atoms: list[dict], draft: str, generate_fn,
        definitions: str = "", target_grade: float = TARGET_GRADE) -> VerifierResult:
    """Verify a draft against its source. Returns a `VerifierResult`.

    `generate_fn(prompt) -> {"text": str, "error": str|None, "model_used"?: str}`.
    Keeping the LLM behind a callable means tests do not need an API key, and the
    agent is model-agnostic by construction.

    On an LLM error or an unparseable reply, the result is NOT empty: deterministic
    findings still populate the verdict, so a quota outage degrades to a weaker
    check rather than silently passing everything.
    """
    atoms = atoms or []
    deterministic = deterministic_check(atoms, draft, target_grade)

    out = generate_fn(PROMPT.format(source=source, atoms=json.dumps(atoms),
                                     definitions=definitions or "(none provided)",
                                     draft=draft))
    raw = (out.get("text") or "").strip()
    err = out.get("error")
    model_used = out.get("model_used")

    llm_verdict = parse_verdict(raw) if raw else None
    if llm_verdict is None:
        # Fall back to a deterministic-only verdict, flagged with an error so a
        # caller knows the LLM half did not contribute. The point of the fallback
        # is that `faithful` cannot be true when guards found something.
        fallback = _merge(dict(_EMPTY_VERDICT), deterministic)
        return VerifierResult(
            source=source, draft=draft,
            verdict=fallback, llm_verdict=None, deterministic=deterministic,
            raw=raw, model_used=model_used,
            error=err or ("unparseable verdict" if raw else "empty response"),
        )

    merged = _merge(llm_verdict, deterministic)
    return VerifierResult(
        source=source, draft=draft,
        verdict=merged, llm_verdict=llm_verdict, deterministic=deterministic,
        raw=raw, model_used=model_used, error=err,
    )


# --------------------------------------------------------------- model wiring

# Model availability has moved twice during this project: gemini-2.0-flash and
# gemini-2.5-flash are retired (404), and gemini-3.6-flash returned 503 on long prompts
# on 2026-10-13 while 3.5-flash handled the same prompts. So the model is a parameter
# with a fallback chain, not a constant. Same chain as src/readability_agent.py so the
# whole pipeline reports under the same models and the ablation stays like-for-like.
DEFAULT_MODELS = ("gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite")


def make_generate_fn(client, models=DEFAULT_MODELS, temperature: float = 0.0,
                     retries: int = 3, base_delay: float = 5.0):
    """Build a generate_fn for run(). Mirrors readability_agent.make_generate_fn.

    Retries transient failures with backoff, then falls back to the next model.
    Returns the usual {"text": str, "error": str|None, "model_used": str|None}.
    """
    import time
    from google.genai import types

    def generate_fn(prompt: str) -> dict:
        last = None
        for model in models:
            delay = base_delay
            for attempt in range(retries):
                try:
                    resp = client.models.generate_content(
                        model=model, contents=prompt,
                        config=types.GenerateContentConfig(temperature=temperature))
                    return {"text": (resp.text or "").strip(), "error": None,
                            "model_used": model}
                except Exception as e:
                    msg = str(e)
                    last = msg[:200]
                    transient = any(k in msg for k in
                                    ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500"))
                    if transient and attempt < retries - 1:
                        time.sleep(delay)
                        delay *= 2
                        continue
                    break      # move on to the next model
        return {"text": "", "error": last or "all models failed",
                "model_used": None}

    return generate_fn
