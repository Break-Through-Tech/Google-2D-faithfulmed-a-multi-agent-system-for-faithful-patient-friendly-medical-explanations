"""FinalCheck agent: the last gate before a patient reads the text.

Owner: Edwin (Joao).

Job
---
Readability rewrites the text AFTER the Verifier has approved it. So the text a patient
actually reads has not been checked by anyone. FinalCheck checks it against the SOURCE.

    final text -> check -> pass ------------------------------> done
                        -> fail -> fix -> check again  (at most max_rounds times)
                                                    -> still failing -> needs_review

Design rules this module encodes
--------------------------------
1. **Two checks, merged.** Deterministic guards run first and cost nothing:
   - every number in the extracted facts must still be in the final text
   - no number may appear in the final text that is not in the SOURCE
   - a negated finding ("no chest pain") must not be stated without its negation
   Then the LLM judge reads the SOURCE and the final text for everything a regex cannot
   see: reworded facts, dropped sentences, invented advice. Findings from both are merged.

2. **`faithful` is recomputed from the problems found.** It is true only when
   unsupported_claims, omissions and negation_errors are all empty after the guards are
   added. A model cannot write "faithful: true" while listing a problem, or pass a changed
   dose the guards caught. A "faithful: false" that names no problem is not a fail, because
   there is nothing to fix; it is counted as `unexplained_fail`. Same rule as the Verifier.

3. **A failed fix is a fixed point, not a loop.** At most `max_rounds` fixes, and every
   fix is checked again. The run always ends on a check, never on an unchecked fix.

4. **A fix that makes things worse is not shipped.** Every checked version is scored by
   its number of problems. If nothing passes, the version with the fewest problems is
   returned, flagged `needs_review`.

5. **If the check itself cannot run, the text is not passed.** An API error or an
   unreadable verdict means we do not know if the text is faithful. The safe answer is
   `needs_review`, never a pass. No fix is attempted, because there is no verdict to fix.

The model call sits behind `generate_fn(prompt) -> {"text", "error", "model_used"}`, the
same contract as readability_agent.py and verifier_agent.py, so tests need no API key.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

import textstat

TARGET_GRADE = 8.0
MAX_ROUNDS = 2

# Same wording as the Verifier, so the final text is held to the same standard as the
# draft. Kept in sync with notebooks/adk_pipeline.py `_verifier_instruction`.
CHECK_PROMPT = """You are checking a patient-friendly explanation for faithfulness.

SOURCE (the ground truth):
{source}

EXTRACTED FACTS (a checklist, may be incomplete):
{atoms}

RETRIEVED DEFINITIONS (allowed background, not from the SOURCE):
{definitions}

EXPLANATION TO CHECK:
{text}

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

FIX_PROMPT = """Fix the final patient explanation using the verdict. Change nothing else.

SOURCE:
{source}

EXPLANATION:
{text}

VERDICT:
{verdict}

Rules:
- Remove every unsupported claim.
- Add back every omission, worded plainly, using only the SOURCE.
- Fix every negation error.
- Keep it easy to read: short sentences, everyday words.
- Do not add explanations, definitions or background that are not in the SOURCE.
- Do not rewrite sentences the verdict did not flag.
- Return only the explanation text."""

_FAIL_KEYS = ("unsupported_claims", "omissions", "negation_errors")
_LIST_KEYS = _FAIL_KEYS + ("unexplained_terms",)

# ------------------------------------------------------------------ deterministic guards

_NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
_NEG = re.compile(r"\b(?:not|never|avoid|stop|don't|do not|cannot|without|no|denies|"
                  r"negative for|absence of|ruled out)\b", re.I)
# Readability may legitimately write "twice" as "2 times", or "2" as "two".
_WORDS = {"once": "1", "one": "1", "twice": "2", "two": "2", "three": "3", "thrice": "3",
          "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
          "ten": "10", "eleven": "11", "twelve": "12", "half": "0.5"}


def _numbers(text: str) -> list[str]:
    out = []
    for n in _NUM.findall(text or ""):
        n = n.replace(",", "").rstrip(".")
        if "." in n:
            n = n.rstrip("0").rstrip(".") or "0"
        out.append(n)
    return out


def _number_set(text: str) -> set[str]:
    """Digits in the text, plus the digits its number words stand for."""
    words = {_WORDS[w] for w in re.findall(r"[a-z]+", (text or "").lower()) if w in _WORDS}
    return set(_numbers(text)) | words


def check_numbers(source: str, atoms: list[dict], text: str) -> dict:
    """Dropped numbers are omissions. Invented numbers are unsupported claims.

    "Dropped" is measured against the extracted facts when there are any, because they are
    the clinically meaningful numbers. Without facts it falls back to the whole SOURCE.
    "Invented" is always measured against the SOURCE, which is the ground truth.
    """
    have = _number_set(text)
    omissions = []
    if atoms:
        for a in atoms:
            fact = (a.get("text") or "").strip()
            missing = [n for n in _numbers(fact) if n not in have]
            if missing:
                omissions.append(f"{fact} (number missing from final text: {missing[0]})")
    else:
        for n in dict.fromkeys(_numbers(source)):
            if n not in have:
                omissions.append(f"number {n} from the source is missing from the final text")

    allowed = _number_set(source)
    invented = [n for n in dict.fromkeys(_numbers(text)) if n not in allowed]
    unsupported = [f"number {n} is not in the source" for n in invented]
    return {"omissions": omissions, "unsupported_claims": unsupported}


def check_negations(atoms: list[dict], text: str) -> list[str]:
    """A negated finding must not be stated as present.

    For every fact marked `negated`, if its term appears in the final text, a negation word
    must appear within ~8 words before it. "No acute change" becoming "acute change" is the
    failure this exists for. A false positive costs one fix round. A false negative tells a
    patient they have something they do not have.
    """
    errors = []
    low = (text or "").lower()
    for a in atoms or []:
        if not a.get("negated"):
            continue
        anchor = (a.get("term") or a.get("text") or "").strip().lower()
        if not anchor or anchor not in low:
            continue
        near = re.compile(r"(?:" + _NEG.pattern + r")\W+(?:[\w'-]+\W+){0,8}?"
                          + re.escape(anchor), re.I)
        if not near.search(text):
            errors.append(f"'{anchor}' is stated without its negation")
    return errors


def deterministic_check(source: str, atoms: list[dict], text: str,
                        target_grade: float = TARGET_GRADE) -> dict:
    """Everything FinalCheck can decide without an API call."""
    nums = check_numbers(source, atoms, text)
    grade = round(textstat.flesch_kincaid_grade(text or ""), 2)
    return {
        "unsupported_claims": nums["unsupported_claims"],
        "omissions": nums["omissions"],
        "negation_errors": check_negations(atoms, text),
        "fk_grade": grade,
        "reading_level_ok": grade <= target_grade,
    }


# ------------------------------------------------------------------ verdicts

def parse_verdict(raw) -> dict | None:
    """Pull a JSON verdict out of the model's reply. None if there is nothing usable."""
    if isinstance(raw, dict):
        obj = raw
    else:
        text = (raw or "").strip()
        if not text:
            return None
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
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None

    def as_list(v):
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str) and v.strip():
            return [v.strip()]
        return []

    out: dict = {k: as_list(obj.get(k)) for k in _LIST_KEYS}
    out["faithful"] = obj.get("faithful") is True
    out["reading_level_ok"] = obj.get("reading_level_ok") is True
    return out


def merge(llm: dict | None, guards: dict) -> dict:
    """Union the LLM's findings with the guards' findings and recompute `faithful`."""
    llm = llm or {}

    def dedup(items):
        seen, out = set(), []
        for x in items:
            k = x.strip().lower()
            if k and k not in seen:
                seen.add(k)
                out.append(x.strip())
        return out

    v: dict = {k: dedup(list(llm.get(k, [])) + list(guards.get(k, []))) for k in _FAIL_KEYS}
    v["unexplained_terms"] = dedup(list(llm.get("unexplained_terms", [])))
    # A fail must name a problem, or there is nothing for the fix step to act on.
    # Measured 2026-10-26: 7 of 28 real verdicts said "faithful: false" with every
    # failure list empty (the model was counting unexplained terms as failures). Treating
    # those as fails made the fix step invent definitions to "fix" them. So `faithful` is
    # derived from the merged lists only, the same rule as the Verifier. The disagreement
    # is kept in `unexplained_fail` so it can be counted.
    v["faithful"] = not any(v[k] for k in _FAIL_KEYS)
    v["unexplained_fail"] = llm.get("faithful") is not True and v["faithful"]
    # textstat is cheap and the LLM's own guess at reading level is unreliable.
    v["reading_level_ok"] = bool(guards.get("reading_level_ok"))
    return v


def problem_count(verdict: dict) -> int:
    return sum(len(verdict.get(k, [])) for k in _FAIL_KEYS)


# ------------------------------------------------------------------ faithfulness report

def faithfulness_report(verdict: dict | None, fixes: int = 0, fixed: list | None = None,
                        text: str = "") -> dict:
    """The report that goes to the user next to the explanation (see the v2 diagram:
    "patient-friendly explanation plus a faithfulness report, flagged if it still fails").

    Plain words, no model call. `fixed` lists problems found earlier that the shipped
    text no longer has.
    """
    v = verdict or {}
    remaining = [x for k in _FAIL_KEYS for x in v.get(k, [])]
    if not verdict:   # the check could not run, which is never a pass
        status, summary = "needs_review", ("We could not check this explanation against "
                                           "the original. A person should review it.")
    elif v.get("faithful"):
        status = "fixed" if fixes else "passed"
        summary = ("Checked against the original text. Nothing was added, changed or "
                   "left out." if not fixes else
                   f"Checked against the original text. Problems were found, the text "
                   f"was corrected ({fixes} time(s)), and the corrected text passed.")
    else:
        status = "needs_review"
        summary = (f"Checked against the original text. {len(remaining)} problem(s) are "
                   f"still there after {fixes} fix attempt(s). A person should review it.")
    grade = round(textstat.flesch_kincaid_grade(text), 1) if text else None
    return {
        "status": status,
        "summary": summary,
        "fixed": list(fixed or []),
        "remaining": remaining,
        "terms_not_explained": list(v.get("unexplained_terms", [])),
        # Flesch-Kincaid: a US school grade, not a score. Lower is easier.
        "reading_grade": grade,
        "reading_level": reading_level_text(grade),
    }


def reading_level_text(grade: float | None, target: float = TARGET_GRADE) -> str:
    """Say what the grade number means, so nobody reads 7.6 as a score out of 10."""
    if grade is None:
        return ""
    verdict = "meets" if grade <= target else "above"
    return (f"School grade {grade:g} (a typical grade {max(1, round(grade)):g} student can "
            f"read it). Goal: grade {target:g} or lower, {verdict} the goal.")


# ------------------------------------------------------------------ the agent

@dataclass
class FinalCheckResult:
    source: str
    original: str                 # the text FinalCheck received (Readability's output)
    final: str                    # the text to show the patient
    passed: bool = False
    needs_review: bool = True
    fix_rounds: int = 0           # how many fixes were attempted
    llm_calls: int = 0
    verdict: dict = field(default_factory=dict)       # verdict on `final`
    history: list = field(default_factory=list)       # every check, in order
    reason: str = ""              # why the run ended
    error: str | None = None
    models_used: list = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.final.strip() != self.original.strip()

    def report(self) -> dict:
        """The faithfulness report shown to the user next to `final`."""
        first = (self.history[0]["verdict"] if self.history else None) or {}
        now = {x.strip().lower() for k in _FAIL_KEYS for x in self.verdict.get(k, [])}
        fixed = [x for k in _FAIL_KEYS for x in first.get(k, [])
                 if self.changed and x.strip().lower() not in now]
        return faithfulness_report(self.verdict or None, self.fix_rounds, fixed, self.final)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["changed"] = self.changed
        d["report"] = self.report()
        return d


def check(source: str, text: str, generate_fn, atoms: list[dict] | None = None,
          definitions: str = "", target_grade: float = TARGET_GRADE) -> dict:
    """One check of one text. Returns {"verdict", "guards", "error", "model_used"}.

    On an API error or an unreadable reply, `verdict` is None. The caller must treat that
    as "unknown", which is not a pass.
    """
    atoms = atoms or []
    guards = deterministic_check(source, atoms, text, target_grade)
    out = generate_fn(CHECK_PROMPT.format(
        source=source, atoms=json.dumps(atoms, ensure_ascii=False),
        definitions=definitions or "(none provided)", text=text))
    llm = parse_verdict(out.get("text")) if not out.get("error") else None
    error = out.get("error") or (None if llm is not None else "unreadable verdict")
    return {"verdict": merge(llm, guards) if llm is not None else None,
            "guards": guards, "error": error, "model_used": out.get("model_used")}


def run(source: str, text: str, generate_fn, atoms: list[dict] | None = None,
        definitions: str = "", max_rounds: int = MAX_ROUNDS,
        target_grade: float = TARGET_GRADE) -> FinalCheckResult:
    """Check the final text, fix it at most `max_rounds` times, flag it if it still fails.

    `generate_fn(prompt) -> {"text": str, "error": str|None, "model_used": str|None}`.
    Worst case cost: 1 + 2 * max_rounds model calls (5 with the default cap).
    """
    res = FinalCheckResult(source=source, original=text, final=text)
    best = None                        # (score, round, text, verdict)
    current = text

    for rnd in range(max_rounds + 1):
        c = check(source, current, generate_fn, atoms, definitions, target_grade)
        res.llm_calls += 1
        if c["model_used"]:
            res.models_used.append(c["model_used"])
        res.history.append({"round": rnd, "text": current, "verdict": c["verdict"],
                            "guards": c["guards"], "error": c["error"]})

        if c["verdict"] is None:
            res.error = c["error"]
            res.reason = "check could not run"
            break

        # A passing version always wins; otherwise the one with the fewest problems.
        score = (not c["verdict"]["faithful"], problem_count(c["verdict"]))
        if best is None or score < best[0]:
            best = (score, rnd, current, c["verdict"])

        if c["verdict"]["faithful"]:
            res.reason = "passed" if rnd == 0 else f"passed after {rnd} fix(es)"
            break
        if rnd == max_rounds:
            res.reason = f"still failing after {max_rounds} fixes"
            break

        # Only the problems go to the fix step. Unexplained terms are a readability gap,
        # and showing them to the fixer made it invent definitions.
        problems = {k: c["verdict"][k] for k in _FAIL_KEYS if c["verdict"][k]}
        fix = generate_fn(FIX_PROMPT.format(
            source=source, text=current,
            verdict=json.dumps(problems, ensure_ascii=False, indent=1)))
        res.llm_calls += 1
        if fix.get("model_used"):
            res.models_used.append(fix["model_used"])
        candidate = (fix.get("text") or "").strip()
        res.fix_rounds += 1
        if fix.get("error") or not candidate:
            res.error = fix.get("error") or "empty fix"
            res.reason = "fix could not run"
            break
        current = candidate

    if best is not None:
        _, _, res.final, res.verdict = best
        res.passed = res.verdict["faithful"]
    else:
        res.final, res.verdict, res.passed = text, {}, False
    res.needs_review = not res.passed
    return res


# ------------------------------------------------------------------ model wiring

# Same chain as readability_agent.py and verifier_agent.py, so every agent reports under
# the same models. gemini-2.0/2.5-flash are retired; 3.6-flash 503s on long prompts.
DEFAULT_MODELS = ("gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite")


def make_generate_fn(client, models=DEFAULT_MODELS, temperature: float = 0.0,
                     retries: int = 3, base_delay: float = 5.0):
    """generate_fn for run(): retries transient errors with backoff, then the next model.

    A 429 (daily quota) is not retried on the same model: it will not clear today.
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
                except Exception as e:  # noqa: BLE001, any API failure moves on
                    msg = str(e)
                    last = f"{model}: {msg[:200]}"
                    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                        break
                    transient = any(k in msg for k in ("503", "UNAVAILABLE", "500"))
                    if transient and attempt < retries - 1:
                        time.sleep(delay)
                        delay *= 2
                        continue
                    break
        return {"text": "", "error": last or "all models failed", "model_used": None}

    return generate_fn
