"""Readability agent: the last step in the pipeline.

Task #8. Owner: Edwin (Joao).

Job
---
Take the verified draft and rewrite it to an 8th-grade reading level WITHOUT changing or
dropping any medical fact. It is the last agent in the chain, so whatever it emits is what
a patient reads.

Why this agent is risky, and what that means for its design
-----------------------------------------------------------
Every other agent in the pipeline either adds information or checks it. This one only
rewrites, and it runs AFTER the Verifier. So a fact it drops or distorts is not caught by
anything downstream. The whole pipeline's faithfulness guarantee can be undone in this
last step.

Two consequences, both built into this module:

1. It is **bounded**. At most `max_passes` rewrites. Each pass is a chance to lose a fact,
   so we stop at the target or at the cap, whichever comes first. We do not loop until the
   grade number looks good.

2. It is **self-checking**. After every pass, numbers and negations are compared against
   the input. If a rewrite loses a number or flips a "do not" into a "do", that pass is
   REJECTED and the previous text is kept. A slightly harder sentence that is correct
   beats an easy sentence that is wrong.

Readability targeting is also genuinely approximate. Flesch-Kincaid counts syllables and
sentence length; it does not know "myocardial infarction" is harder than "heart attack"
for a patient. So we report jargon density alongside the grade, and treat a grade number
on its own with suspicion.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

import textstat

TARGET_GRADE = 8.0

PROMPT = """Rewrite the TEXT so a patient can read it easily.

Target: Flesch-Kincaid grade {target} or below. Current grade: {current}.

How to make it simpler:
- Shorter sentences. Split any sentence longer than about 15 words.
- Everyday words instead of medical words, where an everyday word exists.
- Active voice. "Your doctor will check" not "it will be checked".

What you must NOT change:
- Every number stays exactly as written: doses, percentages, dates, how many times a day.
- Every "not", "do not", "avoid", "stop" stays. Never flip a negative to a positive.
- Every medical fact stays. Do not drop anything to make the text shorter.
- Do not add any fact that is not already in the TEXT.
- If a medical term has no everyday equivalent, keep the term. Do not guess at a synonym.

Return only the rewritten text. No preamble, no notes.

TEXT:
{text}

Rewritten:"""

_NUM = re.compile(r"\b\d+(?:\.\d+)?\s?%?")
_NEG = re.compile(r"\b(?:not|never|avoid|stop|don't|do not|cannot|without|no)\b", re.I)


def measure(text: str) -> dict:
    """Readability metrics. Grade alone is not enough, so jargon density comes too."""
    words = max(textstat.lexicon_count(text, removepunct=True), 1)
    sentences = max(textstat.sentence_count(text), 1)
    return {
        "fk_grade": round(textstat.flesch_kincaid_grade(text), 2),
        "smog": round(textstat.smog_index(text), 2),
        "jargon_density": round(textstat.difficult_words(text) / words, 4),
        "n_words": words,
        "avg_sentence_words": round(words / sentences, 1),
    }


def meets_target(text: str, target: float = TARGET_GRADE) -> bool:
    return textstat.flesch_kincaid_grade(text) <= target


def fact_check(before: str, after: str) -> dict:
    """Did the rewrite lose or distort anything?

    Deterministic and narrow on purpose. It cannot catch every distortion, but it catches
    the two that hurt a patient most: a changed number, and a dropped negation. "Take 500
    mg" becoming "take 800 mg", or "do not drive" becoming "drive", are both caught here.
    """
    nums_before = sorted(n.strip().rstrip("%").strip() for n in _NUM.findall(before))
    nums_after = sorted(n.strip().rstrip("%").strip() for n in _NUM.findall(after))
    dropped = [n for n in nums_before if n not in nums_after]
    invented = [n for n in nums_after if n not in nums_before]

    neg_before = len(_NEG.findall(before))
    neg_after = len(_NEG.findall(after))

    # A big drop in length CAN mean content was cut. But plain-language rewriting
    # legitimately shrinks text a lot: "Subsequent to the administration of the
    # aforementioned pharmacological agent" (9 words) becomes "After you take this
    # medicine" (5). Measured on real simplifications, a good rewrite routinely drops
    # 40%. So the threshold is deliberately loose and is a last-resort signal only;
    # numbers and negations are the checks that actually protect the patient.
    wb, wa = len(before.split()), len(after.split())
    shrink = 1 - (wa / wb) if wb else 0.0

    problems = []
    if dropped:
        problems.append(f"numbers dropped: {dropped}")
    if invented:
        problems.append(f"numbers invented: {invented}")
    if neg_after < neg_before:
        problems.append(f"negations lost: {neg_before} -> {neg_after}")
    if shrink > 0.65:
        problems.append(f"text shrank {shrink:.0%}, likely dropped content")

    return {"ok": not problems, "problems": problems,
            "shrink_pct": round(shrink * 100, 1)}


@dataclass
class ReadabilityResult:
    original: str
    final: str
    passes_used: int = 0
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)
    met_target: bool = False
    rejected_passes: list = field(default_factory=list)
    error: str | None = None

    @property
    def grade_drop(self) -> float:
        return round(self.before.get("fk_grade", 0) - self.after.get("fk_grade", 0), 2)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["grade_drop"] = self.grade_drop
        return d


def run(text: str, generate_fn, target: float = TARGET_GRADE,
        max_passes: int = 2) -> ReadabilityResult:
    """Rewrite toward the target grade, rejecting any pass that breaks a fact.

    generate_fn(prompt) -> {"text": str, "error": str|None}, so this stays model-agnostic
    and testable without an API key.
    """
    res = ReadabilityResult(original=text, final=text, before=measure(text))

    for i in range(max_passes):
        current = measure(res.final)
        if current["fk_grade"] <= target:
            break

        out = generate_fn(PROMPT.format(target=target,
                                        current=current["fk_grade"],
                                        text=res.final))
        if out.get("error"):
            res.error = out["error"]
            break

        candidate = (out.get("text") or "").strip()
        if not candidate:
            res.error = "empty response"
            break

        check = fact_check(res.final, candidate)
        if not check["ok"]:
            # Keep the harder but correct text. This is the point of the agent.
            res.rejected_passes.append({"pass": i + 1, "problems": check["problems"]})
            break

        res.final = candidate
        res.passes_used = i + 1

    res.after = measure(res.final)
    res.met_target = res.after["fk_grade"] <= target
    return res


# --------------------------------------------------------------- model wiring

# Model availability has moved twice during this project: gemini-2.0-flash and
# gemini-2.5-flash are retired (404), and gemini-3.6-flash returned 503 on long prompts
# on 2026-10-13 while answering short ones fine. So the model is a parameter with a
# fallback chain, not a constant. See builder notes for the measurements.
DEFAULT_MODELS = ("gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite")


def make_generate_fn(client, models=DEFAULT_MODELS, temperature: float = 0.0,
                     retries: int = 3, base_delay: float = 5.0):
    """Build a generate_fn for run().

    Retries transient failures with backoff, then falls back to the next model. Retrying
    alone was not enough on 2026-10-13: 3.6-flash 503'd on every long prompt while
    3.5-flash answered the same prompt immediately.

    Returns the usual {"text": str, "error": str|None}, so run() stays model-agnostic.
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
        return {"text": "", "error": last or "all models failed"}

    return generate_fn
