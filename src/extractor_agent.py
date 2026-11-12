"""Extractor agent: the first step of the pipeline. SOURCE in, typed clinical atoms out.

Task: Extractor, October milestone. Owner: Vera.

Job
---
Read the SOURCE and return every clinically meaningful fact as a list of atoms. Every
later step depends on this list: the Retriever looks up each atom's `term`, the
Simplifier must cover every atom, and the Verifier uses the atoms as its omission
checklist. A fact the Extractor drops is a fact the patient may never hear about. A fact
it invents is worse.

Why this is a standalone module and not just the LlmAgent in adk_pipeline.py
--------------------------------------------------------------------------
Same shape as src/verifier_agent.py and src/readability_agent.py: the LLM does the
extraction, plain Python checks it. The model is not trusted on the things a cheap
deterministic check can catch:

  * a number that is not in the SOURCE (changed dose, invented value)  -> rejected
  * an atom whose wording is mostly not in the SOURCE (invented fact)  -> rejected
  * a redaction placeholder such as XXXX treated as a value            -> rejected
  * a `term` that does not appear in the SOURCE                        -> cleared
  * a finding the SOURCE negates but the atom asserts, or vice versa   -> warning
  * an allergy section with no allergy atom                            -> warning
  * "NKDA" / "no known allergies" with no allergy atom                 -> atom added

The atom schema (a superset, so both downstream readers work)
-------------------------------------------------------------
docs/architecture.md (read by the Verifier and the pipeline Retriever) uses
`text, type, term, negated`. src/simplifier.py (Gisele / Renita branch) reads
`id, type, text, assertion, subject`. The Extractor emits both:

    {
      "id": "a3",
      "text": "allergic to penicillin",     # as close to the SOURCE wording as possible
      "type": "allergy",                     # see ATOM_TYPES
      "term": "penicillin",                  # the one word to look up, or ""
      "assertion": "present",                # present | negated | uncertain
      "negated": false,                      # DERIVED: assertion == "negated"
      "subject": "patient",                  # patient | family | other
      "section": "ALLERGIES"                 # MTSamples header it came from, or ""
    }

`negated` is always derived from `assertion` in code, so the two can never disagree.
`uncertain` covers hedged claims ("may prevent"), which are 34% of MedAESQA nuggets.

Types: the six from architecture.md, plus `symptom` and `other`, which the Simplifier
already accepts. MedAESQA answers are general medical claims ("lifestyle changes may
prevent sleep apnea") that fit none of the six; without `other` the model is forced to
mislabel them.

Source shapes
-------------
MedAESQA: prose with PubMed citation markers like [28646811, 33659106]. The markers are
stripped before the model sees the text, so PMIDs cannot leak in as "numbers".

MTSamples: dictated reports with headers written `HEADER:,  text.,NEXT HEADER: , text`.
Sections are not newline-delimited, so split_sections() finds the headers and the prompt
receives one section per line, tagged. A MEDICATIONS header is a stronger type signal
than anything the model can infer from prose.

Synthea: not handled here. Its rows are already typed atoms; it needs a flattener, not
an LLM (architecture.md, open question 3).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

ATOM_TYPES = ("diagnosis", "medication", "lab_value", "procedure", "instruction",
              "allergy", "symptom", "other")
ASSERTIONS = ("present", "negated", "uncertain")
SUBJECTS = ("patient", "family", "other")

# Loose model labels mapped onto the schema. Anything unmapped becomes `other`.
_TYPE_ALIASES = {
    "condition": "diagnosis", "disease": "diagnosis", "problem": "diagnosis",
    "finding": "diagnosis", "drug": "medication", "medicine": "medication",
    "treatment": "medication", "lab": "lab_value", "test": "lab_value",
    "test_result": "lab_value", "vital": "lab_value", "vital_sign": "lab_value",
    "surgery": "procedure", "plan": "instruction", "advice": "instruction",
    "recommendation": "instruction", "allergen": "allergy", "sign": "symptom",
}
_ASSERTION_ALIASES = {
    "positive": "present", "affirmed": "present", "true": "present",
    "confirmed": "present", "known": "present", "yes": "present",
    "absent": "negated", "negative": "negated", "denied": "negated",
    "ruled_out": "negated", "no": "negated",
    "possible": "uncertain", "hedged": "uncertain", "hypothetical": "uncertain",
    "probable": "uncertain", "maybe": "uncertain", "likely": "uncertain",
    "suspected": "uncertain", "potentially": "uncertain", "doubtful": "uncertain",
    "unclear": "uncertain",
}

# Which section headers imply which type. A hint for the model, not a hard rule:
# a MEDICATIONS section can still mention a diagnosis.
SECTION_TYPE_HINTS = {
    "MEDICATIONS": "medication", "CURRENT MEDICATIONS": "medication",
    "DISCHARGE MEDICATIONS": "medication", "ALLERGIES": "allergy",
    "PREOPERATIVE DIAGNOSIS": "diagnosis", "POSTOPERATIVE DIAGNOSIS": "diagnosis",
    "DIAGNOSIS": "diagnosis", "IMPRESSION": "diagnosis", "ASSESSMENT": "diagnosis",
    "PAST MEDICAL HISTORY": "diagnosis", "PROCEDURE": "procedure",
    "PROCEDURE PERFORMED": "procedure", "PROCEDURES": "procedure",
    "PAST SURGICAL HISTORY": "procedure", "PLAN": "instruction",
    "DISCHARGE INSTRUCTIONS": "instruction", "LABORATORY DATA": "lab_value",
    "LABS": "lab_value", "VITAL SIGNS": "lab_value", "FAMILY HISTORY": "diagnosis",
}

# The instruction text must start with "Extract every": src/test_pipeline_offline.py
# identifies the Extractor's prompt by that prefix.
PROMPT = """Extract every clinically meaningful fact from the SOURCE below as a JSON list.

Each item is a JSON object with exactly these keys:
  text: the fact, worded as close to the SOURCE as possible. One fact per item.
  type: one of {types}
  term: the single medical word or phrase a patient might not know, copied exactly as \
it appears in the SOURCE (for example metformin or HbA1c), or an empty string
  assertion: present, negated (the SOURCE says it is absent: "no fracture", "denies \
chest pain", "negative for"), or uncertain (hedged: "may", "possible", "suspected", \
"cannot rule out")
  subject: patient, family (a relative's history), or other
  section: the section header the fact came from, or an empty string

Rules:
- Include allergies as type allergy. Never drop them. "NKDA" is an allergy fact with \
assertion negated.
- Keep every number exactly as written: doses, units, lab values, dates, frequencies. \
Never round or convert.
- Keep hedges. "may prevent" is uncertain, not present.
- Use the section header as a strong hint for type (a fact under MEDICATIONS is \
usually a medication).
- Skip redaction placeholders such as XXXX. They are not values.
- Skip citation markers and reference numbers.
- Do not add any fact that is not in the SOURCE. Do not interpret or explain.
- Return only the JSON list, no other text.

SOURCE:
{source}"""

_CITATION = re.compile(r"\s*\[\s*\d{6,9}(?:\s*,\s*\d{6,9})*\s*\]")
_REDACTION = re.compile(r"\bX{3,}\b|\[\s*redacted\s*\]", re.I)
_NUM = re.compile(r"\d+(?:\.\d+)?")
# MTSamples header: ALL CAPS words then a colon, at the start or after a sentence /
# section break. At least 3 letters so "BP: 120/80" is not taken as a header.
_HEADER = re.compile(r"(?:^|(?<=[.,;])|(?<=\n))\s*([A-Z][A-Z0-9 /&()'\-]{2,60}?)\s*:\s*,?")
_NEG_CUE = re.compile(
    r"\b(?:no|not|denies|denied|deny|negative for|without|free of|ruled out|"
    r"absence of|absent|nor|never)\b", re.I)
_NKDA = re.compile(r"\bNKDA\b|\bno known (?:drug )?allergies\b", re.I)
_ALLERGY_MENTION = re.compile(r"\ballerg(?:y|ies|ic)\b", re.I)
_STOP = set("""a an and are as at be been but by for from had has have he her his in
into is it its of on or she that the their them they this to was were which while
who will with you your patient patients""".split())


# ------------------------------------------------------------------ source prep

def clean_source(text: str) -> str:
    """Strip PubMed citation markers. Leaves everything else as written."""
    return re.sub(r"[ \t]{2,}", " ", _CITATION.sub("", text or "")).strip()


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split an MTSamples-style report into (header, body) pairs.

    Headers are ALL CAPS followed by a colon, separated by commas rather than
    newlines (a dictation artifact). Text before the first header, or text with no
    headers at all (MedAESQA prose), comes back as a single ("", text) pair.
    """
    text = text or ""
    matches = [m for m in _HEADER.finditer(text) if sum(c.isalpha() for c in m.group(1)) >= 3]
    if not matches:
        return [("", text.strip())] if text.strip() else []
    out = []
    if text[:matches[0].start()].strip(" ,"):
        out.append(("", text[:matches[0].start()].strip(" ,")))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip(" ,.")
        if body:
            out.append((m.group(1).strip(), body))
    return out


def format_source_for_prompt(text: str) -> str:
    """What the model sees: cleaned text, one tagged section per line when headers exist."""
    sections = split_sections(clean_source(text))
    if len(sections) <= 1 and not (sections and sections[0][0]):
        return clean_source(text)
    lines = []
    for header, body in sections:
        hint = SECTION_TYPE_HINTS.get(header)
        tag = f"[{header}]" + (f" (usually {hint})" if hint else "") if header else ""
        lines.append(f"{tag} {body}".strip())
    return "\n".join(lines)


def build_prompt(source: str) -> str:
    return PROMPT.format(types=", ".join(ATOM_TYPES), source=format_source_for_prompt(source))


# ------------------------------------------------------------------ parsing

def parse_atoms(raw) -> list[dict] | None:
    """Pull a JSON list of atoms out of the model's reply.

    Tolerates ```json fences, prose around the list, and a dict wrapper such as
    {"atoms": [...]}. Returns None when nothing JSON-shaped is present, [] for a
    genuinely empty list.
    """
    if isinstance(raw, list):
        return [a for a in raw if isinstance(a, dict)]
    if isinstance(raw, dict):
        inner = raw.get("atoms") or raw.get("facts")
        return parse_atoms(inner) if isinstance(inner, list) else None
    text = str(raw or "").strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.M).strip()
    for pattern in (r"\[.*\]", r"\{.*\}"):
        m = re.search(pattern, text, flags=re.S)
        if not m:
            continue
        try:
            return parse_atoms(json.loads(m.group(0)))
        except json.JSONDecodeError:
            continue
    return None


# ------------------------------------------------------------------ guards

def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.'][a-z0-9]+)*", (text or "").lower())


def _stem(w: str) -> str:
    for suf in ("ies", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def grounding(atom_text: str, source: str) -> float:
    """Share of the atom's content words that appear in the SOURCE (0..1)."""
    src = {_stem(w) for w in _words(source)}
    content = [_stem(w) for w in _words(atom_text) if w not in _STOP and len(w) > 2
               and not w.isdigit()]
    if not content:
        return 1.0
    return sum(w in src for w in content) / len(content)


def numbers_not_in_source(atom_text: str, source: str) -> list[str]:
    src = set(_NUM.findall(source or ""))
    return [n for n in _NUM.findall(atom_text or "") if n not in src]


def _anchor(atom: dict) -> str:
    term = (atom.get("term") or "").strip()
    if term:
        return term
    # Keep stop words so multi-word phrases like "shortness of breath" match the source
    _neg_words = {"no", "not", "denies", "denied", "negative", "without"}
    content = [w for w in _words(atom.get("text", "")) if w not in _neg_words]
    return " ".join(content[:4])


def source_negates(anchor: str, source: str, window: int = 6) -> bool | None:
    """Does a negation cue appear shortly before `anchor` in the SOURCE?

    None when the anchor is not found, so the caller can tell "not negated" from
    "could not check".
    """
    if not anchor:
        return None
    hits = list(re.finditer(re.escape(anchor), source, flags=re.I))
    if not hits:
        return None
    for h in hits:
        # only look back within the same sentence / clause
        before = re.split(r"[.;:\n]|\bbut\b|\bhowever\b|\bwith\b", source[:h.start()],
                          flags=re.I)[-1]
        if _NEG_CUE.search(" ".join(before.split()[-window:])):
            return True
    return False


def _coerce_choice(value, allowed, aliases, default):
    v = str(value or "").strip().lower().replace(" ", "_")
    if v in allowed:
        return v
    return aliases.get(v, default)


def normalize_atoms(raw_atoms: list[dict], source: str,
                    min_grounding: float = 0.5) -> tuple[list[dict], list[dict], list[str]]:
    """Coerce, check and repair the model's atoms.

    Returns (atoms, rejected, warnings). `rejected` keeps each dropped atom with its
    reason, so an evaluation run can count them rather than losing them silently.
    """
    src = clean_source(source)
    atoms, rejected, warnings, seen = [], [], [], set()

    for a in raw_atoms or []:
        text = " ".join(str(a.get("text") or "").split())
        if not text:
            continue
        if _REDACTION.search(text):
            rejected.append({"atom": a, "reason": "redaction placeholder"})
            continue
        bad_nums = numbers_not_in_source(text, src)
        if bad_nums:
            rejected.append({"atom": a, "reason": f"number not in source: {', '.join(bad_nums)}"})
            continue
        g = grounding(text, src)
        if g < min_grounding:
            rejected.append({"atom": a, "reason": f"not grounded in source ({g:.0%} of words found)"})
            continue

        raw_type = str(a.get("type") or "").strip().lower().replace(" ", "_")
        atype = _coerce_choice(raw_type, ATOM_TYPES, _TYPE_ALIASES, "other")
        if raw_type and raw_type not in ATOM_TYPES and raw_type not in _TYPE_ALIASES:
            warnings.append(f"unknown type '{raw_type}' for '{text}', set to other")

        # assertion wins if given; otherwise fall back to the v2 `negated` flag
        if a.get("assertion") not in (None, ""):
            raw_assertion = str(a.get("assertion") or "").strip().lower().replace(" ", "_")
            assertion = _coerce_choice(a.get("assertion"), ASSERTIONS, _ASSERTION_ALIASES, "present")
            if raw_assertion and raw_assertion not in ASSERTIONS and raw_assertion not in _ASSERTION_ALIASES:
                warnings.append(f"unknown assertion '{raw_assertion}' for '{text}', set to present")
        else:
            assertion = "negated" if a.get("negated") is True else "present"
        subject = _coerce_choice(a.get("subject"), SUBJECTS, {"relative": "family"}, "patient")

        term = " ".join(str(a.get("term") or "").split())
        if term and term.lower() not in src.lower():
            warnings.append(f"term '{term}' not found in source, cleared")
            term = ""

        key = (text.lower(), assertion, subject)
        if key in seen:
            continue
        seen.add(key)

        atom = {"id": "", "text": text, "type": atype, "term": term,
                "assertion": assertion, "negated": assertion == "negated",
                "subject": subject, "section": str(a.get("section") or "").strip()}

        neg = source_negates(_anchor(atom), src)
        if neg is True and assertion == "present":
            warnings.append(f"source negates '{_anchor(atom)}' but atom '{text}' is present")
        elif neg is False and assertion == "negated" and not _NEG_CUE.search(src):
            warnings.append(f"atom '{text}' is negated but the source has no negation cue")
        atoms.append(atom)

    # NKDA is easy to drop and safety-relevant, so it is added deterministically.
    has_allergy = any(x["type"] == "allergy" for x in atoms)
    m = _NKDA.search(src)
    if m and not has_allergy:
        atoms.append({"id": "", "text": m.group(0), "type": "allergy", "term": "",
                      "assertion": "negated", "negated": True, "subject": "patient",
                      "section": "ALLERGIES" if "ALLERGIES" in src else "",
                      "added_by": "rule:nkda"})
        has_allergy = True
    if _ALLERGY_MENTION.search(src) and not has_allergy:
        warnings.append("source mentions allergies but no allergy atom was extracted")

    for i, x in enumerate(atoms, 1):
        x["id"] = f"a{i}"
    return atoms, rejected, warnings


# ------------------------------------------------------------------ run

@dataclass
class ExtractorResult:
    source: str
    atoms: list = field(default_factory=list)
    rejected: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    raw: str = ""
    model_used: str | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)

    def atoms_json(self) -> str:
        return json.dumps(self.atoms)


def run(source: str, generate_fn) -> ExtractorResult:
    """Extract atoms from one SOURCE. Returns an `ExtractorResult`.

    `generate_fn(prompt) -> {"text": str, "error": str|None, "model_used"?: str}`, the
    same contract as verifier_agent.run, so tests need no API key.

    On a model failure the result has no atoms and `error` set. There is no
    deterministic fallback for extraction: an empty list with an error is honest, a
    half-guessed list is not. The NKDA rule still runs, since it needs no model.
    """
    out = generate_fn(build_prompt(source))
    raw = (out.get("text") or "").strip()
    parsed = parse_atoms(raw)
    atoms, rejected, warnings = normalize_atoms(parsed or [], source)
    error = out.get("error")
    if parsed is None:
        error = error or ("unparseable reply" if raw else "empty response")
    return ExtractorResult(source=source, atoms=atoms, rejected=rejected,
                           warnings=warnings, raw=raw, model_used=out.get("model_used"),
                           error=error)


# ------------------------------------------------------------------ ADK hooks

def adk_instruction(ctx) -> str:
    """InstructionProvider for the ADK LlmAgent: builds the prompt from state['source'].

    A provider rather than a "{source}" template, because the source is reformatted
    (citations stripped, sections tagged) before the model sees it.
    """
    return build_prompt(ctx.state.get("source", ""))


def adk_after_extractor(callback_context):
    """after_agent_callback for the ADK Extractor: replace state['atoms'] with checked atoms.

    Runs the same guards as run(), so the pipeline and the standalone agent agree.
    Rejections and warnings are kept in state for the evaluation harness.
    """
    state = callback_context.state
    parsed = parse_atoms(state.get("atoms"))
    atoms, rejected, warnings = normalize_atoms(parsed or [], state.get("source", ""))
    state["atoms"] = json.dumps(atoms)
    state["extractor_report"] = json.dumps({
        "n_atoms": len(atoms), "rejected": rejected, "warnings": warnings,
        "parse_failed": parsed is None})
    return None


# --------------------------------------------------------------- model wiring

# Same fallback chain as verifier_agent / readability_agent, so the ablation reports
# every agent under the same models.
DEFAULT_MODELS = ("gemini-3.5-flash", "gemini-3.6-flash", "gemini-3.1-flash-lite")


def make_generate_fn(client, models=DEFAULT_MODELS, temperature: float = 0.0,
                     retries: int = 3, base_delay: float = 5.0):
    """Build a generate_fn for run(). Asks for JSON output; retries, then falls back."""
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
                        config=types.GenerateContentConfig(
                            temperature=temperature,
                            response_mime_type="application/json"))
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
                    break
        return {"text": "", "error": last or "all models failed", "model_used": None}

    return generate_fn


if __name__ == "__main__":
    import argparse
    import os
    import sys

    p = argparse.ArgumentParser(description="Run the Extractor on one source text.")
    p.add_argument("source", nargs="?", help="text file; reads stdin if omitted")
    p.add_argument("--show-prompt", action="store_true", help="print the prompt and exit (no API call)")
    args = p.parse_args()
    text = open(args.source).read() if args.source else sys.stdin.read()
    if args.show_prompt:
        print(build_prompt(text))
        sys.exit(0)
    from google import genai
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    result = run(text, make_generate_fn(client))
    print(json.dumps({k: v for k, v in result.as_dict().items() if k not in ("source", "raw")}, indent=2))
