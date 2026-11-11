"""
FaithfulMed - Simplifier agent (Task #5)

Takes the Extractor's atoms and rewrites them as a patient-friendly explanation, using plain-language
definitions from the MedlinePlus ChromaDB index (RAG) to explain medical terms.

Pipeline position:  Extractor -> SIMPLIFIER -> Verifier -> Refiner -> Readability

How it works (a fixed sequence, no free-form agent loops):
  1. Load atoms                        (only `text` and `type` are required)
  2. RAG: find glossary terms          (deterministic exact match of each atom's `term` and its
                                        wording against the glossary's titles + alternate names;
                                        no LLM, no API call)
  3. Gemini rewrites the atoms         (structured output: sentences, each citing atom ids and
                                        glossary topic ids)
  4. Plain-Python faithfulness checks  (does not trust the LLM):
       * every atom must be covered by some sentence            -> uncovered_atoms
       * every number in an atom must survive into its sentence -> missing_numbers
       * no new numbers may appear                              -> numbers_not_in_cited_atoms
       * negated / uncertain / family atoms must keep that meaning in the sentence (cue-word
         check on both the atom's field and its wording)
       * sentences must cite real atoms and real glossary entries

RAG modes (useful for ablations):
  "off"            no glossary
  "exact"          (default) only exact glossary matches are given to the model as definitions
  "exact+related"  also gives the nearest topic for atoms with no exact match, labelled RELATED
                   BACKGROUND. These can be near-misses (e.g. "metformin" -> "Diabetes Medicines"), so
                   this mode needs the Gemini embedding API and is riskier for faithfulness.
                   All leftover atoms are embedded in ONE request, not one request each.

Atoms may carry a `term` (the one word a patient might not know, per docs/architecture.md). When
absent it is derived from the atom's wording, so the lookup never queries a whole sentence.

Needs src/medlineplus_index.py and a built index (python src/medlineplus_index.py build).

Models:  a fallback chain, tried in order, because availability keeps moving and a model
         that hangs is not a 503. Every request has a deadline (REQUEST_TIMEOUT_S); on a
         timeout or a used-up daily quota the next model is tried straight away.

Setup:   pip install google-genai pandas numpy chromadb
         $env:GEMINI_API_KEY="..."
         optional: $env:FAITHFULMED_MODEL="m1,m2"   $env:FAITHFULMED_TIMEOUT="60"

Usage (repo root):
    python src/simplifier.py --demo --dry-run      # test the glossary lookup only; NO Gemini call
    python src/simplifier.py --demo                # full run on built-in sample atoms
    python src/simplifier.py atoms.json --out result.json     # atoms from the Extractor
    python src/simplifier.py --demo --rag off      # ablation: no glossary

From another agent / notebook:
    from simplifier import simplify
    result = simplify(atoms)            # atoms: list of dicts, the Extractor's result dict, or JSON text
    result["explanation"]               # the draft for the Verifier
    result["sentences"]                 # each sentence with atom_ids, glossary_topic_ids, flags
    result["checks"]                    # uncovered atoms, dropped numbers, clean True/False
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

# ----------------------------------------------------------------------------- config
# Model availability keeps moving: 2.0-flash and 2.5-flash are retired (404), 3.6-flash
# 503'd on long prompts on 2026-10-13, and on 2026-11-09 gemini-3.8-flash left the demo
# prompt unanswered for over five minutes. So this is a fallback chain, not a constant,
# and every request has a deadline. Same chain as readability_agent.DEFAULT_MODELS.
# Override with FAITHFULMED_MODEL (one model, or a comma-separated chain).
DEFAULT_MODELS = tuple(m.strip() for m in os.environ.get(
    "FAITHFULMED_MODEL", "gemini-3.5-flash,gemini-3.6-flash,gemini-3.1-flash-lite"
).split(",") if m.strip())
DEFAULT_MODEL = DEFAULT_MODELS[0]          # kept for callers that import the old name
REQUEST_TIMEOUT_S = float(os.environ.get("FAITHFULMED_TIMEOUT", "60"))
MAX_OUTPUT_TOKENS = 16384
RETRIES_PER_MODEL = 2   # low on purpose: the chain, not the retry count, handles a bad model
TARGET_GRADE = 8
RAG_MODES = ("off", "exact", "exact+related")
MAX_TERMS = 20          # glossary entries handed to the model per document
DEFINITION_CHARS = 700  # how much of each glossary summary the model sees
RELATED_TYPES = {"diagnosis", "symptom", "procedure", "lab_value", "other"}

# ----------------------------------------------------------------------------- prompt
SYSTEM_PROMPT = """You rewrite clinical facts into a patient-friendly explanation. FAITHFULNESS matters \
more than anything else: the explanation may contain only what the ATOMS say.

You receive:
- ATOMS: a JSON list. Each has id, type, text, assertion (present / negated / uncertain) and subject \
(patient / family / other).
- GLOSSARY (may be empty): plain-language definitions from MedlinePlus, each with a topic_id. An entry \
marked RELATED BACKGROUND may not define the exact term; use it only if it clearly matches.

You return sentences. For each sentence give:
- text: the sentence
- atom_ids: the ids of the atoms that sentence states
- glossary_topic_ids: the topic_ids of glossary entries you used to explain a term ([] if none)

Rules:
1. Use ONLY facts in the atoms. Never add a diagnosis, cause, risk, prognosis, treatment, advice, \
reassurance, or number that is not in an atom. Do not say things like "don't worry" or "this is common".
2. Cover EVERY atom in at least one sentence. Never drop an atom, especially medications, test \
results, and instructions.
3. Keep these exactly as written: drug names, doses, units, numbers, and time periods. Never round, \
convert, or change a number. Keep the original medical term in the sentence and put its everyday \
meaning next to it.
4. Keep the meaning. If an atom is negated, say it is NOT present ("You do not have...", "There was no..."). \
If an atom is uncertain, hedge ONCE ("may" or "might", never both "might" and "possible"). If an atom's \
subject is family, say it is about a family member, not you ("Your mother had...").
5. Explain hard terms. Assume the reader has no medical training. Every diagnosis, test name, lab name, \
or abbreviation that an 8th grader would not know must be explained in the same sentence, in a few \
everyday words, for example "type 2 diabetes (a condition where blood sugar stays too high)". An \
explanation may say ONLY what the term means in general. It must not add a cause, symptom, treatment, \
statistic, number, or any claim about this patient that is not in the atom. If a glossary entry matches \
the term, use its wording, shortened. If there is no entry, give a general meaning only if you are sure \
it is correct; if you are not sure, keep the term as written. Never say what a drug is for, because that \
is not in the atoms. Spell out abbreviations ("ED" becomes "emergency department").
6. Write to the patient as "you", at about a {grade}th-grade reading level: short sentences (under 20 \
words), common words ("hard exercise" instead of "strenuous exercise"), active voice, one idea per \
sentence. No bullet points.
7. Turn instruction atoms into clear "you should" sentences and keep their conditions ("until cleared"). \
Say "you take" for medications the patient is on, and use "you should" only when the atom is an instruction.
8. Never mention atoms, ids, or the glossary inside a sentence's text.
9. Do not say whether a result is normal, abnormal, high, or low unless an atom says so. Give the \
number and any range the atom gives, for example "Your troponin I was 0.02 ng/mL. The normal range \
is below 0.04."
10. Explain every medical word in a phrase too, including words like "acute" and "ST changes", in \
a few everyday words. Replace uncommon words with common ones ("hard exercise" instead of \
"strenuous exercise").
11. Write every sentence directly to the patient, including medications that were started \
("You were started on aspirin, 81 mg daily.").

Example
ATOMS: a1 (symptom, negated) "No fever"; a2 (diagnosis, uncertain) "Possible pneumonia"; a3 (medication) \
"Start amoxicillin 500 mg three times daily"; a4 (lab_value) "Troponin I 0.02 ng/mL (normal < 0.04)".
Sentences:
  "You do not have a fever."                                        atom_ids [a1]
  "Your doctor thinks you may have pneumonia, an infection in the lungs."   atom_ids [a2]
  "You should start taking amoxicillin, 500 mg, three times a day."  atom_ids [a3]
  "Your troponin I, a blood test that checks for heart muscle damage, was 0.02 ng/mL. A normal result is below 0.04."   atom_ids [a4]
"""

try:
    from pydantic import BaseModel

    class SentenceSchema(BaseModel):
        text: str
        atom_ids: list[str]
        glossary_topic_ids: list[int]

    class SimplifierSchema(BaseModel):
        sentences: list[SentenceSchema]
except ImportError:  # pragma: no cover
    SimplifierSchema = None

# ----------------------------------------------------------------------------- text helpers
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_NEG = re.compile(r"\b(no|not|never|none|without|neither|nor|negative|free of|ruled out)\b|n't", re.I)
_HEDGE = re.compile(
    r"\b(may|might|could|possible|possibly|probably|probable|likely|unsure|suspect(?:ed)?|seems?|"
    r"appears?|chance|not sure|not certain|unclear|think|thinks)\b", re.I)
_FAMILY = re.compile(
    r"\b(mother|mom|father|dad|sister|brother|parents?|family|relatives?|grandmother|grandfather|"
    r"grandparents?|aunt|uncle|cousin|son|daughter|child|children)\b", re.I)
_STOP = {"the", "and", "for", "with", "was", "were", "has", "had", "have", "are", "not", "but", "from",
         "that", "this", "his", "her", "she", "you", "your", "any", "all", "one", "two", "may", "can"}
# Words that do not change which condition an atom is about. A one-word glossary match (e.g. "stress")
# is only trusted when the atom is essentially just that word ("No fever"), never inside a longer
# phrase ("Stress test was negative...").
_FILLER = {"no", "not", "denies", "denied", "without", "possible", "possibly", "probable", "suspected",
           "history", "hx", "of", "the", "a", "an", "had", "has", "have", "with", "and", "in", "on", "patient",
           "mother", "father", "sister", "brother", "family", "was", "is", "are", "were", "likely", "mild",
           "moderate", "severe", "chronic", "acute"}
# Everything above, plus doses, units, frequencies and report verbs: words that are never
# the term being looked up. Used to turn an atom's wording into a lookup key when the
# Extractor supplied no `term`.
_TERM_NOISE = _FILLER | {
    "mg", "mcg", "ug", "g", "kg", "ml", "dl", "l", "ng", "mmol", "mol", "mm", "cm", "hg",
    "iu", "unit", "units", "tablet", "tablets", "capsule", "capsules", "dose", "doses",
    "daily", "twice", "once", "three", "four", "times", "time", "day", "days", "week",
    "weeks", "month", "months", "year", "years", "hour", "hours", "bid", "tid", "qid",
    "po", "prn", "every", "per", "for", "to", "from", "at", "by", "started", "start",
    "continue", "continued", "taking", "take", "showed", "show", "shows", "revealed",
    "reveals", "demonstrated", "noted", "found", "normal", "range", "negative",
    "positive", "result", "results", "value", "level", "levels"}


def _clean(v) -> str:
    return "" if v is None else str(v).strip()


def _is_true(v) -> bool:
    """A JSON boolean, or the string a model sometimes emits instead. "false" is not true."""
    return v is True or _clean(v).lower() in {"true", "yes", "1"}


def _strip(s: str) -> str:
    """lowercase words/numbers only, so 'Crohn's disease' and "crohn’s  disease" compare equal"""
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def _trim(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", _clean(text))
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    return cut[:end + 1] if end > 100 else cut.rstrip() + "..."


# ----------------------------------------------------------------------------- atoms
def load_atoms(atoms) -> list[dict]:
    """Accepts a list of atoms, the Extractor's result dict, or JSON text of either.
    Only `text` is truly needed; missing fields get safe defaults."""
    if isinstance(atoms, str):
        atoms = json.loads(atoms)
    if isinstance(atoms, dict):
        atoms = atoms.get("atoms", [])
    out, seen = [], set()
    for i, a in enumerate(atoms, start=1):
        if not isinstance(a, dict) or not _clean(a.get("text")):
            continue
        atom = {
            "id": _clean(a.get("id")) or f"a{i}",
            "type": (_clean(a.get("type")) or "other").lower(),
            "text": _clean(a.get("text")),
            "term": _clean(a.get("term")),
            # Two Extractor formats are in use: this file's `assertion`, and the ADK
            # pipeline's boolean `negated` (docs/architecture.md). Reading only one of
            # them loaded "No chest pain" as present, so a flipped rewrite passed clean.
            # An explicit `assertion` wins; `negated` is the fallback.
            "assertion": (_clean(a.get("assertion"))
                          or ("negated" if _is_true(a.get("negated")) else "present")).lower(),
            "subject": (_clean(a.get("subject")) or "patient").lower(),
            "flags": list(a.get("flags") or []),
        }
        if atom["id"] in seen:
            raise ValueError(f"Duplicate atom id: {atom['id']}")
        seen.add(atom["id"])
        out.append(atom)
    return out


# ----------------------------------------------------------------------------- RAG
def derive_term(text: str) -> str:
    """Best-effort lookup key for an atom whose Extractor gave no `term`.

        "Metformin 500 mg twice daily"              -> "metformin"
        "ECG showed no acute ST changes"            -> "ecg st changes"
        "Troponin I was 0.02 ng/mL (normal < 0.04)" -> "troponin i"

    docs/architecture.md: the lookup uses the term, not the sentence, because "metformin
    500 mg twice daily" matches nothing. Dropping the negation here is deliberate and
    safe: a glossary entry says what a term MEANS in general, and it is the atom's
    `assertion` - never the definition - that carries whether the finding is present.
    """
    kept = [t for t in _strip(text).split() if t not in _TERM_NOISE and not t.isdigit()]
    return " ".join(kept[:6])


def atom_term(atom: dict) -> str:
    """The text to look an atom up by: the Extractor's `term` when it gave one."""
    return _clean(atom.get("term")) or derive_term(atom.get("text", ""))


def _search_batch(retriever, queries: list[str]) -> list[list[dict]]:
    """Vector lookups for many queries in one embedding request where possible.

    medlineplus_index.Retriever.search_many does them in one; a custom or test retriever
    that only exposes .search still works, one query at a time.
    """
    if not queries:
        return []
    many = getattr(retriever, "search_many", None)
    if callable(many):
        return many(queries, k=1, use_synonyms=False)
    return [retriever.search(q, k=1, use_synonyms=False) for q in queries]


def spot_terms(text: str, stripped_to_surface: dict[str, str], max_n: int = 6) -> list[str]:
    """Greedy longest-match: which glossary names appear in this text? Returns glossary keys."""
    toks = _strip(text).split()
    content = [t for t in toks if t not in _FILLER and not t.isdigit()]
    single_ok = len(content) == 1          # one-word matches only when the atom is just that term
    found, i = [], 0
    while i < len(toks):
        for n in range(min(max_n, len(toks) - i), 0, -1):
            cand = " ".join(toks[i:i + n])
            usable = len(cand) >= 3 and not cand.isdigit() and (n > 1 or (single_ok and cand not in _STOP))
            if usable and cand in stripped_to_surface:
                found.append(stripped_to_surface[cand])
                i += n
                break
        else:
            i += 1
    return found


def _default_retriever():
    try:
        from medlineplus_index import Retriever
    except ImportError as e:
        raise RuntimeError(
            "Could not import medlineplus_index. Run from the repo root as "
            "'python src/simplifier.py', or add src/ to your Python path.") from e
    try:
        return Retriever()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Could not open the MedlinePlus index. Build it first with "
            "'python src/medlineplus_index.py build', or run with --rag off.") from e


def gather_definitions(atoms: list[dict], retriever, rag_mode: str, max_terms: int = MAX_TERMS) -> list[dict]:
    """Glossary entries to hand to the model. Exact matches never call the embedding API."""
    if rag_mode == "off":
        return []
    surfaces = getattr(retriever, "surface_to_id", None)
    if not surfaces:
        raise RuntimeError("The retriever must expose .surface_to_id (medlineplus_index.Retriever does).")
    stripped = {}
    for surf in surfaces:
        stripped.setdefault(_strip(surf), surf)

    entries: dict[int, dict] = {}

    def add(hit, kind, term, atom_id):
        tid = int(hit["topic_id"])
        e = entries.get(tid)
        if e is None:
            if len(entries) >= max_terms:
                return
            e = entries[tid] = {
                "topic_id": tid, "title": hit["title"], "url": hit["url"], "match_type": kind,
                "matched_terms": [], "atom_ids": [], "definition": _trim(hit["definition"], DEFINITION_CHARS)}
        if term not in e["matched_terms"]:
            e["matched_terms"].append(term)
        if atom_id not in e["atom_ids"]:
            e["atom_ids"].append(atom_id)

    has_exact = set()
    for a in atoms:
        # The Extractor's `term` is the most reliable key when it gave one: scanning the
        # wording alone misses a single-word term inside a longer atom ("asthma" in
        # "Patient has asthma and chest pain"), which spot_terms refuses on purpose.
        candidates = []
        term = _clean(a.get("term"))
        if term and _strip(term) in stripped:
            candidates.append(stripped[_strip(term)])
        candidates += spot_terms(a["text"], stripped)
        for surf in dict.fromkeys(candidates):
            hits = retriever.search(surf, k=1, use_synonyms=True)   # exact key -> no API call
            if hits and hits[0]["match_type"] == "synonym":
                add(hits[0], "exact", surf, a["id"])
                has_exact.add(a["id"])

    if rag_mode == "exact+related":
        # One embedding request for every leftover atom, not one per atom: the free tier
        # allows about 20/day, and a per-atom call is what drains a key mid-run.
        pending = [(a, atom_term(a)) for a in atoms
                   if a["id"] not in has_exact and a["type"] in RELATED_TYPES]
        pending = [(a, q) for a, q in pending if q]
        for (a, q), hits in zip(pending, _search_batch(retriever, [q for _, q in pending])):
            if hits:
                add(hits[0], "related", q, a["id"])
    return list(entries.values())


def format_glossary(glossary: list[dict]) -> str:
    if not glossary:
        return "(none)"
    lines = []
    for g in glossary:
        label = "EXACT" if g["match_type"] == "exact" else "RELATED BACKGROUND (may not define the exact term)"
        lines.append(f'- topic_id {g["topic_id"]} | {label} | matched "{"; ".join(g["matched_terms"])}" '
                     f'| title "{g["title"]}"\n  definition: {g["definition"]}')
    return "\n".join(lines)


# ----------------------------------------------------------------------------- prompt assembly
def prepare(atoms, *, retriever=None, rag_mode: str = "exact", grade: int = TARGET_GRADE,
            max_terms: int = MAX_TERMS):
    """Everything up to (not including) the LLM call. Handy for debugging the RAG step."""
    if rag_mode not in RAG_MODES:
        raise ValueError(f"rag_mode must be one of {RAG_MODES}")
    atoms = load_atoms(atoms)
    glossary = []
    if atoms and rag_mode != "off":
        glossary = gather_definitions(atoms, retriever or _default_retriever(), rag_mode, max_terms)
    system = SYSTEM_PROMPT.replace("{grade}", str(grade))
    llm_atoms = [{k: a[k] for k in ("id", "type", "text", "assertion", "subject")} for a in atoms]
    user = (f"ATOMS:\n{json.dumps(llm_atoms, ensure_ascii=False, indent=1)}\n\n"
            f"GLOSSARY:\n{format_glossary(glossary)}\n\nWrite the patient explanation.")
    return atoms, glossary, system, user


# ----------------------------------------------------------------------------- Gemini call
_client = None


def _is_timeout(e: Exception) -> bool:
    """True when a request ran out of time rather than failing outright.

    A hang is not a 503, so no retry rule caught it before: the run just stopped. The
    SDK surfaces it as an httpx timeout, sometimes wrapped, so check the chained cause
    and the message as well as the type.
    """
    names = [type(e).__name__]
    for inner in (e.__cause__, e.__context__):
        if inner is not None:
            names.append(type(inner).__name__)
    if any("timeout" in n.lower() for n in names):
        return True
    msg = str(e).lower()
    return any(k in msg for k in ("timed out", "timeout", "504", "deadline_exceeded"))


def _make_gemini_llm_fn(models=DEFAULT_MODELS, temperature: float | None = None,
                        timeout_s: float = REQUEST_TIMEOUT_S):
    """Build llm_fn(system, user) -> raw JSON, trying each model in the chain in turn.

    Every request carries a deadline, so a model that hangs costs `timeout_s`, not the
    whole run. A timeout or a used-up daily quota moves straight to the next model,
    because neither gets better by asking the same model again; only 429/503/500 are
    retried in place, briefly.

    The model that actually answered is recorded on llm_fn.model_used.
    """
    models = (models,) if isinstance(models, str) else (tuple(models) or DEFAULT_MODELS)

    def llm_fn(system: str, user: str) -> str:
        global _client
        from google import genai
        from google.genai import types

        if _client is None:
            _client = genai.Client()  # reads GEMINI_API_KEY from the environment
        cfg = {"system_instruction": system, "response_mime_type": "application/json",
               "max_output_tokens": MAX_OUTPUT_TOKENS,
               "http_options": types.HttpOptions(timeout=int(timeout_s * 1000))}
        if SimplifierSchema is not None:
            cfg["response_schema"] = SimplifierSchema
        if temperature is not None:
            cfg["temperature"] = temperature

        llm_fn.model_used = None
        errors, out_of_quota = [], []
        for model in models:
            for attempt in range(RETRIES_PER_MODEL + 1):
                try:
                    resp = _client.models.generate_content(
                        model=model, contents=user,
                        config=types.GenerateContentConfig(**cfg))
                    if not resp.text:
                        raise ValueError("returned no text (empty or blocked)")
                    llm_fn.model_used = model
                    return resp.text
                except Exception as e:  # noqa: BLE001
                    msg = str(e)
                    errors.append(f"{model}: {msg[:160]}")
                    if _is_timeout(e):
                        print(f"{model} did not answer within {timeout_s:g}s - "
                              f"moving to the next model.")
                        break   # a model that hangs will most likely hang again
                    if "PerDay" in msg or "per day" in msg.lower():
                        out_of_quota.append(model)
                        print(f"{model}: daily quota used up - moving to the next model.")
                        break   # waiting will not help this model today
                    is_rate_limit = "429" in msg or "RESOURCE_EXHAUSTED" in msg
                    is_busy = any(k in msg for k in ("503", "UNAVAILABLE", "500"))
                    if not (is_rate_limit or is_busy) or attempt == RETRIES_PER_MODEL:
                        break   # not retryable, or out of attempts on this model
                    if is_rate_limit:
                        wait, why = min(20 * (2 ** attempt), 60), "Rate limit reached"
                    else:
                        wait, why = min(5 * (2 ** attempt), 20), "Model is busy"
                    print(f"{model}: {why}. Waiting {wait}s "
                          f"({attempt + 1}/{RETRIES_PER_MODEL})...")
                    time.sleep(wait)

        if out_of_quota and len(out_of_quota) == len(models):
            raise RuntimeError(
                "Gemini DAILY quota is used up for every model in the chain "
                f"({', '.join(models)}) - waiting will not help. Try again after the "
                "quota resets, or use an API key from a different Google project.")
        raise RuntimeError("Every model in the chain failed:\n  " + "\n  ".join(errors))

    llm_fn.model_used = None
    return llm_fn


def _parse(raw: str) -> list:
    raw = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw.strip())
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON ({e}). First 200 chars: {raw[:200]!r}") from e
    sentences = data.get("sentences") if isinstance(data, dict) else data
    if not isinstance(sentences, list):
        raise ValueError("Model output has no 'sentences' list.")
    return sentences


# ----------------------------------------------------------------------------- checks
def check_explanation(raw_sentences: list, atoms: list[dict], glossary: list[dict]) -> dict:
    """Verify the LLM's sentences against the atoms. Pure Python, no API calls."""
    by_id = {a["id"]: a for a in atoms}
    gloss_ids = {g["topic_id"] for g in glossary}
    sentences = []

    for raw in raw_sentences:
        if not isinstance(raw, dict) or not _clean(raw.get("text")):
            continue
        text, flags = _clean(raw["text"]), []

        ids = []
        for x in raw.get("atom_ids") or []:
            x = str(x)
            if x not in by_id:
                flags.append(f"unknown_atom_id:{x}")
            elif x not in ids:
                ids.append(x)
        gids = []
        for g in raw.get("glossary_topic_ids") or []:
            try:
                g = int(g)
            except (TypeError, ValueError):
                flags.append(f"bad_glossary_id:{g}")
                continue
            if g not in gloss_ids:
                flags.append(f"unknown_glossary_id:{g}")
            elif g not in gids:
                gids.append(g)

        cited = [by_id[i] for i in ids]
        if not cited:
            flags.append("no_atom_cited")   # a sentence that states nothing from the atoms
        else:
            allowed = set()
            for a in cited:
                allowed.update(_NUMBER.findall(a["text"]))
            extra = list(dict.fromkeys(n for n in _NUMBER.findall(text) if n not in allowed))
            if extra:
                flags.append("numbers_not_in_cited_atoms:" + ",".join(extra))
            # Check the atom's own wording as well as its structured field. A cue can sit
            # inside a finding the Extractor called "present" ("ECG showed no acute ST
            # changes"), and the ADK format has no `subject` field at all, so neither
            # field alone catches a flipped rewrite.
            if (any(a["assertion"] == "negated" or _NEG.search(a["text"]) for a in cited)
                    and not _NEG.search(text)):
                flags.append("negation_may_be_lost")
            if (any(a["assertion"] == "uncertain" or _HEDGE.search(a["text"]) for a in cited)
                    and not _HEDGE.search(text)):
                flags.append("uncertainty_may_be_lost")
            if (any(a["subject"] == "family" or _FAMILY.search(a["text"]) for a in cited)
                    and not _FAMILY.search(text)):
                flags.append("family_subject_may_be_lost")

        sentences.append({"id": f"s{len(sentences) + 1}", "text": text, "atom_ids": ids,
                          "glossary_topic_ids": gids, "flags": flags})

    cited_text: dict[str, list[str]] = {a["id"]: [] for a in atoms}
    for s in sentences:
        for i in s["atom_ids"]:
            cited_text[i].append(s["text"])
    uncovered = [i for i, t in cited_text.items() if not t]
    missing_numbers = {}
    for i, texts in cited_text.items():
        if not texts:
            continue
        have = set(_NUMBER.findall(" ".join(texts)))
        miss = [n for n in _NUMBER.findall(by_id[i]["text"]) if n not in have]
        if miss:
            missing_numbers[i] = miss

    flagged = [s["id"] for s in sentences if s["flags"]]
    return {
        "sentences": sentences,
        "checks": {
            "uncovered_atoms": uncovered,
            "missing_numbers": missing_numbers,
            "flagged_sentences": flagged,
            "clean": not (uncovered or missing_numbers or flagged),
        },
    }


def _fk_grade(text: str):
    try:
        import textstat
    except ImportError:
        return None
    return round(textstat.flesch_kincaid_grade(text), 1)


# ----------------------------------------------------------------------------- public API
def simplify(atoms, *, retriever=None, rag_mode: str = "exact", model=None,
             temperature: float | None = None, llm_fn=None, grade: int = TARGET_GRADE,
             max_terms: int = MAX_TERMS, timeout_s: float = REQUEST_TIMEOUT_S) -> dict:
    """Rewrite atoms as a patient-friendly explanation with traceability and checks.

    model: one model name, or a sequence of them tried in order. Defaults to
           DEFAULT_MODELS. stats["model"] reports the one that answered.
    timeout_s: how long one model may take before the next is tried.
    llm_fn(system, user) -> raw JSON string. Leave None to use Gemini; pass your own to test.
    retriever: anything with .search(query, k, use_synonyms) and .surface_to_id
               (medlineplus_index.Retriever). An optional .search_many(queries, ...) lets
               "exact+related" embed every lookup in one request. Leave None to open the
               default index.
    """
    models = (model,) if isinstance(model, str) else (tuple(model) if model else DEFAULT_MODELS)
    atoms, glossary, system, user = prepare(atoms, retriever=retriever, rag_mode=rag_mode,
                                            grade=grade, max_terms=max_terms)
    if not atoms:
        parsed = {"sentences": [], "checks": {"uncovered_atoms": [], "missing_numbers": {},
                                              "flagged_sentences": [], "clean": True}}
    else:
        llm_fn = llm_fn or _make_gemini_llm_fn(models, temperature, timeout_s)
        parsed = check_explanation(_parse(llm_fn(system, user)), atoms, glossary)

    explanation = " ".join(s["text"] for s in parsed["sentences"])
    warnings = [f"extractor flag on {a['id']}: {', '.join(a['flags'])}" for a in atoms if a["flags"]]
    covered = len(atoms) - len(parsed["checks"]["uncovered_atoms"])
    return {
        "explanation": explanation,
        "sentences": parsed["sentences"],
        "glossary_used": glossary,
        "checks": parsed["checks"],
        "warnings": warnings,
        "stats": {
            "model": getattr(llm_fn, "model_used", None) or models[0],
            "models_configured": list(models),
            "rag_mode": rag_mode, "n_atoms": len(atoms), "n_atoms_covered": covered,
            "n_sentences": len(parsed["sentences"]),
            "n_glossary_exact": sum(g["match_type"] == "exact" for g in glossary),
            "n_glossary_related": sum(g["match_type"] == "related" for g in glossary),
            "flesch_kincaid_grade": _fk_grade(explanation) if explanation else None,
        },
    }


# ----------------------------------------------------------------------------- CLI
DEMO_ATOMS = [  # what the Extractor produces for the sample note in extractor.py
    dict(id="a1", type="diagnosis", term="hypertension", text="Hypertension", assertion="present", subject="patient"),
    dict(id="a2", type="diagnosis", term="type 2 diabetes", text="Type 2 diabetes mellitus", assertion="present", subject="patient"),
    dict(id="a3", type="diagnosis", term="breast cancer", text="Mother had breast cancer", assertion="present", subject="family"),
    dict(id="a4", type="symptom", term="chest pain", text="Chest pain", assertion="present", subject="patient"),
    dict(id="a5", type="symptom", term="shortness of breath", text="No shortness of breath", assertion="negated", subject="patient"),
    dict(id="a6", type="symptom", term="fever", text="No fever", assertion="negated", subject="patient"),
    dict(id="a7", type="lab_value", term="ECG", text="ECG showed no acute ST changes", assertion="present", subject="patient"),
    dict(id="a8", type="lab_value", term="troponin", text="Troponin I was 0.02 ng/mL (normal < 0.04)", assertion="present", subject="patient"),
    dict(id="a9", type="lab_value", term="stress test", text="Stress test was negative for ischemia", assertion="present", subject="patient"),
    dict(id="a10", type="diagnosis", term="gastroesophageal reflux disease", text="Possible gastroesophageal reflux disease", assertion="uncertain", subject="patient"),
    dict(id="a11", type="medication", term="metformin", text="Metformin 500 mg twice daily", assertion="present", subject="patient"),
    dict(id="a12", type="medication", term="lisinopril", text="Lisinopril 10 mg once daily", assertion="present", subject="patient"),
    dict(id="a13", type="medication", term="aspirin", text="Aspirin 81 mg daily was started", assertion="present", subject="patient"),
    dict(id="a14", type="instruction", text="Follow up with primary care in 1 week", assertion="present", subject="patient"),
    dict(id="a15", type="instruction", text="Avoid strenuous exercise until cleared", assertion="present", subject="patient"),
    dict(id="a16", type="instruction", text="Return to the ED for worsening chest pain", assertion="present", subject="patient"),
]


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="FaithfulMed Simplifier")
    p.add_argument("file", nargs="?", help="JSON file: a list of atoms, or the Extractor's saved result")
    p.add_argument("--demo", action="store_true", help="use the built-in sample atoms")
    p.add_argument("--rag", default="exact", choices=RAG_MODES)
    p.add_argument("--dry-run", action="store_true", help="show the glossary lookup + prompt; no Gemini call")
    p.add_argument("--model", default=None,
                   help="one model, or a comma-separated fallback chain "
                        f"(default: {','.join(DEFAULT_MODELS)})")
    p.add_argument("--timeout", type=float, default=REQUEST_TIMEOUT_S,
                   help="seconds one model may take before the next is tried "
                        f"(default: {REQUEST_TIMEOUT_S:g})")
    p.add_argument("--out", default=None, help="write the full result JSON here")
    args = p.parse_args(argv[1:])

    if args.demo:
        atoms = DEMO_ATOMS
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            atoms = json.load(f)
    else:
        p.print_help()
        return 0

    if args.dry_run:
        atoms, glossary, system, user = prepare(atoms, rag_mode=args.rag)
        print(f"{len(atoms)} atoms | rag mode: {args.rag} | glossary entries found: {len(glossary)}\n")
        for g in glossary:
            print(f"[{g['match_type']:7}] {g['title']}  <- {g['matched_terms']}  (atoms {g['atom_ids']})")
        uncovered = [a for a in atoms if not any(a["id"] in g["atom_ids"] for g in glossary)]
        print(f"\nAtoms with no glossary entry: {len(uncovered)}")
        for a in uncovered:
            src = "extractor" if _clean(a.get("term")) else "derived"
            print(f"   - {a['text']}\n       lookup key ({src}): {atom_term(a) or '(none)'}")
        print("\n----- prompt sent to the model (user part) -----\n" + user)
        return 0

    models = [m.strip() for m in args.model.split(",") if m.strip()] if args.model else None
    result = simplify(atoms, rag_mode=args.rag, model=models, timeout_s=args.timeout)
    print("EXPLANATION\n" + (result["explanation"] or "(empty)") + "\n")
    for s in result["sentences"]:
        flag = f"   <-- {', '.join(s['flags'])}" if s["flags"] else ""
        print(f"{s['id']:>4}  atoms {','.join(s['atom_ids']) or '-':<10} gloss {s['glossary_topic_ids']}  {s['text']}{flag}")
    print("\nchecks:", json.dumps(result["checks"]))
    for w in result["warnings"]:
        print("warning:", w)
    print("stats:", json.dumps(result["stats"]))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"saved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
