"""
FaithfulMed - Simplifier agent (Task #5)

Takes the Extractor's atoms and rewrites them as a patient-friendly explanation, using plain-language
definitions from the MedlinePlus ChromaDB index (RAG) to explain medical terms.

Pipeline position:  Extractor -> SIMPLIFIER -> Verifier -> Refiner -> Readability

How it works (a fixed sequence, no free-form agent loops):
  1. Load atoms                        (only `text` and `type` are required)
  2. RAG: find glossary terms          (deterministic exact match of atom wording against the
                                        glossary's titles + alternate names; no LLM, no API call)
  3. Gemini rewrites the atoms         (structured output: sentences, each citing atom ids and
                                        glossary topic ids)
  4. Plain-Python faithfulness checks  (does not trust the LLM):
       * every atom must be covered by some sentence            -> uncovered_atoms
       * every number in an atom must survive into its sentence -> missing_numbers
       * no new numbers may appear                              -> numbers_not_in_cited_atoms
       * negated / uncertain / family atoms must keep that meaning in the sentence (cue-word check)
       * sentences must cite real atoms and real glossary entries

RAG modes (useful for ablations):
  "off"            no glossary
  "exact"          (default) only exact glossary matches are given to the model as definitions
  "exact+related"  also gives the nearest topic for atoms with no exact match, labelled RELATED
                   BACKGROUND. These can be near-misses (e.g. "metformin" -> "Diabetes Medicines"), so
                   this mode needs the Gemini embedding API and is riskier for faithfulness.

Needs src/medlineplus_index.py and a built index (python src/medlineplus_index.py build).

Setup:   pip install google-genai pandas numpy chromadb
         $env:GEMINI_API_KEY="..."   (optional: $env:FAITHFULMED_MODEL="...")

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
DEFAULT_MODEL = os.environ.get("FAITHFULMED_MODEL", "gemini-3.8-flash")
MAX_OUTPUT_TOKENS = 16384
MAX_RETRIES = 8
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


def _clean(v) -> str:
    return "" if v is None else str(v).strip()


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
            "assertion": (_clean(a.get("assertion")) or "present").lower(),
            "subject": (_clean(a.get("subject")) or "patient").lower(),
            "flags": list(a.get("flags") or []),
        }
        if atom["id"] in seen:
            raise ValueError(f"Duplicate atom id: {atom['id']}")
        seen.add(atom["id"])
        out.append(atom)
    return out


# ----------------------------------------------------------------------------- RAG
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
        for surf in spot_terms(a["text"], stripped):
            hits = retriever.search(surf, k=1, use_synonyms=True)   # exact key -> no API call
            if hits and hits[0]["match_type"] == "synonym":
                add(hits[0], "exact", surf, a["id"])
                has_exact.add(a["id"])

    if rag_mode == "exact+related":
        for a in atoms:
            if a["id"] in has_exact or a["type"] not in RELATED_TYPES:
                continue
            hits = retriever.search(a["text"], k=1, use_synonyms=False)
            if hits:
                add(hits[0], "related", a["text"], a["id"])
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


def _make_gemini_llm_fn(model: str, temperature: float | None):
    def llm_fn(system: str, user: str) -> str:
        global _client
        from google import genai
        from google.genai import types

        if _client is None:
            _client = genai.Client()  # reads GEMINI_API_KEY from the environment
        cfg = {"system_instruction": system, "response_mime_type": "application/json",
               "max_output_tokens": MAX_OUTPUT_TOKENS}
        if SimplifierSchema is not None:
            cfg["response_schema"] = SimplifierSchema
        if temperature is not None:
            cfg["temperature"] = temperature

        for attempt in range(MAX_RETRIES + 1):
            try:
                resp = _client.models.generate_content(
                    model=model, contents=user, config=types.GenerateContentConfig(**cfg))
                break
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if "PerDay" in msg or "per day" in msg.lower():
                    raise RuntimeError(
                        "Gemini DAILY quota used up - waiting will not help. Try again after the "
                        "quota resets, or use an API key from a different Google project.") from e
                is_rate_limit = "429" in msg or "RESOURCE_EXHAUSTED" in msg
                is_busy = "503" in msg or "UNAVAILABLE" in msg
                if not (is_rate_limit or is_busy) or attempt == MAX_RETRIES:
                    raise
                if is_rate_limit:
                    wait, why = min(60 * (2 ** attempt), 300), "Rate limit reached"
                else:
                    wait, why = min(5 * (2 ** attempt), 60), "Model is busy (503)"
                print(f"{why}. Waiting {wait}s ({attempt + 1}/{MAX_RETRIES})...")
                time.sleep(wait)
        if not resp.text:
            raise ValueError("Gemini returned no text (the response was empty or blocked).")
        return resp.text

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
            if any(a["assertion"] == "negated" for a in cited) and not _NEG.search(text):
                flags.append("negation_may_be_lost")
            if any(a["assertion"] == "uncertain" for a in cited) and not _HEDGE.search(text):
                flags.append("uncertainty_may_be_lost")
            if any(a["subject"] == "family" for a in cited) and not _FAMILY.search(text):
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
def simplify(atoms, *, retriever=None, rag_mode: str = "exact", model: str | None = None,
             temperature: float | None = None, llm_fn=None, grade: int = TARGET_GRADE,
             max_terms: int = MAX_TERMS) -> dict:
    """Rewrite atoms as a patient-friendly explanation with traceability and checks.

    llm_fn(system, user) -> raw JSON string. Leave None to use Gemini; pass your own to test.
    retriever: anything with .search(query, k, use_synonyms) and .surface_to_id
               (medlineplus_index.Retriever). Leave None to open the default index.
    """
    model = model or DEFAULT_MODEL
    atoms, glossary, system, user = prepare(atoms, retriever=retriever, rag_mode=rag_mode,
                                            grade=grade, max_terms=max_terms)
    if not atoms:
        parsed = {"sentences": [], "checks": {"uncovered_atoms": [], "missing_numbers": {},
                                              "flagged_sentences": [], "clean": True}}
    else:
        llm_fn = llm_fn or _make_gemini_llm_fn(model, temperature)
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
            "model": model, "rag_mode": rag_mode, "n_atoms": len(atoms), "n_atoms_covered": covered,
            "n_sentences": len(parsed["sentences"]),
            "n_glossary_exact": sum(g["match_type"] == "exact" for g in glossary),
            "n_glossary_related": sum(g["match_type"] == "related" for g in glossary),
            "flesch_kincaid_grade": _fk_grade(explanation) if explanation else None,
        },
    }


# ----------------------------------------------------------------------------- CLI
DEMO_ATOMS = [  # what the Extractor produces for the sample note in extractor.py
    dict(id="a1", type="diagnosis", text="Hypertension", assertion="present", subject="patient"),
    dict(id="a2", type="diagnosis", text="Type 2 diabetes mellitus", assertion="present", subject="patient"),
    dict(id="a3", type="diagnosis", text="Mother had breast cancer", assertion="present", subject="family"),
    dict(id="a4", type="symptom", text="Chest pain", assertion="present", subject="patient"),
    dict(id="a5", type="symptom", text="No shortness of breath", assertion="negated", subject="patient"),
    dict(id="a6", type="symptom", text="No fever", assertion="negated", subject="patient"),
    dict(id="a7", type="lab_value", text="ECG showed no acute ST changes", assertion="present", subject="patient"),
    dict(id="a8", type="lab_value", text="Troponin I was 0.02 ng/mL (normal < 0.04)", assertion="present", subject="patient"),
    dict(id="a9", type="lab_value", text="Stress test was negative for ischemia", assertion="present", subject="patient"),
    dict(id="a10", type="diagnosis", text="Possible gastroesophageal reflux disease", assertion="uncertain", subject="patient"),
    dict(id="a11", type="medication", text="Metformin 500 mg twice daily", assertion="present", subject="patient"),
    dict(id="a12", type="medication", text="Lisinopril 10 mg once daily", assertion="present", subject="patient"),
    dict(id="a13", type="medication", text="Aspirin 81 mg daily was started", assertion="present", subject="patient"),
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
    p.add_argument("--model", default=None)
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
        uncovered = [a["text"] for a in atoms if not any(a["id"] in g["atom_ids"] for g in glossary)]
        print(f"\nAtoms with no glossary entry: {len(uncovered)}")
        for t in uncovered:
            print("   -", t)
        print("\n----- prompt sent to the model (user part) -----\n" + user)
        return 0

    result = simplify(atoms, rag_mode=args.rag, model=args.model)
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
