# Extractor agent

Owner: Vera. Code: `src/extractor_agent.py`. Tests: `src/test_extractor_agent.py`.
Evaluation: `src/eval_extractor.py`.

## What it hands on

One list of atoms in `state["atoms"]`. Each atom carries both the fields in
`architecture.md` (read by the Verifier and the pipeline Retriever) and the fields
`src/simplifier.py` reads, so neither side has to change:

```json
{"id": "a2", "text": "metformin 500 mg twice daily", "type": "medication",
 "term": "metformin", "assertion": "present", "negated": false,
 "subject": "patient", "section": "MEDICATIONS"}
```

| Field | Values | Who reads it |
|---|---|---|
| `id` | `a1`, `a2`, ... | Simplifier (sentences cite atom ids) |
| `text` | as close to the source wording as possible | everyone |
| `type` | diagnosis, medication, lab_value, procedure, instruction, allergy, **symptom, other** | Simplifier, eval |
| `term` | the one word to look up, copied from the source, or `""` | Retriever |
| `assertion` | present, negated, **uncertain** | Simplifier |
| `negated` | derived: `assertion == "negated"` | Verifier negation guard |
| `subject` | patient, family, other | Simplifier |
| `section` | MTSamples header, or `""` | eval, debugging |

**Schema changes to confirm with the team:**

- `symptom` and `other` are new types. The Simplifier already accepts both. MedAESQA
  answers are general claims ("lifestyle changes may prevent sleep apnea") that none
  of the original six types fit.
- `uncertain` covers hedges. 34% of MedAESQA nuggets have one.
- `negated` is computed from `assertion` in code, never by the model, so the two cannot
  disagree.

## What the guards do

The model extracts. Plain Python checks the result before anything downstream sees it.

| Check | Action |
|---|---|
| A number in the atom is not in the source | reject the atom |
| Less than 50% of the atom's content words are in the source | reject (invented fact) |
| Redaction placeholder (`XXXX`) | reject |
| `term` is not in the source | clear it, warn |
| The source negates the finding but the atom says present | warn |
| Source mentions allergies, no allergy atom | warn |
| `NKDA` / "no known allergies" and no allergy atom | add a negated allergy atom |
| Duplicates | collapse |
| Unknown type / assertion labels | map aliases (Drug → medication), else `other` |

Rejected atoms and warnings are kept in `state["extractor_report"]` so the evaluation
can count them.

## How each source is handled

- **MedAESQA prose:** PubMed citation markers are removed before the model sees the
  text, so a PMID can't be extracted as a number.
- **MTSamples:** `split_sections()` splits comma-delimited `HEADER:,` sections and
  passes each one on its own line with a type hint ("[MEDICATIONS] (usually medication)").
- **Synthea:** not handled here. It needs a flattener (open question 3 in
  `architecture.md`).

## Running it

```bash
python src/test_extractor_agent.py       # 34 offline checks, no key
python src/test_pipeline_offline.py      # full pipeline with a fake model, 29 checks
python src/extractor_agent.py note.txt --show-prompt   # see the exact prompt, no API call
GOOGLE_API_KEY=... python src/extractor_agent.py note.txt
```

## Evaluation against the MedAESQA nuggets

```bash
python src/eval_extractor.py --mode sentences            # free reference run
python src/eval_extractor.py --mode gemini --limit 10    # 10 API calls
```

Source text is the 40 expert answers, the same frozen set the baseline uses. Gold is
the 498 nuggets in `data/processed/medaesqa_nuggets.csv`. A nugget is covered when
every concept in it (the text before each `[CUI]` tag) matches one atom. It's a strict
lexical match with no LLM judge, so it's cheap and reproducible.

Gemini replies are cached in `results/extractor_cache.jsonl`, so rescoring is free.

| Run | Nugget recall | Hedges kept | Atoms / question |
|---|---|---|---|
| Reference: one atom per sentence, no LLM | 0.725 | 0.735 | 4.8 |
| Gemini Extractor | *to run* | | |

The reference isn't a real extractor. It shows the scorer works, and it sets the bar a
real extractor should beat. Most of its misses are coreference: "This gene mostly
affects bone development" doesn't match the nugget `Affects(runx2 mutation, bone
development)`.

## Still open

1. Run the Gemini evaluation (40 calls, about two days of free-tier quota per key).
2. Measure how dense medications and lab values are in MTSamples, which needs
   the MTSamples CSV locally. Nobody has done this yet, and it shows how often the
   medication retrieval gap will come up.
3. Synthea flattener: unowned.
4. Coreference ("this gene", "it") is the biggest recall loss on MedAESQA. The prompt
   could tell the model to name the subject explicitly in each atom.
