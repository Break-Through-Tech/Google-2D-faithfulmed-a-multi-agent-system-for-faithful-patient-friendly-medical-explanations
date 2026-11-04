# Architecture walkthrough (v2)

A 15-minute walk through the pipeline for the team. Full detail is in `architecture.md`;
this is the version you can talk from.

---

## 1. The shape of it (3 min)

```
SOURCE
  -> 1. Extractor     pulls out the facts            -> atoms
  -> 2. Retriever     looks up plain definitions     -> definitions
  -> 3. Simplifier    rewrites in plain language     -> draft
  -> 4. Verifier  <-> 5. Refiner   check and fix, max 2 rounds
  -> 6. Readability   gets it to 8th grade           -> final
  -> 7. FinalCheck    checks what the patient reads  -> final_verdict
```

Say three things:

- **Each step hands the next one a named thing.** If you are building an agent, your
  section in `architecture.md` tells you exactly what you read and what you write.
- **Only the Retriever looks things up.** It is plain Python, not an AI call, so an exact
  match costs no quota. Because its output is saved, the Verifier can tell a looked-up
  definition from a made-up one.
- **One loop, capped at two rounds.** The technical review asked for this. It keeps runs
  cheap and repeatable.

## 2. How lookup works (2 min)

1. **Exact match** against 2,868 alternate names. "HbA1c" gives A1C. "Pimples" gives Acne.
   No AI call.
2. **Vector search** only if that misses.

The Extractor now tags each fact with the one word to look up, like "metformin", because
looking up the whole sentence "metformin 500 mg twice daily" matches nothing.

`src/glossary_lookup.py` works today without the vector index, so Gisele can build the
Simplifier before Renita's embeddings are done.

## 3. What changed from v1, and why (4 min)

I checked v1 against everyone's notebooks and the brief, then refined it. Here is what
changed.

**The main change: how we handle medications.** The glossary defines zero medications, so
drug names usually have no definition. In v2 the Simplifier keeps a drug name as written,
and the Verifier counts that as faithful, because it is copied straight from the source. It
gets listed as "unexplained", which is a readability issue, not a faithfulness one. The
Verifier only flags things that are *made up*.

Four more changes:

| v1 | v2 |
|---|---|
| The Verifier checked the draft against the extracted facts | It also reads the original source, so it catches anything extraction missed |
| Readability was the last step | New FinalCheck step reads the final text. It reports and does not edit, so no endless loop |
| Retrieval was part of the Simplifier | New Retriever step, so the definitions are saved and the Verifier can see them |
| Five fact types | Allergy type added, and every fact carries a negated yes/no so "no chest pain" stays negative |

**How we know it works:** `python src/test_pipeline_offline.py` runs the whole thing with a
fake model. 26 checks pass: right order, every input reaches the right agent, the Verifier
really sees the source, FinalCheck really reads the final text. No API key needed.

What it does not prove yet is that Gemini follows the instructions. That needs one real
run, which today's quota did not allow.

## 4. The numbers behind the design (3 min)

All measured, all from our own notebooks.

**Medications: 0 of 5.** The glossary covers conditions 13 of 13 and medications 0 of 5.
MedQuAD cannot fill the gap either: its drug answers are 100% empty, removed for copyright.

**Our source text is three different shapes.**

| Corpus | Shape | Reading level |
|---|---|---|
| MedAESQA | prose, ~93 words | grade 11.8 |
| MTSamples | dictated notes with ALL-CAPS headers, ~465 words | grade 10.3 |
| Synthea | 16 CSV files, no text at all | n/a |

**Only 47% of MTSamples rows are unique.** Isabella flagged this. Deduplicate before
sampling or we average over copies.

## 5. What we still need to decide (3 min)

**Isha: does the new Verifier work for you?** It now reads the source and treats an
unexplained term as faithful. The verdict also has two new fields, `negation_errors` and
`unexplained_terms`. You own this agent, so it is your call.

**Vera: the Extractor schema.** Each fact now has `term`, `negated`, and an `allergy`
type. Does that work for your prompt?

**Ram: medications.** No dataset in this project can explain a drug in plain language.
Either we add a source outside the brief's list, or we report it as a limitation.

**Team: patient instructions.** Neither MTSamples nor Synthea really contains discharge
instructions, which is what patients need most. Worth deciding what the December demo
shows.
