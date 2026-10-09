# FaithfulMed Architecture

Task #3, October milestone. Owner: Edwin (Joao).
Updated to include the RAG pipeline, as advised by our Challenge Advisor.

This document exists so that six people building six pieces agree on what passes between
them. If you are building an agent, the section for your agent tells you what you receive
and what you must hand on.

---

## The pipeline

```
                       ┌──────────────────────────┐
                       │  MedlinePlus glossary    │
                       │  1,017 topics            │
                       │  2,868 alternate names   │
                       └────────────┬─────────────┘
                                    │  (built once, offline)
                                    ▼
                       ┌──────────────────────────┐
                       │  ChromaDB vector index   │
                       │  Gemini text-embedding   │
                       └────────────┬─────────────┘
                                    │  lookup
                                    │
 SOURCE              ┌──────────────┴───┐
 clinical  ──────►   │                  │
 text                │   2. SIMPLIFIER  │
                     │                  │
        ┌────────────┴──────┐           │
        │  1. EXTRACTOR     │──atoms───►│
        └───────────────────┘           │
                                        │ draft
                                        ▼
                        ┌───────────────────────────┐
                        │  3. VERIFIER ⇄ 4. REFINER │   loop, max 2 passes
                        └───────────┬───────────────┘
                                    │ verified draft
                                    ▼
                        ┌───────────────────────────┐
                        │     5. READABILITY        │
                        └───────────┬───────────────┘
                                    │
                                    ▼
                            patient-friendly
                               explanation
```

**Retrieval happens in exactly one place: the Simplifier.** No other agent queries the
index. This is deliberate. If three agents could each retrieve, a fact could enter the
explanation without appearing in the extracted atoms, and the Verifier would have no way
to tell an invented claim from a retrieved one.

---

## What passes between agents

Each agent reads the keys listed under "in" and writes the key listed under "out". These
names match the `output_key` values in `notebooks/adk_pipeline.py`.

### 1. Extractor
**in:** `source` (the raw clinical text)
**out:** `atoms`

A list of clinical facts, each tagged with a type:

```json
[
  {"text": "prescribed metformin 500 mg twice daily", "type": "medication"},
  {"text": "HbA1c was 8.2%", "type": "lab_value"},
  {"text": "type 2 diabetes", "type": "diagnosis"}
]
```

Types: `diagnosis`, `medication`, `lab_value`, `procedure`, `instruction`.

The atoms are the contract for everything downstream. The Verifier checks the final
explanation against these, so a fact missing here cannot be verified later.

### 2. Simplifier
**in:** `source`, `atoms`, plus the retrieval index
**out:** `draft`

Rewrites into plain language. This is the only agent allowed to query the glossary index.

### 3. Verifier
**in:** `atoms`, `draft`
**out:** `verdict`

```json
{"faithful": true, "unsupported_claims": [], "omissions": [], "reading_level_ok": false}
```

### 4. Refiner
**in:** `draft`, `verdict`
**out:** `draft` (overwrites, so the loop re-checks the corrected version)

Fixes only what the Verifier flagged. No free rewriting.

### 5. Readability
**in:** `draft` (verified)
**out:** `final`

Rewrites to an 8th-grade reading level without changing or dropping any fact.

---

## The retrieval contract

This is the part Task #3 was opened to pin down. Renita and Gisele build the index,
Gisele builds the Simplifier, and both need the same answer to "what does a lookup
return?"

### What goes into the index

Source: MedlinePlus Health Topics XML, release 2026-09-15, frozen at SHA-256
`35970c9cc955221b720e0768bb49b1b6873147eb73eb4e7116d26b8d07cf6762`.
Measured in `notebooks/exploratory-data-analysis/medlineplus_glossary_eda.ipynb`.

| | |
|---|---|
| Topics | 1,017 English |
| Plain-language summaries | 1,012 |
| Alternate names | 2,868 surface forms mapping to those topics |

**Chunking.** Summaries run from 33 to 1,225 words, and 27% exceed about 512 words. One
vector per topic would truncate roughly a quarter of the corpus, so long summaries must be
split. Keep the topic title with every chunk so a retrieved fragment is still attributable.

### The query

The Simplifier does **not** embed the whole source document. It looks up one term at a
time: for each atom that contains a medical term a patient would not know, query that term.

Two steps, cheap one first:

1. **Exact match against the alternate-name table.** 2,868 surface forms map to canonical
   topics, including lay terms. "Pimples" returns Acne. "HbA1C" returns A1C. "Bellyache"
   returns Abdominal Pain. This is a plain dictionary lookup, no embedding call, and it is
   exact rather than approximate.
2. **Vector search only if step 1 misses.** Embed the term, retrieve the nearest topic
   summaries from ChromaDB.

Doing the dictionary step first saves embedding calls, which matters because the free
Gemini tier allows about 20 requests per day per key.

### What comes back

```json
{
  "term": "HbA1c",
  "found": true,
  "canonical_topic": "A1C",
  "definition": "A1C is a blood test for type 2 diabetes...",
  "source_url": "https://medlineplus.gov/a1c.html",
  "match_type": "exact_synonym",
  "score": null
}
```

`match_type` is `exact_synonym`, `vector`, or `none`. `score` is the similarity for a
vector match and null for an exact one. The Simplifier should treat a low-scoring vector
match with more caution than an exact synonym hit.

### When a lookup finds nothing

**This case is common and must be handled explicitly.** Tested against 30 terms of the
kind that appear in discharge papers:

| Content type | Found |
|---|---|
| Conditions | 13 of 13 |
| Symptoms | 3 of 3 |
| Lab tests | 2 of 4 |
| Procedures | 2 of 5 |
| **Medications** | **0 of 5** |

The glossary explains conditions well and **contains no medication entries at all**. No
metformin, no warfarin, no statin, no NSAID, no beta-blocker. Discharge papers are largely
medications, so a miss is the expected case, not an edge case.

**Required behavior on a miss:** the Simplifier keeps the term as written in the source and
does not invent a definition. Carrying an unexplained term forward is a readability
problem. Inventing a definition for a drug is a patient-safety problem, and it is the exact
failure this project exists to prevent.

The Verifier should treat an unretrieved definition as an unsupported claim.

Open question for Ram: do we add a second source for medications, or accept the gap and
report it as a limitation? Flagged in the 9/15 team update.

---

## Design rules

**Fixed order, one loop.** The main flow is a `SequentialAgent`. The only loop is
`LoopAgent(max_iterations=2)` around Verifier and Refiner. Open-ended agent recursion is
hard to debug, burns shared rate limits, and produces traces we cannot reproduce for the
December report. This came from the SME technical review.

**Temperature 0 everywhere.** The November ablation compares the pipeline with each agent
switched off. That comparison only means something if the runs are reproducible.

**Model: `gemini-3.6-flash`.** The `gemini-2.0-flash` in `adk_pipeline.py` is retired and
returns 404, as is `gemini-2.5-flash`. Record the model string with every result, since
availability is clearly shifting during this project.

**Everything is measured against the single-Gemini baseline.** One well-prompted call with
no agents, on the frozen MedAESQA set. The ablation table is that row repeated with one
agent removed at a time.

---

## Owners

| Piece | Owner | Due |
|---|---|---|
| Vector index: embeddings | Renita | 10/13 |
| Vector index: ChromaDB | Gisele | 10/13 |
| Single-Gemini baseline | Vera | 10/13 |
| Architecture and retrieval contract | Edwin | 10/13 |
| Extractor | Vera | 10/27 |
| Simplifier | Gisele | 10/27 |
| RAG wiring | Renita | 10/27 |
| Verifier | Isha | 10/27 |
| Refiner | Isabella | 10/27 |
| Readability | Edwin | 10/27 |
