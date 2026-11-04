# FaithfulMed Architecture

Task #3, October milestone. Owner: Edwin (Joao).
**Version 2, 2026-10-13.** Includes the RAG pipeline, as advised by our Challenge Advisor.

This document exists so that six people building six pieces agree on what passes between
them. If you are building an agent, the section for your agent tells you what you receive
and what you must hand on. The code that matches it is `notebooks/adk_pipeline.py`, and
`src/test_pipeline_offline.py` runs the whole chain end to end with a fake model, so you
can check the wiring without an API key.

---

## What's new in version 2

Version 2 adds two steps and tightens how verification works. Everything below is in the
code and covered by the offline test.

| # | Area | v1 | v2 |
|---|---|---|---|
| 1 | What the Verifier reads | The draft and the extracted facts | Also the original **source**. The facts become a checklist. |
| 2 | Unexplained terms | Counted against faithfulness | **Faithful**, reported as a readability gap under `unexplained_terms` |
| 3 | The final text | Readability was the last step | A **FinalCheck** reads the final text. Report only, so it cannot loop. |
| 4 | Retrieval | Part of the Simplifier step | Its own **Retriever** step. Output stays in state, so the Verifier can see it. |
| 5 | Fact schema | Five types | Adds an `allergy` type, a `negated` flag, and a `term` to look up |

What stays the same: five agents, fixed order, one loop capped at two rounds,
temperature 0, and only one step allowed to query the glossary.

---

## The pipeline

```
 SOURCE (in session state)
    │
    ▼
 1. EXTRACTOR ─────────► atoms
    │
    ▼
 2. RETRIEVER ◄───────── glossary: 2,868 names -> 1,017 topics (exact match first,
    │                    ChromaDB vector search only on a miss)
    ├────────────────► definitions, unexplained_terms
    ▼
 3. SIMPLIFIER ────────► draft
    │
    ▼
 ┌─ 4. VERIFIER ───────► verdict        reads SOURCE + atoms + definitions + draft
 │  5. REFINER ────────► draft          fixes only what the verdict flagged
 └─ loop, at most 2 rounds
    │
    ▼
 6. READABILITY ───────► final
    │
    ▼
 7. FINALCHECK ────────► final_verdict  reads SOURCE + final. Report only, no edits.
```

**Retrieval happens in exactly one place: the Retriever.** It is a plain Python step, not
an LLM, so an exact match costs no quota. Its output stays in session state, which is what
lets the Verifier tell a retrieved definition from an invented one. If several agents
could each look things up, a fact could enter the explanation without leaving a trace.

**Why FinalCheck does not edit.** If it could edit, it would need checking too, and we are
back to an open loop. It reports, and the report is what the December evaluation reads.

---

## What passes between agents

Each step reads the keys under "in" and writes the keys under "out". The names match
`notebooks/adk_pipeline.py`, and the offline test fails if they drift.

**Before a run,** put the source text in session state:

```python
state = new_session_state(source_text)    # {"source": source_text}
```

Every `{placeholder}` in an instruction must exist in state or ADK raises a KeyError, so
this is required, not optional.

### 1. Extractor
**in:** `source`
**out:** `atoms`

```json
[
  {"text": "metformin 500 mg twice daily", "type": "medication", "term": "metformin", "negated": false},
  {"text": "HbA1c 8.2%", "type": "lab_value", "term": "HbA1c", "negated": false},
  {"text": "allergic to penicillin", "type": "allergy", "term": "penicillin", "negated": false},
  {"text": "chest pain", "type": "diagnosis", "term": "", "negated": true}
]
```

Types: `diagnosis`, `medication`, `lab_value`, `procedure`, `instruction`, `allergy`.

- `term` is the one word or phrase a patient might not know. The Retriever looks up this,
  not the whole sentence, because "metformin 500 mg twice daily" matches nothing.
- `negated` keeps "no chest pain" from becoming "chest pain".
- Redaction placeholders such as `XXXX` are skipped. 1.3% of MTSamples documents have them.

**What `source` is depends on the corpus.** MedAESQA is prose, MTSamples is dictated notes
with comma-separated section headers, Synthea is CSV rows. See "The three shapes of source
text" below.

### 2. Retriever (no LLM)
**in:** `atoms`
**out:** `definitions`, `unexplained_terms`

Looks up each atom's `term` with `src/glossary_lookup.py`. `definitions` is a text block
listing what was found and, separately, what was not. `unexplained_terms` is a JSON list.

With no glossary attached it still runs and marks every term as having no definition, so
the pipeline works before the vector index is finished.

### 3. Simplifier
**in:** `source`, `atoms`, `definitions`
**out:** `draft`

Rewrites into plain language. Uses the definitions where given. Keeps any term without a
definition exactly as written. Keeps every negation. States lab values as the source does,
without calling them high or low unless the source does.

### 4. Verifier
**in:** `source`, `atoms`, `definitions`, `draft`
**out:** `verdict`

```json
{
  "faithful": true,
  "unsupported_claims": [],
  "omissions": [],
  "negation_errors": [],
  "unexplained_terms": ["metformin"],
  "reading_level_ok": true
}
```

- **Unsupported** means supported by neither the source nor the retrieved definitions.
- **An unexplained term is not unsupported.** It goes in `unexplained_terms` and does not
  make `faithful` false.
- A negation error counts as unsupported.
- `faithful` is true only when `unsupported_claims` and `omissions` are both empty.

This is the verdict Isha calibrates against MedAESQA's human labels.

### 5. Refiner
**in:** `source`, `draft`, `verdict`
**out:** `draft` (overwrites, so the next round re-checks the corrected text)

Removes unsupported claims, restores omissions using only the source, fixes negation
errors. Changes nothing the verdict did not flag. Returns the draft unchanged if it was
already faithful.

### 6. Readability
**in:** `draft`
**out:** `final`

Rewrites to grade 8 or below. Every number and every negation stays. Drops nothing, adds
nothing, keeps a term that has no everyday equivalent.

`src/readability_agent.py` is the stricter standalone version used for evaluation: it
rejects any rewrite that changes a number or loses a negation, and keeps the harder text.

### 7. FinalCheck
**in:** `source`, `atoms`, `definitions`, `final`
**out:** `final_verdict`

Same instruction as the Verifier, pointed at the final text. Report only.

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

The Retriever does **not** embed the whole source document. It looks up one term at a
time: the `term` field of each atom, which is the one word or phrase a patient might not
know. Atoms with an empty `term` are skipped.

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
vector match and null for an exact one. A vector match below the score threshold is
reported as a miss, not as a weak hit, because a wrong definition is worse than none.

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

**How the Verifier treats a miss (changed in v2).** A term copied from the source is
faithful, so it is **not** an unsupported claim. It is listed under `unexplained_terms`,
which is a readability measure. What the Verifier does flag is an *invented* definition: an
explanation that appears in neither the source nor the retrieved definitions.

This matters because the glossary defines no medications, so drug names are the most
common unexplained terms.

Open question for Ram: do we add a second source for medications, or accept the gap and
report it as a limitation? MedQuAD cannot fill it: its drug answers are 100% empty.

---

## Design decisions behind v2

The reasoning for each v2 change, for anyone building against it.

### The Verifier reads the source, not only the extracted facts

The brief asks for explanations "provably faithful **to the source**". If the Verifier
compared the draft only against the extracted facts, any fact the Extractor dropped would
be invisible to it, and verification would inherit every extraction error. So the Verifier
and FinalCheck both read the source. The extracted facts are still passed in, as a
checklist for omissions.

The cost is a longer prompt, which matters on a tight quota. It is the only way to make the
faithfulness claim match what the brief asks for.

### An unexplained term is faithful

Three rules have to work together on a term with no definition:

1. The Simplifier **keeps the term as written**.
2. The Verifier decides whether that counts against faithfulness.
3. The Readability agent must **reach grade 8 without changing any fact**, and keeps a
   medical term that has no everyday equivalent.

The glossary defines zero medications, so this is the common case. A term copied verbatim
from the source is as faithful as text gets, so the Verifier does **not** count it as an
unsupported claim. It is listed under `unexplained_terms`, which is a readability measure.
What the Verifier flags is an *invented* definition: an explanation found in neither the
source nor the retrieved definitions.

### FinalCheck verifies the text the patient reads

Readability rewrites the text after the refine loop, and the loop's last step is a Refiner
edit. So the final text needs its own check. FinalCheck runs the Verifier's instruction on
the final text and reports. It does not edit, because anything that edits would need
checking too.

`src/readability_agent.py` adds a second guard for evaluation runs: it rejects any rewrite
that changes a number or drops a negation, and keeps the harder but correct text.

### Retrieval is its own step

Retrieval runs as a plain Python step between the Extractor and the Simplifier. Exact
matches cost no quota, and the result is written to state as `definitions`, so both the
Simplifier and the Verifier can read it. Every `{placeholder}` in an ADK instruction must
exist in state at run time, so a dedicated step is also what guarantees `definitions` is
there.

### Facts carry a type, a term, and a negation flag

- **`allergy`** is a sixth type. Synthea records 597 allergies and ALLERGIES appears 261
  times in a 1,500-document MTSamples sample. They are safety-relevant and need a home.
- **`negated`** keeps "no acute cardiopulmonary change" from becoming "acute
  cardiopulmonary change". MTSamples is full of negated findings.
- **`term`** is what the Retriever looks up. "metformin" matches; "metformin 500 mg twice
  daily" does not.

Social history (smoking status, quality-of-life scores) has no type yet. It can wait.

---

## Open questions

| # | Question | Status | Who decides |
|---|---|---|---|
| 1 | Lab values need interpretation; no agent provides it | Default: state the value, no high or low unless the source says so. Reference ranges open. | team |
| 2 | Neither corpus contains patient instructions | Open | team, Ram |
| 3 | On Synthea the Extractor is a flattener, not an agent | Open | team |
| 4 | No dataset here can define a medication | Open | **Ram** |
| 5 | PLABA missing, no owner | Open | team |

**Also to confirm:** Isha on the Verifier changes and Vera on the fact schema, since those
agents are theirs.

### 1. Lab values need interpretation, which no agent provides

> **Default for now:** the Simplifier states lab values as the source does and never calls them high or low on its own. Interpreting them still needs reference ranges.

A patient does not need "glucose is a sugar in your blood". They need "your blood sugar
was higher than normal". That requires a reference range.

Synthea stores VALUE and UNITS as separate columns, so the number is available, but
nothing in the pipeline turns 142 mg/dL into "higher than normal", and the glossary
carries definitions, not ranges. The `lab_value` type needs an interpretation step that
the pipeline does not have yet.

**Options:** add reference ranges as a lookup; state values without interpretation; or
scope lab interpretation out and say so.

### 2. Neither source corpus contains the instructions a patient most needs

MTSamples is dominated by operative notes and radiology reports, records of what was done
rather than what the patient should do. PLAN appears 202 times in a 1,500-document sample,
and Discharge Summary is only 108 of 4,999 rows before deduplication. Synthea has no
instruction text at all, only coded care-plan descriptions.

The `instruction` atom type is therefore the rarest thing in our data and the most
important thing in a patient-facing explanation. If the December demo is meant to show
discharge instructions being made readable, the corpus does not really contain them.

### 3. On Synthea the Extractor is not an LLM agent at all

Synthea has no clinical narrative. Its rows are already typed atoms:
conditions map to diagnosis, medications to medication, observations to lab_value,
procedures to procedure, careplans to instruction. The mapping is almost exact.

That is convenient, and it breaks the November ablation. "Extractor removed" is not a
meaningful condition when the Extractor is a deterministic flattener. **The ablation has to
run on MTSamples or MedAESQA, and the Synthea arm needs separate scoping.** Renita already
frames Synthea as an input-stage stress test rather than an evaluation set, which is the
right call, but no one has written down the slicing rule: a median patient yields several
hundred atoms and the largest yields thousands, which does not fit a 20-request daily quota.

### 4. No dataset here can define a medication

The September milestone says the index spans "the MedlinePlus lay-language glossary **and
authoritative lay-health guidelines**", and the dataset table lists MedQuAD for "RAG
grounding". This document and the current plan cover MedlinePlus only.

This matters because of the measured medication gap: MedlinePlus defines no drugs. MedQuAD has roughly 47k consumer-health Q&A pairs from NIH sources, so it looks like the
obvious second source for drug definitions.

**It is not, and this is now measured.** Gisele's report shows MedQuAD is 65.4% empty
answers overall, and the drug sources specifically are 100% empty: MedlinePlus Drugs
(12,889 pairs), MedlinePlus Herbs and Supplements (792), and ADAM (17,348). The answers
were redacted at collection for copyright. The one dataset that looked like it could fill
the medication gap is empty exactly where we need it.

So the question for Ram is this: **no dataset currently in this project can define a
medication in plain language.** Either we add a source outside the
brief's list, or we accept the gap and report it.

### 5. PLABA is unused and uncommitted

The brief calls "MTSamples + PLABA" the simplification core, and PLABA is listed as gold
professional-to-plain-language pairs for the Simplifier. There is no PLABA data or notebook
in the repo. Those pairs are the natural evaluation set for the Simplifier and the
Readability agent, so this is a real gap, not a bookkeeping one.

---

## The three shapes of source text

This is the part most likely to break the Extractor, and it was not in the first draft of
this document. The corpora are structurally different:

| Corpus | Shape | Size | Reading level | Measured by |
|---|---|---|---|---|
| **MedAESQA** | Prose answers | ~93 words median | FK 11.79 | Isha |
| **MTSamples** | Section-structured reports with headers (HISTORY OF PRESENT ILLNESS, MEDICATIONS, ALLERGIES) | ~465 words mean | FK 10.30, 90.3% above grade 8 | Isabella |
| **Synthea** | 16 linked CSV files, not prose at all | 1,171 patients, 42,989 medication rows, 299,697 observations | not applicable | Renita |

Three consequences:

1. **The baseline and the pipeline currently run on different input.** The baseline
   simplifies MedAESQA expert answers, which are prose. The brief describes simplifying
   clinical documents. If the multi-agent pipeline is later run on MTSamples reports, the
   comparison against the baseline is not like for like. **Freeze which corpus the
   headline comparison uses, and say so.**
2. **MTSamples sections are structure the Extractor can exploit.** A MEDICATIONS header is
   a stronger type signal than anything inferable from prose. Worth using rather than
   flattening away.
3. **Synthea needs a flattener before an LLM sees it**, which the brief also warns about.
   Rows across 16 CSVs are not a document. Whoever uses Synthea builds that first.

### MTSamples details the Extractor owner needs

**Only 47.1% of rows are unique** (2,357 of 4,999). Duplication is wildly uneven by
specialty: Radiology is 91% unique and Surgery 88%, but Cardiovascular/Pulmonary is 7%,
Consult History and Physical 11%, and Gastroenterology 13%. A random row sample will
over-represent whichever templates happen to repeat, and every faithfulness score becomes
an average over near-identical documents. **Deduplicate before sampling.** Isabella flagged
this as needing a team decision and nothing in the repo shows it has been made.

**Sections are not newline-delimited.** They appear as `ALL CAPS:` followed by a comma,
with commas also separating sections, which is a dictation artifact. A naive line splitter
will fail. There are 1,168 distinct header strings in a 1,500-document sample, so the top
20 are stable but the vocabulary cannot be enumerated. Isabella exported
`mtsamples_header_vocabulary.csv` for exactly this.

**Redaction placeholders.** 67 documents (1.3%) contain `XXXX` where identifiers were
removed. These look like extractable values. The Extractor must be told to skip them.

**Also:** 33 documents have no transcription at all, and 21.4% are missing keywords.

**Unmeasured and worth measuring:** nobody has counted how dense medications and lab values
are inside MTSamples prose. That is the number the Extractor owner actually needs, and it
would tell us how often the medication retrieval gap will bite.

### Two things to check in the Synthea notebook

Flagged for Renita to confirm, not changed by me.

**The records-per-patient number looks wrong.** Cell output reports 8,155 entities with a
mean of 57.7 and a median of 1. But there are 1,171 patients. The loop appears to count
organizations, payers and providers as patients, because the helper that finds a patient
column falls through to their own `Id`. That would add 1,119 plus 10 plus 5,855 rows, each
with one record, which explains both the 8,155 and the median of 1. Summing the per-table
means for real patients gives several hundred records each, not 57.7. **This number should
not go in the December report as written.**

**The example patient timeline does not demonstrate what it claims.** Most resources show
`NaT` for the date and raw codes like `185345009` instead of descriptions, because the
column picker matches `code` before `description` in several tables. The claim that a
patient history can be ordered chronologically is asserted but not shown.

Neither affects any architecture decision here. Both affect numbers that could reach the
report.

**One housekeeping item:** a personal absolute path appears in the committed notebook
output. Harmless, but worth clearing before the repo is shown as a portfolio artifact.

---

## Metrics every agent is measured on

From the brief, and matching `notebooks/eval_harness.py` so the whole team reports the same
numbers:

| Metric | Where |
|---|---|
| Flesch-Kincaid grade (primary) | `eval_harness.readability_scores` |
| SMOG (secondary) | same |
| Medical-jargon density | same, currently a Dale-Chall proxy |
| Output length | same |
| Refusal rate | per agent |
| Faithfulness, calibrated to human labels | Verifier, October |
| Latency, cost, openness | leaderboard, November |

The jargon measure is a known approximation. `eval_harness.py` notes it uses the Dale-Chall
familiar-word list as a proxy and should be swapped for a medical-term lookup later. Until
then, treat jargon density as directional.

---

## Design rules

**Fixed order, one loop.** The main flow is a `SequentialAgent`. The only loop is
`LoopAgent(max_iterations=2)` around Verifier and Refiner. Open-ended agent recursion is
hard to debug, burns shared rate limits, and produces traces we cannot reproduce for the
December report. This came from the SME technical review.

**Temperature 0 everywhere.** The November ablation compares the pipeline with each agent
switched off. That comparison only means something if the runs are reproducible. It is set
once in `adk_pipeline.py` and applied to every agent.

**The model is a parameter.** `build_pipeline(model=...)`, default `gemini-3.5-flash`.
`gemini-2.0-flash` and `gemini-2.5-flash` are retired and return 404. `gemini-3.6-flash`
returned 503 on every long prompt on 2026-10-13 while 3.5-flash answered the same prompts.
Record the model string with every result.

**Build a fresh pipeline per configuration.** An ADK agent can only have one parent, so
the ablation calls `build_pipeline()` once per row rather than reusing agents.

**Check the wiring before spending quota.** `python src/test_pipeline_offline.py` runs the
whole chain with a fake model in seconds. If you rename a state key, it fails.

**ADK version is pinned.** `google-adk` 2.12 marks `SequentialAgent` and `LoopAgent` as
deprecated in favor of a new `Workflow` class, to be removed in a future version. They
still work and the brief asks for them by name, so `requirements.txt` pins
`google-adk==2.12.0` to stop a silent upgrade from breaking the pipeline mid-semester.

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

**Unassigned work this document surfaced:**

| Piece | Why it matters | Owner |
|---|---|---|
| Synthea flattener and slicing rule | Synthea has no document to extract from, and a whole patient record does not fit the quota | nobody |
| PLABA download and EDA | The brief calls it the simplification core and gold pairs for the Simplifier. Not in the repo. | nobody |
| MTSamples deduplication decision | 47.1% unique. Affects every sample drawn from it. | flagged by Isabella, undecided |
| Second source for medication definitions | No dataset in the project can define a drug in plain language | needs Ram |
