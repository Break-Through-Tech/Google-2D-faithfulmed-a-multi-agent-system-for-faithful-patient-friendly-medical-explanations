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
**in:** `source`
**out:** `atoms`

**What `source` actually is depends on the corpus, and they are not alike.** See
"The three shapes of source text" below. The Extractor must handle at least MedAESQA
prose and MTSamples section-structured reports, which are different parsing problems.

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

## Open problems in this design

**Summary, for anyone who reads one table in this document:**

| # | Problem | Who decides | Blocks |
|---|---|---|---|
| 0 | Three rules here contradict each other on an unretrieved term | Isha, Edwin, Gisele | Simplifier and Verifier |
| 0b | Lab values need interpretation; no agent provides it | team | `lab_value` atoms |
| 0c | Neither corpus contains patient instructions | team, Ram | the December demo |
| 1 | Verifier checks atoms, not source | Isha | faithfulness claim |
| 2 | Readability runs after the last verification | Edwin, team | final output |
| 3 | The loop ends on an unverified Refiner write | team | |
| 3b | No atom type for allergies; negations can invert | Vera | Extractor schema |
| 3c | On Synthea the Extractor is a flattener, not an agent | team | November ablation |
| 4 | No dataset here can define a medication | **Ram** | the whole Simplifier case |
| 5 | PLABA missing, no owner | team | Simplifier evaluation |

Items 0, 1 and 4 are the ones worth meeting about. The rest can be decided in the group chat.

These were found by auditing this document against the repo and the advisor's brief on
2026-10-13. They are listed here rather than quietly fixed because three of them need a
team or advisor decision, not a unilateral one.

### 1. The Verifier checks the draft against the atoms, not against the source

The scaffold's Verifier instruction is "Compare {draft} against {atoms}". But the project
brief's stated goal is explanations "provably faithful **to the source**".

Those are not the same claim. If the Extractor drops a fact, the atoms are already
incomplete, and the Verifier will pass a draft that is missing it. Verification inherits
every extraction error silently.

**Options:** give the Verifier the source as well as the atoms, which costs tokens on a
tight quota; or add an extraction-completeness check; or accept the limitation and state
it plainly in the December report. **Needs a decision. Isha owns the Verifier.**

### 2. Readability runs after the Verifier, so its output is never verified

Order is Extractor, Simplifier, Verifier and Refiner loop, then Readability. The
Readability agent rewrites the text after the last verification, so a fact it breaks
reaches the patient unchecked.

`src/readability_agent.py` mitigates this with its own fact-check: any pass that changes a
number or drops a negation is rejected. That is a guard, not verification. It catches
numeric and negation damage, not a subtly reworded claim.

**Options:** move Readability inside the loop; or re-verify after it; or keep the current
order and rely on the agent's internal check. **Needs a decision.**

### 3. The loop ends on a Refiner write, so the last edit is unverified

`LoopAgent(sub_agents=[verifier, refiner], max_iterations=2)` runs verify, refine, verify,
refine. The final Refiner edit is never read back by the Verifier.

**Option:** a single verification pass after the loop closes, used as a report-only signal
rather than another edit.

### 0. Three rules in this document contradict each other

This is the most serious problem found, and all three rules are ones I wrote.

1. The Simplifier, on a retrieval miss, **keeps the term as written**.
2. The Verifier **treats an unretrieved definition as an unsupported claim**.
3. The Readability agent must **hit grade 8 without changing or dropping any fact**, and
   is told to keep a medical term that has no everyday equivalent.

Now trace a medication atom, which is the common case since the glossary defines zero
drugs. The Simplifier correctly leaves "Hydrochlorothiazide 25 MG Oral Tablet" untouched.
The Verifier then flags it as unsupported, so the expected case becomes a verification
failure. The Refiner is asked to fix something that is not broken. The Readability agent
inherits a grade-12 string and is told both to reach grade 8 and not to change it.

**Rule 2 is the wrong one.** "No definition was retrieved" and "this claim is not
supported by the source" are different things. A term copied verbatim from the source is
maximally faithful to it. The Verifier should flag an *invented* definition, not an
*absent* one.

**Proposed fix, needs Isha's agreement:** the Verifier flags a definition that does not
appear in the source or the retrieved context. An unexplained term passes faithfulness and
is reported separately as a readability gap, which is what it actually is.

### 0b. Lab values need interpretation, which no agent provides

A patient does not need "glucose is a sugar in your blood". They need "your blood sugar
was higher than normal". That requires a reference range.

Synthea stores VALUE and UNITS as separate columns, so the number is available, but
nothing in the pipeline turns 142 mg/dL into "higher than normal", and the glossary
carries definitions, not ranges. The `lab_value` atom type implies an interpretation step
the design has no component for.

**Options:** add reference ranges as a lookup; state values without interpretation; or
scope lab interpretation out and say so.

### 0c. Neither source corpus contains the instructions a patient most needs

MTSamples is dominated by operative notes and radiology reports, records of what was done
rather than what the patient should do. PLAN appears 202 times in a 1,500-document sample,
and Discharge Summary is only 108 of 4,999 rows before deduplication. Synthea has no
instruction text at all, only coded care-plan descriptions.

The `instruction` atom type is therefore the rarest thing in our data and the most
important thing in a patient-facing explanation. If the December demo is meant to show
discharge instructions being made readable, the corpus does not really contain them.

### 3b. The atom type set has no slot for allergies or social history

Five types are defined: diagnosis, medication, lab_value, procedure, instruction.

Synthea records 597 allergies, and ALLERGIES appears 261 times in a 1,500-document
MTSamples sample. An allergy is patient-safety relevant and currently has nowhere to go,
so it gets mistyped or silently dropped. Synthea also records smoking status on 12,552
rows and quality-of-life indices on 10,121, which are not lab values in any useful sense.

**Proposal:** add `allergy` as a sixth type. Social history can wait, but allergies
should not.

**Negated findings are the dangerous gap.** MTSamples is full of phrases like "no acute
cardiopulmonary change". If the Extractor records the finding without its negation, the
atom inverts the meaning, and the Verifier checks the draft against the atoms rather than
the source, so nothing catches it. An atom schema needs a negation flag, or the extraction
prompt has to be explicit that negated findings stay negated.

### 3c. On Synthea the Extractor is not an LLM agent at all

Synthea has no clinical narrative. Its rows are already typed atoms:
conditions map to diagnosis, medications to medication, observations to lab_value,
procedures to procedure, careplans to instruction. The mapping is almost exact.

That is convenient, and it breaks the November ablation. "Extractor removed" is not a
meaningful condition when the Extractor is a deterministic flattener. **The ablation has to
run on MTSamples or MedAESQA, and the Synthea arm needs separate scoping.** Renita already
frames Synthea as an input-stage stress test rather than an evaluation set, which is the
right call, but no one has written down the slicing rule: a median patient yields several
hundred atoms and the largest yields thousands, which does not fit a 20-request daily quota.

### 4. The index should probably cover more than MedlinePlus

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

So the open question for Ram is sharper than first written: **no dataset currently in this
project can define a medication in plain language.** Either we add a source outside the
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

**Unassigned work this document surfaced:**

| Piece | Why it matters | Owner |
|---|---|---|
| Synthea flattener and slicing rule | Synthea has no document to extract from, and a whole patient record does not fit the quota | nobody |
| PLABA download and EDA | The brief calls it the simplification core and gold pairs for the Simplifier. Not in the repo. | nobody |
| MTSamples deduplication decision | 47.1% unique. Affects every sample drawn from it. | flagged by Isabella, undecided |
| Second source for medication definitions | No dataset in the project can define a drug in plain language | needs Ram |
