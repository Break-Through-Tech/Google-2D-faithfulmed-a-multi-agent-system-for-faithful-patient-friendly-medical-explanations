# Architecture walkthrough

A 15-minute walk through the pipeline for the team, ending in three decisions we need to
make together. The full detail is in `architecture.md`; this is the version you can talk
from.

---

## 1. The shape of the thing (2 min)

```
SOURCE  ->  Extractor  ->  Simplifier  ->  Verifier <-> Refiner  ->  Readability  ->  OUTPUT
                               ^              (max 2 rounds)
                               |
                        glossary index
```

Five agents, fixed order, one loop. Say three things:

- **Each agent hands the next one a named thing.** Extractor produces `atoms`, Simplifier
  produces `draft`, Verifier produces `verdict`, Refiner overwrites `draft`, Readability
  produces `final`. These names match the scaffold exactly, so if you are building an
  agent, that is your input and output.
- **Only the Simplifier touches the glossary.** If three agents could each look things up,
  a fact could reach the explanation without ever being in the atoms, and the Verifier
  checks the draft against the atoms. It would have no way to tell a retrieved fact from
  an invented one.
- **One loop, capped at two rounds.** Open-ended agent recursion is undebuggable and burns
  the shared quota. This came from the technical review, not from us.

## 2. How lookup works (2 min)

Two steps, cheap one first.

1. **Exact match** against 2,868 alternate names. A plain dictionary, no API call.
   "HbA1c" gives A1C. "Pimples" gives Acne. "Bellyache" gives Abdominal Pain.
2. **Vector search** only if step 1 misses.

Doing the dictionary first matters because the free tier is about 20 requests a day per
key, so every embedding call we skip is one we can spend elsewhere.

`src/glossary_lookup.py` implements this and works right now without the vector index. So
Gisele can build and test the Simplifier before Renita's embeddings are finished.

## 3. What I got wrong (3 min)

Worth saying plainly, because it is the most useful part.

I wrote the first version from my own notes. Then I checked it against everyone's EDA
notebooks and the advisor's brief, and found ten problems. Three of them are mine.

**The big one: three rules I wrote contradict each other.**

- The Simplifier keeps a term it cannot look up, as written.
- The Verifier treats an unlooked-up definition as an unsupported claim.
- The Readability agent must reach 8th grade without changing any fact.

Walk a medication through that. The glossary defines zero drugs, so the Simplifier
correctly leaves "Hydrochlorothiazide 25 MG Oral Tablet" alone. The Verifier then flags it
as unsupported, which makes the **normal case a failure**. Then Readability is told both
to simplify that string and not to touch it.

My read: the Verifier rule is the wrong one. "I could not find a definition" is not the
same as "this claim is not supported." A term copied straight from the source is as
faithful as text gets.

## 4. The three numbers that change what we build (4 min)

All measured, all from our own notebooks.

**Zero medications.** Tested 30 terms of the kind that show up in discharge papers against
the glossary. Conditions 13 of 13. Symptoms 3 of 3. Labs 2 of 4. Procedures 2 of 5.
**Medications 0 of 5.** No metformin, no warfarin, no statin.

And MedQuAD cannot fill that gap. Gisele measured it: 65% of its answers are empty, and
the drug sources are 100% empty because the answers were redacted for copyright. So **no
dataset in this project can define a medication in plain language**. That one is for Ram.

**Our source text is three different shapes, not one.**

| Corpus | Shape | Size | Reading level |
|---|---|---|---|
| MedAESQA | prose | ~93 words | grade 11.8 |
| MTSamples | dictated notes with ALL-CAPS headers | ~465 words | grade 10.3 |
| Synthea | 16 linked CSVs, no narrative at all | 1,171 patients | not applicable |

Two consequences. The baseline simplifies MedAESQA prose while the brief talks about
simplifying clinical documents, so if we later run the pipeline on MTSamples the
comparison is not like for like. And on Synthea the Extractor is a flattener rather than
an agent, which means "Extractor removed" is not a meaningful ablation condition there.

**Only 47% of MTSamples rows are unique.** Isabella flagged this. Duplication is uneven:
Radiology is 91% unique, Cardiovascular/Pulmonary is 7%. Any sample has to deduplicate
first or we are averaging over copies of the same document.

## 5. What we have to decide (4 min)

Ten open problems are in `architecture.md` with a table showing who decides each. Three
need the group.

**Decision 1: what does the Verifier compare against?**
Right now it compares the draft to the atoms. The brief's goal is faithfulness *to the
source*. If the Extractor misses a fact, the atoms are already wrong and the Verifier
passes a draft missing it. Options: give the Verifier the source too, which costs tokens;
check extraction completeness separately; or accept it and say so in December. **Isha.**

**Decision 2: what counts as unfaithful?**
The contradiction above. My proposal is that the Verifier flags an *invented* definition,
not an *absent* one, and an unexplained term gets reported as a readability gap instead.
**Isha, Gisele, me.**

**Decision 3: do we get a medication source?**
No dataset we have can define a drug. Either we add a source outside the brief's list, or
we accept the gap and report it as a limitation. **Ram.**

## 6. Two smaller things (if there is time)

- Nobody owns PLABA. The brief calls it the simplification core and gold pairs for the
  Simplifier, and it is not in the repo. It is the natural evaluation set for both the
  Simplifier and the Readability agent.
- The model in the starter code is retired. Use `gemini-3.6-flash`, and note it returned
  503 on long prompts on 10/13 while 3.5-flash handled the same prompts. Treat the model
  as a parameter, not a constant.

---

## If someone asks

**"Why did you change my agent's instructions?"**
I updated the Simplifier and Readability prompts in `adk_pipeline.py` to spell out the
retrieval-miss behavior and the no-changing-numbers rule. Nothing about who owns what. Say
the word if you want it reverted.

**"Is the architecture wrong then?"**
The structure is right and matches the scaffold exactly. What was wrong was my assuming
source text was one thing, and three rules that cannot all hold at once. Both are
documented rather than quietly patched, because they change what other people build.

**"How sure are you about these numbers?"**
Every number here comes from a committed notebook's saved output and I checked each one
against the source file. Two numbers in the Synthea notebook look wrong to me and I
flagged them for Renita rather than changing them.
