"""
FaithfulMed — minimal Google ADK pipeline skeleton.

Shows the *shape* the SME review asked for: a deterministic DAG (SequentialAgent), not open-ended
recursion, with a single BOUNDED loop (LoopAgent, max_iterations=2) for Refiner <-> Verifier.

This is a scaffold to react to, not a finished pipeline. Each fellow fleshes out their own agent's
instruction, output schema, and tools.

Setup:
    pip install google-adk
    export GEMINI_API_KEY=...        # free key from https://aistudio.google.com
Docs: https://google.github.io/adk-docs

Note: ADK's exact class/kwarg names evolve — check the docs for your installed version. The intent
(SequentialAgent for the DAG, LoopAgent with a hard iteration cap for the refine step) is the point.
"""
from google.adk.agents import LlmAgent, SequentialAgent, LoopAgent

MODEL = "gemini-3.6-flash"  # 2.0-flash and 2.5-flash are RETIRED and return 404.
                           # Check client.models.list() if this one 404s too.

# 1) Extractor — pull clinical "atoms" (facts + types) from the source document.
extractor = LlmAgent(
    name="Extractor",
    model=MODEL,
    instruction=(
        "Extract every clinically meaningful fact from the SOURCE as a JSON list of atoms, each "
        "with {text, type} where type is one of: diagnosis, medication, lab_value, procedure, "
        "instruction. Do not add facts that are not in the SOURCE."
    ),
    output_key="atoms",
)

# 2) Simplifier — rewrite into plain language, grounded by RAG.
#
# This is the ONLY agent that queries the glossary index. See docs/architecture.md.
# Retrieval runs BEFORE this agent: src/glossary_lookup.py looks up each atom and
# format_for_prompt() renders the result into {definitions}, which lists both the
# definitions found and the terms that had none.
#
# The miss case is not an edge case. The glossary covers conditions well and contains
# no medication entries at all (0 of 5 drugs tested), while discharge summaries are
# largely medications. Inventing a drug definition is the exact failure this project
# exists to prevent, so the instruction below is explicit about it.
simplifier = LlmAgent(
    name="Simplifier",
    model=MODEL,
    instruction=(
        "Rewrite the SOURCE for a patient at an 8th-grade reading level using only the "
        "extracted {atoms}.\n\n"
        "{definitions}\n\n"
        "Rules:\n"
        "- Use the plain-language definitions above where they are given.\n"
        "- For any term listed as having NO definition, keep it exactly as written in the "
        "SOURCE. Do not guess at what it means and do not invent an explanation.\n"
        "- Add no facts that are not in {atoms}. Drop no facts that are."
    ),
    output_key="draft",
)

# 3) Verifier — LLM-as-judge: is every claim in {draft} supported by {atoms}? Flag omissions/additions.
verifier = LlmAgent(
    name="Verifier",
    model=MODEL,
    instruction=(
        "Compare {draft} against {atoms}. Return JSON: {faithful: bool, unsupported_claims: [...], "
        "omissions: [...], reading_level_ok: bool}. Be strict: an unsupported clinical claim is a failure."
    ),
    output_key="verdict",
)

# 4) Refiner — fix only what the Verifier flagged. No free rewriting.
refiner = LlmAgent(
    name="Refiner",
    model=MODEL,
    instruction=(
        "Given {draft} and {verdict}, produce a corrected draft that removes unsupported claims and "
        "restores omissions, changing nothing else. If {verdict.faithful} is true, return {draft} unchanged."
    ),
    output_key="draft",  # overwrites the draft so the loop re-verifies the corrected version
)

# Bounded refine loop: Verifier -> Refiner, at most twice. This is the ONLY loop in the system.
refine_loop = LoopAgent(
    name="RefineLoop",
    sub_agents=[verifier, refiner],
    max_iterations=2,
)

# 5) Readability — final pass to hit the target grade level and tone.
#
# This agent runs AFTER the Verifier, so nothing downstream catches a fact it breaks.
# src/readability_agent.py therefore wraps it with two guards:
#   - bounded at 2 passes (each pass is a chance to lose a fact)
#   - every pass is fact-checked against its input; a rewrite that changes a number or
#     drops a negation is REJECTED and the previous, harder text is kept
# Use that module rather than this bare agent when faithfulness matters.
readability = LlmAgent(
    name="Readability",
    model=MODEL,
    instruction=(
        "Rewrite the draft so a patient can read it easily. Target Flesch-Kincaid grade 8 "
        "or below.\n"
        "- Shorter sentences. Split anything longer than about 15 words.\n"
        "- Everyday words where an everyday word exists.\n"
        "- Every number stays exactly as written: doses, percentages, how many times a day.\n"
        "- Every 'not', 'do not', 'avoid', 'stop' stays. Never flip a negative.\n"
        "- Drop no fact. Add no fact.\n"
        "- If a medical term has no everyday equivalent, keep the term."
    ),
    output_key="final",
)

# The deterministic DAG: Extractor -> Simplifier -> (Verifier <-> Refiner, bounded) -> Readability.
faithfulmed = SequentialAgent(
    name="FaithfulMed",
    sub_agents=[extractor, simplifier, refine_loop, readability],
)

if __name__ == "__main__":
    print("FaithfulMed ADK pipeline assembled:")
    print("  Extractor -> Simplifier -> [Verifier <-> Refiner x<=2] -> Readability")
    print("Run it with an ADK Runner + session (see the ADK quickstart) once GEMINI_API_KEY is set.")
