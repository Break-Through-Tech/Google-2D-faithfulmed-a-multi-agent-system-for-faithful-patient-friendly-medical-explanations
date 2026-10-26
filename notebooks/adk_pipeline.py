"""
FaithfulMed: Google ADK pipeline, architecture v2.

See docs/architecture.md for the reasoning behind every rule below.

    Extractor -> Retriever -> Simplifier -> [Verifier <-> Refiner] x<=2 -> Readability
              -> FinalCheck -> [fix with Refiner, check again] x<=2 -> NeedsReview flag

FinalCheck = an LLM check plus deterministic guards (numbers, negations) from
src/final_check_agent.py, merged by FinalGuard after every check.

What changed from v1, and why
-----------------------------
1. The Verifier checks the draft against the SOURCE, not only the atoms.
   The brief asks for explanations "provably faithful to the source". Checking only the
   atoms meant any fact the Extractor missed was invisible to verification.

2. "No definition found" is no longer a faithfulness failure.
   The glossary defines zero medications, so an unexplained drug name is the normal
   case. A term copied verbatim from the source is faithful. It is reported as a
   readability gap (`unexplained_terms`), not as an unsupported claim.

3. A FinalCheck runs after Readability, with its own small fix loop.
   Readability rewrites the text after the main loop, so the text a patient reads needs
   its own check. If FinalCheck fails, the text goes straight to a Refiner (not back to
   the Verifier, which would only repeat the same check) and is checked again, at most
   twice. Every run ends on a check, never on an unchecked fix. If it still fails after
   two rounds, the output is flagged `needs_review` instead of looping further.

4. Retrieval is its own deterministic step, and its output stays in state.
   v1 referenced {definitions} in the Simplifier but nothing ever wrote it, which would
   have raised a KeyError at runtime. The Retriever writes it, and the Verifier can now
   see what was retrieved, so it can tell a retrieved definition from an invented one.

5. Atoms gain a `term`, a `negated` flag, and an `allergy` type.
   `term` is what the Retriever looks up ("metformin", not the whole sentence).
   `negated` stops "no acute change" from becoming "acute change".

Running it
----------
The source text must be in session state under `source` before the run:

    session = await session_service.create_session(
        app_name="faithfulmed", user_id="u1", state=new_session_state(text))

Model: gemini-2.0-flash and gemini-2.5-flash are RETIRED (404). gemini-3.6-flash
returned 503 on long prompts on 2026-10-13 while gemini-3.5-flash did not, so the model
is a parameter. Temperature 0 everywhere, so the ablation compares like with like.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import AsyncGenerator, Optional

from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from glossary_lookup import GlossaryLookup, format_for_prompt  # noqa: E402
import final_check_agent as fc  # noqa: E402

DEFAULT_MODEL = "gemini-3.5-flash"
ATOM_TYPES = ("diagnosis", "medication", "lab_value", "procedure", "instruction", "allergy")
_GEN = types.GenerateContentConfig(temperature=0.0)


def new_session_state(source: str) -> dict:
    """Initial state for one run. Every {placeholder} below must resolve from this."""
    return {"source": source}


# ------------------------------------------------------------------ 1. Extractor

EXTRACTOR_INSTRUCTION = (
    "Extract every clinically meaningful fact from the SOURCE below as a JSON list.\n"
    "Each item has exactly these keys:\n"
    "  text: the fact, worded as close to the SOURCE as possible\n"
    "  type: one of " + ", ".join(ATOM_TYPES) + "\n"
    "  term: the single medical word or phrase a patient might not know "
    "(for example 'metformin' or 'HbA1c'), or an empty string\n"
    "  negated: true if the SOURCE says the finding is absent ('no fracture', "
    "'denies chest pain'), otherwise false\n\n"
    "Rules:\n"
    "- Include allergies as type allergy. Never drop them.\n"
    "- Keep every number exactly as written: doses, lab values, dates, frequencies.\n"
    "- Skip redaction placeholders such as XXXX. They are not values.\n"
    "- Do not add any fact that is not in the SOURCE.\n"
    "- Return only the JSON list, no other text.\n\n"
    "SOURCE:\n{source}"
)


# ------------------------------------------------------------------ 2. Retriever

def parse_atoms(raw) -> list[dict]:
    """Parse the Extractor's output. Tolerates ```json fences and stray prose."""
    if isinstance(raw, list):
        return [a for a in raw if isinstance(a, dict)]
    text = str(raw or "")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
    m = re.search(r"\[.*\]", text, flags=re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [a for a in data if isinstance(a, dict)]


class Retriever(BaseAgent):
    """Deterministic lookup step. No LLM, no quota cost for exact matches.

    The ONLY place in the pipeline that queries the glossary. Writes:
      definitions        text block for the Simplifier and Verifier prompts
      unexplained_terms  terms with no definition, as a JSON list
    """

    lookup: Optional[GlossaryLookup] = None
    model_config = {"arbitrary_types_allowed": True}

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        atoms = parse_atoms(ctx.session.state.get("atoms"))
        queries = [{"text": (a.get("term") or "").strip()} for a in atoms]
        queries = [q for q in queries if q["text"]]

        if self.lookup is None:
            definitions = ("NO GLOSSARY CONFIGURED. Treat every medical term as having no "
                           "definition: keep it exactly as written in the SOURCE.")
            missing = [q["text"] for q in queries]
        else:
            results = self.lookup.lookup_atoms(queries)
            definitions = format_for_prompt(results) or "No medical terms to look up."
            missing = [r.term for r in results.values() if not r.found]

        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(state_delta={
                "definitions": definitions,
                "unexplained_terms": json.dumps(missing),
            }),
        )


# ------------------------------------------------------------------ 3. Simplifier

SIMPLIFIER_INSTRUCTION = (
    "Rewrite the SOURCE for a patient at an 8th-grade reading level.\n\n"
    "SOURCE:\n{source}\n\n"
    "EXTRACTED FACTS (every one must appear in your rewrite):\n{atoms}\n\n"
    "{definitions}\n\n"
    "Rules:\n"
    "- Use the plain-language definitions above where they are given.\n"
    "- For a term with no definition, keep it exactly as written. Do not guess what it "
    "means.\n"
    "- Keep every negation. 'No fracture' must not become 'fracture'.\n"
    "- State lab values exactly as the SOURCE does. Do not say whether a value is high, "
    "low or normal unless the SOURCE says so.\n"
    "- Add no fact that is not in the SOURCE. Drop no fact that is.\n"
    "- Plain prose only."
)


# ------------------------------------------------------------------ 4. Verifier / FinalCheck

def _verifier_instruction(text_key: str) -> str:
    return (
        "You are checking a patient-friendly explanation for faithfulness.\n\n"
        "SOURCE (the ground truth):\n{source}\n\n"
        "EXTRACTED FACTS (a checklist, may be incomplete):\n{atoms}\n\n"
        "RETRIEVED DEFINITIONS (allowed background, not from the SOURCE):\n{definitions}\n\n"
        "EXPLANATION TO CHECK:\n{" + text_key + "}\n\n"
        "Return a JSON object with these keys:\n"
        "  faithful: true only if unsupported_claims and omissions are both empty\n"
        "  unsupported_claims: statements in the explanation that are supported by NEITHER "
        "the SOURCE NOR the retrieved definitions\n"
        "  omissions: facts in the SOURCE that the explanation leaves out\n"
        "  negation_errors: findings whose negation was lost or flipped\n"
        "  unexplained_terms: medical terms left without a plain explanation\n"
        "  reading_level_ok: true if a typical 8th grader could read it\n\n"
        "Important: a medical term copied from the SOURCE without explanation is NOT an "
        "unsupported claim. It is faithful. List it under unexplained_terms only.\n"
        "Treat a negation error as an unsupported claim as well.\n"
        "Return only the JSON object."
    )


def parse_verdict(raw) -> dict:
    """Parse a Verifier or FinalCheck verdict. Tolerates ```json fences.

    An unreadable verdict counts as NOT faithful. If we cannot tell whether the text is
    faithful, the safe answer is to fix it or flag it, never to pass it.
    """
    if isinstance(raw, dict):
        return raw
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(raw or "").strip(), flags=re.M)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return {"faithful": False, "parse_error": True}


def _is_faithful(raw) -> bool:
    return parse_verdict(raw).get("faithful") is True


class FinalGuard(BaseAgent):
    """Runs right after every final check. No LLM, no quota.

    Adds the FinalCheck agent's deterministic guards (src/final_check_agent.py) to the
    model's verdict: a changed or dropped number, an invented number, a flipped negation.
    Then recomputes `faithful` from the merged lists, so the model cannot pass a text
    while listing a problem, and cannot pass a changed dose it did not notice.
    An unreadable verdict stays a fail.
    """

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        llm = fc.parse_verdict(state.get("final_verdict"))
        if llm is None:
            merged = {"faithful": False, "parse_error": True}
        else:
            guards = fc.deterministic_check(state.get("source", ""),
                                            parse_atoms(state.get("atoms")),
                                            state.get("final", ""))
            merged = fc.merge(llm, guards)
        yield Event(author=self.name, invocation_id=ctx.invocation_id,
                    actions=EventActions(state_delta={"final_verdict": json.dumps(merged)}))


class StopIfFaithful(BaseAgent):
    """Ends the final fix loop as soon as the latest final_verdict passes. No LLM."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        passed = _is_faithful(ctx.session.state.get("final_verdict"))
        yield Event(author=self.name, invocation_id=ctx.invocation_id,
                    actions=EventActions(escalate=passed))


class NeedsReviewFlag(BaseAgent):
    """Last step. Records whether the final text still failed after the fix loop, and
    writes the faithfulness report that goes to the user with the explanation. No LLM."""

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        passed = _is_faithful(state.get("final_verdict"))
        verdict = fc.parse_verdict(state.get("final_verdict"))
        if verdict is not None:
            verdict["faithful"] = passed
        report = fc.faithfulness_report(verdict, text=state.get("final", ""))
        yield Event(author=self.name, invocation_id=ctx.invocation_id,
                    actions=EventActions(state_delta={
                        "needs_review": not passed,
                        "faithfulness_report": json.dumps(report)}))


# ------------------------------------------------------------------ 5. Refiner

REFINER_INSTRUCTION = (
    "Fix the explanation using the verdict. Change nothing else.\n\n"
    "SOURCE:\n{source}\n\n"
    "EXPLANATION:\n{draft}\n\n"
    "VERDICT:\n{verdict}\n\n"
    "Rules:\n"
    "- Remove every unsupported claim.\n"
    "- Add back every omission, worded plainly, using only the SOURCE.\n"
    "- Fix every negation error.\n"
    "- Do not rewrite sentences the verdict did not flag.\n"
    "- If the verdict says faithful is true, return the explanation unchanged.\n"
    "- Return only the explanation text."
)


FINAL_REFINER_INSTRUCTION = (
    "Fix the final patient explanation using the verdict. Change nothing else.\n\n"
    "SOURCE:\n{source}\n\n"
    "EXPLANATION:\n{final}\n\n"
    "VERDICT:\n{final_verdict}\n\n"
    "Rules:\n"
    "- Remove every unsupported claim.\n"
    "- Add back every omission, worded plainly, using only the SOURCE.\n"
    "- Fix every negation error.\n"
    "- Keep it easy to read: short sentences, everyday words.\n"
    "- Do not add explanations, definitions or background that are not in the SOURCE.\n"
    "- Do not rewrite sentences the verdict did not flag.\n"
    "- Return only the explanation text."
)


# ------------------------------------------------------------------ 6. Readability

READABILITY_INSTRUCTION = (
    "Rewrite the explanation so a patient can read it easily. Target Flesch-Kincaid "
    "grade 8 or below.\n\n"
    "EXPLANATION:\n{draft}\n\n"
    "- Shorter sentences. Split anything longer than about 15 words.\n"
    "- Everyday words where an everyday word exists.\n"
    "- Every number stays exactly as written: doses, values, how many times a day.\n"
    "- Every 'not', 'no', 'do not', 'avoid', 'stop' stays. Never flip a negative.\n"
    "- Drop no fact. Add no fact.\n"
    "- If a medical term has no everyday equivalent, keep the term.\n"
    "- Return only the rewritten text."
)


# ------------------------------------------------------------------ assembly

def build_pipeline(model=DEFAULT_MODEL, lookup: Optional[GlossaryLookup] = None,
                   max_refine_rounds: int = 2, max_final_rounds: int = 2) -> SequentialAgent:
    """Build a fresh pipeline. Fresh every call, because an ADK agent can only have one
    parent, so the ablation needs a new instance per configuration."""
    def llm(name, instruction, key):
        return LlmAgent(name=name, model=model, instruction=instruction,
                        output_key=key, generate_content_config=_GEN)

    return SequentialAgent(
        name="FaithfulMed",
        sub_agents=[
            llm("Extractor", EXTRACTOR_INSTRUCTION, "atoms"),
            Retriever(name="Retriever", lookup=lookup),
            llm("Simplifier", SIMPLIFIER_INSTRUCTION, "draft"),
            LoopAgent(
                name="RefineLoop",
                max_iterations=max_refine_rounds,
                sub_agents=[
                    llm("Verifier", _verifier_instruction("draft"), "verdict"),
                    llm("Refiner", REFINER_INSTRUCTION, "draft"),
                ],
            ),
            llm("Readability", READABILITY_INSTRUCTION, "final"),
            # Check the text the patient will actually read.
            llm("FinalCheck", _verifier_instruction("final"), "final_verdict"),
            FinalGuard(name="FinalGuard"),
            # If it failed: fix with a Refiner and check again, at most twice. Each round
            # starts with StopIfFaithful, so a passing check ends the loop, and every round
            # ends on a check, never on an unchecked fix.
            LoopAgent(
                name="FinalFixLoop",
                max_iterations=max_final_rounds,
                sub_agents=[
                    StopIfFaithful(name="StopIfFaithful"),
                    llm("FinalRefiner", FINAL_REFINER_INSTRUCTION, "final"),
                    llm("FinalRecheck", _verifier_instruction("final"), "final_verdict"),
                    FinalGuard(name="FinalRecheckGuard"),
                ],
            ),
            # Still failing after the cap: ship it flagged, never loop further.
            NeedsReviewFlag(name="NeedsReviewFlag"),
        ],
    )


# Default instance for `adk web`. No glossary attached here; attach one with
# build_pipeline(lookup=GlossaryLookup(topics_csv, synonyms_csv)).
faithfulmed = build_pipeline()
root_agent = faithfulmed

if __name__ == "__main__":
    print("FaithfulMed v2:")
    labels = {"RefineLoop": "[Verifier <-> Refiner] x<=2",
              "FinalFixLoop": "[fix + recheck + guard] x<=2"}
    print("  " + " -> ".join(labels.get(a.name, a.name) for a in faithfulmed.sub_agents))
