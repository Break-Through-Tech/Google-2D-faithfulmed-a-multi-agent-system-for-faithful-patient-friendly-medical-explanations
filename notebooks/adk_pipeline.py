"""
FaithfulMed: Google ADK pipeline, architecture v2.

See docs/architecture.md for the reasoning behind every rule below.

    Extractor -> Simplifier (+RAG) -> [Verifier <-> Refiner] x<=2 -> Readability
              -> FinalCheck -> [fix with Refiner, check again] x<=2 -> NeedsReview flag

Simplifier = src/agents/simplifier.py: looks up MedlinePlus definitions in the ChromaDB
index itself (the RAG step that used to be a separate Retriever), rewrites the atoms, and
runs its own coverage / number / negation checks.
Readability = src/agents/readability_agent.py: bounded rewrite passes toward the target
grade, rejecting any pass that changes a number or drops a negation.
FinalCheck = an LLM check plus deterministic guards (numbers, negations) from
src/agents/final_check_agent.py, merged by FinalGuard after every check.

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

4. What was retrieved stays in state.
   Retrieval now happens inside the Simplifier, which writes the definitions it used to
   state["definitions"]. The Verifier and FinalCheck read them, so they can tell a
   retrieved definition from an invented one.

5. Atoms gain a `term`, a `negated` flag, and an `allergy` type.
   `term` is what the Simplifier looks up ("metformin", not the whole sentence).
   `negated` stops "no acute change" from becoming "acute change".

Running it
----------
The source text must be in session state under `source` before the run:

    session = await session_service.create_session(
        app_name="faithfulmed", user_id="u1", state=new_session_state(text))

A real run needs GEMINI_API_KEY in the environment and, unless rag_mode="off", the
ChromaDB index built (python src/medlineplus_index.py build).

Model: gemini-2.0-flash and gemini-2.5-flash are RETIRED (404). gemini-3.6-flash
returned 503 on long prompts on 2026-10-13 while gemini-3.5-flash did not, so the model
is a parameter. Temperature 0 everywhere, so the ablation compares like with like.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Optional

from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "agents"))
import final_check_agent as fc  # noqa: E402
import readability_agent as _R  # noqa: E402
import simplifier as _S  # noqa: E402

DEFAULT_MODEL = "gemini-3.5-flash"
_GEN = types.GenerateContentConfig(temperature=0.0)


def new_session_state(source: str) -> dict:
    """Initial state for one run. Every {placeholder} below must resolve from this."""
    return {"source": source}


# ------------------------------------------------------------------ 1. Extractor

# The Extractor lives in src/agents/extractor_agent.py (owner: Vera). The prompt is built from
# state["source"] by an InstructionProvider (citations stripped, MTSamples sections
# tagged), and an after_agent_callback runs the deterministic guards on the reply, so
# state["atoms"] always holds checked atoms in the schema both readers expect.
import extractor_agent as _X  # noqa: E402

EXTRACTOR_INSTRUCTION = _X.PROMPT          # kept for anyone importing the old name
ATOM_TYPES = _X.ATOM_TYPES


# ------------------------------------------------------------------ 2. Atom parsing

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


# ------------------------------------------------------------------ 3. Simplifier (+RAG)

class Simplifier(BaseAgent):
    """Runs src/agents/simplifier.py on the Extractor's atoms. Owner: Gisele.

    The glossary lookup (RAG) happens inside simplify(): exact matches against the
    MedlinePlus ChromaDB index, no API call. Writes:
      draft              the explanation, for the Verifier
      definitions        the glossary entries it used, for the Verifier and FinalCheck
      unexplained_terms  atom terms with no glossary entry, as a JSON list
      simplifier_report  its own checks (uncovered atoms, dropped numbers...) and stats

    simplify() is synchronous and calls Gemini itself, so it runs in a worker thread.
    """

    models: Optional[tuple] = None        # None -> simplifier.DEFAULT_MODELS fallback chain
    rag_mode: str = "exact"
    retriever: Any = None                 # None -> the default ChromaDB index
    llm_fn: Optional[Callable] = None     # (system, user) -> JSON text; for tests/other models
    model_config = {"arbitrary_types_allowed": True}

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        atoms = _S.load_atoms(parse_atoms(ctx.session.state.get("atoms")))
        result = await asyncio.to_thread(
            _S.simplify, atoms, retriever=self.retriever, rag_mode=self.rag_mode,
            model=self.models, temperature=0.0, llm_fn=self.llm_fn)

        glossary = result["glossary_used"]
        defined = {i for g in glossary for i in g["atom_ids"]}
        missing = [a["term"] for a in atoms if a["term"] and a["id"] not in defined]
        definitions = (_S.format_glossary(glossary) if glossary else
                       "No glossary definitions were used.")
        report = {k: result[k] for k in ("sentences", "checks", "warnings", "stats")}
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(state_delta={
                "draft": result["explanation"],
                "definitions": definitions,
                "unexplained_terms": json.dumps(missing),
                "simplifier_report": json.dumps(report),
            }),
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

    Adds the FinalCheck agent's deterministic guards (src/agents/final_check_agent.py) to the
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

class Readability(BaseAgent):
    """Runs src/agents/readability_agent.py on the verified draft. Owner: Edwin (Joao).

    At most `max_passes` rewrites toward `target_grade`. A pass that changes a number or
    loses a negation is rejected and the previous text kept, so this step cannot undo the
    Verifier's work. If the draft already meets the target, no model call is made. Writes:
      final               the text the FinalCheck reads
      readability_report  grades before/after, passes used, rejected passes, any error
    """

    models: Optional[tuple] = None          # None -> readability_agent.DEFAULT_MODELS
    generate_fn: Optional[Callable] = None  # prompt -> {"text", "error"}; for tests/other models
    target_grade: float = _R.TARGET_GRADE
    max_passes: int = 2
    model_config = {"arbitrary_types_allowed": True}

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        generate_fn = self.generate_fn
        if generate_fn is None:
            from google import genai
            generate_fn = _R.make_generate_fn(genai.Client(),
                                              models=self.models or _R.DEFAULT_MODELS)
        res = await asyncio.to_thread(_R.run, ctx.session.state.get("draft", ""),
                                      generate_fn, target=self.target_grade,
                                      max_passes=self.max_passes)
        report = res.as_dict()
        report.pop("original")
        report.pop("final")
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            actions=EventActions(state_delta={
                "final": res.final,
                "readability_report": json.dumps(report),
            }),
        )


# ------------------------------------------------------------------ assembly

def build_pipeline(model=DEFAULT_MODEL, max_refine_rounds: int = 2, max_final_rounds: int = 2,
                   rag_mode: str = "exact", retriever: Any = None,
                   target_grade: float = _R.TARGET_GRADE,
                   simplifier_llm_fn: Optional[Callable] = None,
                   readability_generate_fn: Optional[Callable] = None) -> SequentialAgent:
    """Build a fresh pipeline. Fresh every call, because an ADK agent can only have one
    parent, so the ablation needs a new instance per configuration.

    model: a Gemini model name, used by every agent so the ablation compares like with
        like. The Simplifier and Readability call Gemini through their own modules; a
        non-Gemini model (an ADK BaseLlm such as LiteLlm) is not passed through to them
        yet, so for one of those simplifier_llm_fn and readability_generate_fn are
        required.
    rag_mode: "exact" (default), "exact+related" or "off" (the no-glossary ablation).
    retriever: None opens the default MedlinePlus ChromaDB index.
    """
    if not isinstance(model, str) and (simplifier_llm_fn is None
                                       or readability_generate_fn is None):
        raise ValueError(
            "The Simplifier and Readability agents only call Gemini by name. With a "
            "non-string model, pass simplifier_llm_fn and readability_generate_fn too.")
    models = (model,) if isinstance(model, str) else None

    def llm(name, instruction, key):
        return LlmAgent(name=name, model=model, instruction=instruction,
                        output_key=key, generate_content_config=_GEN)

    return SequentialAgent(
        name="FaithfulMed",
        sub_agents=[
            LlmAgent(name="Extractor", model=model, instruction=_X.adk_instruction,
                     output_key="atoms", generate_content_config=_GEN,
                     after_agent_callback=_X.adk_after_extractor),
            Simplifier(name="Simplifier", models=models, rag_mode=rag_mode,
                       retriever=retriever, llm_fn=simplifier_llm_fn),
            LoopAgent(
                name="RefineLoop",
                max_iterations=max_refine_rounds,
                sub_agents=[
                    llm("Verifier", _verifier_instruction("draft"), "verdict"),
                    llm("Refiner", REFINER_INSTRUCTION, "draft"),
                ],
            ),
            Readability(name="Readability", models=models, target_grade=target_grade,
                        generate_fn=readability_generate_fn),
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


# Default instance for `adk web`. The Simplifier opens the ChromaDB index on first use;
# build_pipeline(rag_mode="off") runs without it.
faithfulmed = build_pipeline()
root_agent = faithfulmed

if __name__ == "__main__":
    print("FaithfulMed v2:")
    labels = {"RefineLoop": "[Verifier <-> Refiner] x<=2",
              "FinalFixLoop": "[fix + recheck + guard] x<=2"}
    print("  " + " -> ".join(labels.get(a.name, a.name) for a in faithfulmed.sub_agents))
