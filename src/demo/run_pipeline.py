"""Run the FaithfulMed ADK pipeline once, on real Gemini, with live progress.

Usage (from the repo root, with the venv):
    export GEMINI_API_KEY="your-key"
    python src/demo/run_pipeline.py                                  # short built-in note
    python src/demo/run_pipeline.py src/demo/sample_note.txt         # sample discharge summary
    python src/demo/run_pipeline.py my_note.txt --rag off            # without the glossary
    python src/demo/run_pipeline.py my_note.txt --model gemini-3.6-flash

Needs the MedlinePlus ChromaDB index (python src/medlineplus_index.py build) unless --rag off.
Prints each step as it starts and finishes; the full final state is saved to results/.

Cost: about 7-13 Gemini calls per run (the loops stop early when checks pass).
"""
import argparse
import asyncio
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

REPO = Path.cwd()
sys.path[:0] = [str(REPO / "notebooks"), str(REPO / "src"), str(REPO / "src" / "agents")]

from google.adk.plugins.base_plugin import BasePlugin  # noqa: E402

# What each step is doing, shown while it runs. Containers (the pipeline and its loops)
# are left out: only steps that do work are printed.
WHAT = {
    "Extractor": "pulling out the medical facts (Gemini)",
    "Simplifier": "looking up definitions + writing the plain-language draft (Gemini)",
    "Verifier": "checking the draft against the source (Gemini)",
    "StopIfDraftFaithful": "did the draft pass? if yes, skip the Refiner (no API)",
    "Refiner": "fixing what the Verifier flagged (Gemini)",
    "Readability": "simplifying toward grade 8; skips Gemini if already there",
    "FinalCheck": "checking the final text against the source (Gemini)",
    "FinalGuard": "number + negation guards (no API)",
    "StopIfFaithful": "did the final check pass? (no API)",
    "FinalRefiner": "fixing the final text (Gemini)",
    "FinalRecheck": "re-checking the fixed text (Gemini)",
    "FinalRecheckGuard": "number + negation guards (no API)",
    "NeedsReviewFlag": "writing the faithfulness report (no API)",
}
HEARTBEAT_S = 15   # print "still waiting" this often during a long step


class Progress(BasePlugin):
    """Prints each step as it starts and finishes, plus a heartbeat while it waits."""

    def __init__(self):
        super().__init__(name="progress")
        self.t0 = time.time()
        self.started: dict[str, float] = {}
        self.seen: dict[str, int] = {}
        self.current = None

    def _clock(self) -> str:
        return f"[{time.time() - self.t0:6.1f}s]"

    async def before_agent_callback(self, *, agent, callback_context):
        if agent.name not in WHAT:
            return None
        self.seen[agent.name] = self.seen.get(agent.name, 0) + 1
        n = self.seen[agent.name]
        round_ = f" (round {n})" if n > 1 or agent.name in ("Verifier", "Refiner") else ""
        print(f"{self._clock()} ▶ {agent.name}{round_}: {WHAT[agent.name]}", flush=True)
        self.started[agent.name] = time.time()
        self.current = agent.name
        return None

    async def after_agent_callback(self, *, agent, callback_context):
        if agent.name not in WHAT:
            return None
        took = time.time() - self.started.pop(agent.name, time.time())
        note = ""
        if agent.name in ("FinalGuard", "FinalRecheckGuard"):
            v = json.loads(callback_context.state.get("final_verdict", "{}") or "{}")
            note = "  -> passed" if v.get("faithful") is True else "  -> FAILED, will fix"
        elif agent.name == "Verifier":
            from adk_pipeline import _is_faithful
            ok = _is_faithful(callback_context.state.get("verdict"))
            note = "  -> faithful" if ok else "  -> issues found"
        print(f"{self._clock()} ✓ {agent.name} done in {took:.1f}s{note}", flush=True)
        self.current = None
        return None

    async def on_model_error_callback(self, *, callback_context, llm_request, error):
        print(f"{self._clock()} ✗ model error in {callback_context.agent_name}: "
              f"{str(error)[:200]}", flush=True)
        return None

    async def heartbeat(self):
        while True:
            await asyncio.sleep(HEARTBEAT_S)
            if self.current and self.current in self.started:
                waited = time.time() - self.started[self.current]
                if waited >= HEARTBEAT_S:
                    print(f"{self._clock()}   … still on {self.current} ({waited:.0f}s so far)",
                          flush=True)


SAMPLE = ("Patient is a 58-year-old male with hypertension and type 2 diabetes. "
          "HbA1c 8.2%. Started metformin 500 mg twice daily. Denies chest pain. "
          "Follow up in 3 months.")


def main():
    if not (REPO / "notebooks" / "adk_pipeline.py").exists():
        sys.exit("Run this from the repo root (the folder that contains notebooks/ and src/).")

    # ADK reads GOOGLE_API_KEY; the agents' own Gemini clients read GEMINI_API_KEY.
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit('No API key. Run:  export GEMINI_API_KEY="your-key"   '
                 "(free at aistudio.google.com)")
    os.environ.setdefault("GEMINI_API_KEY", key)
    os.environ.setdefault("GOOGLE_API_KEY", key)
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")

    import adk_pipeline as P

    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", help="text file with the clinical note (default: a sample)")
    ap.add_argument("--rag", default="exact", choices=["exact", "exact+related", "off"])
    ap.add_argument("--model", default=P.DEFAULT_MODEL)
    args = ap.parse_args()
    source = Path(args.file).read_text() if args.file else SAMPLE

    agent = P.build_pipeline(model=args.model, rag_mode=args.rag)
    print(f"model: {args.model}   rag: {args.rag}\n\nSOURCE:\n{source}\n\nPROGRESS:")
    state = asyncio.run(run(P, agent, source))
    report(state)


async def run(P, agent, source, plugin=None):
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    plugin = plugin or Progress()
    runner = InMemoryRunner(agent=agent, app_name="faithfulmed", plugins=[plugin])
    session = await runner.session_service.create_session(
        app_name="faithfulmed", user_id="me", state=P.new_session_state(source))
    beat = asyncio.create_task(plugin.heartbeat())
    try:
        async for _ in runner.run_async(
                user_id="me", session_id=session.id,
                new_message=types.Content(role="user", parts=[types.Part(text="go")])):
            pass
    finally:
        beat.cancel()
    st = (await runner.session_service.get_session(
        app_name="faithfulmed", user_id="me", session_id=session.id)).state
    print(f"{plugin._clock()} finished", flush=True)
    return dict(st)


def report(st):
    def show(title, value):
        print(f"\n=== {title} ===\n{value}")

    show("ATOMS (Extractor)", json.dumps(json.loads(st.get("atoms", "[]")), indent=1)[:2000])
    show("DEFINITIONS USED (Simplifier RAG)", st.get("definitions", "")[:1500])
    show("DRAFT (after Verifier <-> Refiner)", st.get("draft", ""))
    show("READABILITY REPORT", st.get("readability_report", ""))
    show("FINAL TEXT (what the patient reads)", st.get("final", ""))
    show("FAITHFULNESS REPORT",
         json.dumps(json.loads(st.get("faithfulness_report", "{}")), indent=1))
    print(f"\nneeds_review: {st.get('needs_review')}")

    out = Path("results") / f"pipeline_run_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(st, indent=1, default=str))
    print(f"full state saved to {out}")


if __name__ == "__main__":
    main()
