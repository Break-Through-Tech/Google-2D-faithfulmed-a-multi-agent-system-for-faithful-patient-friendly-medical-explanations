"""Walk-through demo of the Refiner agent.

Usage
-----
    python src/demo_refiner.py

If `GEMINI_API_KEY` is set (via `export` or a `.env` file at the repo root),
the demo runs live against Gemini. Otherwise it runs with a stubbed
generate_fn so the shape of a refine is still visible.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import refiner_agent as r


SOURCE = ("Patient with hypertension. Started metformin 500 mg twice daily. "
          "HbA1c 8.2%. Allergic to penicillin. No chest pain.")

GOOD = ("You have high blood pressure. Take metformin 500 mg twice daily. "
        "Your HbA1c was 8.2%. You are allergic to penicillin. "
        "You have no chest pain.")

FAITHFUL = {"faithful": True, "unsupported_claims": [], "omissions": [],
            "negation_errors": [], "unexplained_terms": ["metformin"],
            "reading_level_ok": True}


def _load_dotenv(path: Path) -> None:
    """Minimal KEY=VALUE loader so the demo has no extra dependency."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, val = line.partition("=")
        os.environ.setdefault(k.strip(), val.strip().strip('"').strip("'"))


def _build_generate_fn():
    """Return (generate_fn, mode_label). Live if the key is set, stub otherwise."""
    _load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        def stub(prompt: str) -> dict:
            # Targeted fix: drop the extra claim, restore the dose / negation.
            # The fact_guard still runs, which is the point of the merge.
            return {"text": GOOD, "error": None, "model_used": "stub"}
        return stub, "STUB (no GEMINI_API_KEY found)"
    from google import genai
    client = genai.Client(api_key=key)
    return r.make_generate_fn(client), "LIVE (Gemini)"


def _show(title: str, result: r.RefinerResult) -> None:
    print(f"\n-- {title} --")
    print(f"  rounds_used    : {result.rounds_used}")
    print(f"  stopped_reason : {result.stopped_reason}")
    print(f"  model_used     : {result.model_used}")
    if result.rejected_rounds:
        print(f"  rejected       : {result.rejected_rounds}")
    if result.error:
        print(f"  error          : {result.error}")
    print(f"  original       : {result.original}")
    print(f"  final          : {result.final}")


def main() -> int:
    generate_fn, mode = _build_generate_fn()
    print(f"Refiner demo - mode: {mode}")

    _show("1. Already faithful (no model call)",
          r.run(SOURCE, GOOD, FAITHFUL, generate_fn))

    extra = GOOD + " Call a cardiologist within 48 hours."
    unsupp = {**FAITHFUL, "faithful": False,
              "unsupported_claims": ["Call a cardiologist within 48 hours"],
              "unexplained_terms": []}
    _show("2. Drop an unsupported claim",
          r.run(SOURCE, extra, unsupp, generate_fn))

    omitted = GOOD.replace("500 mg twice daily", "as needed")
    omission_v = {**FAITHFUL, "faithful": False,
                  "omissions": ["metformin 500 mg twice daily"],
                  "unexplained_terms": []}
    _show("3. Restore an omitted dose from the source",
          r.run(SOURCE, omitted, omission_v, generate_fn))

    flipped = GOOD.replace("You have no chest pain.", "You have chest pain.")
    neg_v = {**FAITHFUL, "faithful": False,
             "negation_errors": ["'chest pain' asserted without its negation"],
             "unexplained_terms": []}
    _show("4. Restore a dropped negation",
          r.run(SOURCE, flipped, neg_v, generate_fn))

    def invent(prompt):
        return {"text": GOOD.replace("500", "800"), "error": None, "model_used": "stub-unsafe"}
    _show("5. Unsafe rewrite (invented dose) is rejected",
          r.run(SOURCE, omitted, omission_v, invent))

    def verify_fn(draft):
        if "cardiologist" in draft:
            return unsupp
        return FAITHFUL
    _show("6. Bounded loop stops once the draft is faithful",
          r.run_loop(SOURCE, extra, generate_fn, verify_fn, max_rounds=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
