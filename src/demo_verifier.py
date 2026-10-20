"""Walk-through demo of the Verifier agent.

Usage
-----
    python src/demo_verifier.py

If `GEMINI_API_KEY` is set (via `export` or a `.env` file at the repo root),
the demo runs live against Gemini. Otherwise it runs with a stubbed
generate_fn so the shape of the verdict is still visible.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import verifier_agent as v


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
            # Pretend the LLM returned nothing problematic; the deterministic
            # guards still fire, which is the point of the merge.
            return {"text": json.dumps({
                "unsupported_claims": [], "omissions": [],
                "negation_errors": [], "unexplained_terms": ["lisinopril"],
            }), "error": None, "model_used": "stub"}
        return stub, "STUB (no GEMINI_API_KEY found)"
    from google import genai
    client = genai.Client(api_key=key)
    return v.make_generate_fn(client), "LIVE (Gemini)"


def _show(title: str, result: v.VerifierResult) -> None:
    print(f"\n── {title} ──")
    print(f"  faithful           : {result.verdict['faithful']}")
    print(f"  unsupported_claims : {result.verdict['unsupported_claims']}")
    print(f"  omissions          : {result.verdict['omissions']}")
    print(f"  negation_errors    : {result.verdict['negation_errors']}")
    print(f"  unexplained_terms  : {result.verdict['unexplained_terms']}")
    print(f"  reading_level_ok   : {result.verdict['reading_level_ok']}")
    print(f"  model_used         : {result.model_used}")
    if result.error:
        print(f"  error              : {result.error}")


def main() -> int:
    generate_fn, mode = _build_generate_fn()
    print(f"Verifier demo — mode: {mode}")

    source = "Start lisinopril 10 mg daily for high blood pressure."
    atoms = [
        {"term": "lisinopril",  "text": "lisinopril",            "negated": False},
        {"term": "10 mg daily", "text": "lisinopril 10 mg daily", "negated": False},
    ]

    _show("1. Clean draft (verbatim term → unexplained, not unsupported)",
          v.run(source, atoms,
                "Take lisinopril, 10 mg once a day, to lower your blood pressure.",
                generate_fn))

    _show("2. Wrong dose (deterministic number guard catches this)",
          v.run(source, atoms,
                "Take lisinopril, 20 mg once a day, to lower your blood pressure.",
                generate_fn))

    _show("3. Flipped negation (deterministic negation guard catches this)",
          v.run("Do not take ibuprofen with this medication.",
                [{"term": "ibuprofen", "negated": True}],
                "You can take ibuprofen with this medication.",
                generate_fn))
    return 0


if __name__ == "__main__":
    sys.exit(main())
