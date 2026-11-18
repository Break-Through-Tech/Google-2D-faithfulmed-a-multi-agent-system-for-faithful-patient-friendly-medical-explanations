# FaithfulMed: A Multi-Agent System for Faithful, Patient-Friendly Medical Explanations

**Company / Org:** Google  
**Challenge Advisor:** Ram Seshadri  
**Program:** Break Through Tech AI Studio — Fall 2026  
**AI Studio Coach:** Rohil Saraf ([rohil.saraf@breakthroughtech.org](mailto:rohil.saraf@breakthroughtech.org))

---

### 👥 **Team Members**

**Example:**

| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
| Taylor Nguyen    | @taylornguyen | Data exploration, visualization, overall project coordination            |
| Jordan Ramirez   | @jramirez     | Data collection, exploratory data analysis (EDA), dataset documentation  |
| Amina Hassan     | @aminahassan  | Data preprocessing, feature engineering, data validation                 |
| Priya Mehta      | @pmehta       | Model selection, hyperparameter tuning, model training and optimization  |
| Chris Park       | @chrispark    | Model evaluation, performance analysis, results interpretation           |

---

## 🎯 **Project Highlights**

- Multi-agent pipeline on **Google ADK** that turns clinical source text into patient-friendly explanations that stay faithful to the source.
- Agents: **Extractor → Retriever → Simplifier → [Verifier ↔ Refiner] ×≤2 → Readability → FinalCheck** (with a short fix loop; still failing → `needs_review`).
- RAG over a **MedlinePlus** lay-language glossary (exact synonym match first, ChromaDB vector search only on miss).
- Deterministic **number / negation guards** on Extractor, Simplifier, Verifier, Refiner, Readability, and FinalCheck so patient-safety failures do not rely on the LLM alone.
- Evaluation targets: ≤8th-grade Flesch-Kincaid readability, high faithfulness vs. human labels (MedAESQA), and ablations vs. a single-LLM baseline.

---

## 👩🏽‍💻 **Setup and Installation**

```bash
git clone <repo-url>
cd Google-2D-faithfulmed-a-multi-agent-system-for-faithful-patient-friendly-medical-explanations

pip install -r requirements.txt
export GEMINI_API_KEY=...   # https://aistudio.google.com — or a root `.env`
```

**Datasets:** see `data/README.md`. Large corpora are git-ignored; download locally. Start with MedAESQA for evaluation and MedlinePlus cleaned CSVs under `data/cleaned-medlineplus/` for RAG.

**ADK pipeline (notebook):**

```bash
# from repo root, with GEMINI_API_KEY set
python -c "from notebooks.adk_pipeline import build_pipeline; print(build_pipeline())"
```

Or open `notebooks/adk_pipeline.py` / use `adk web` with `root_agent` exported there.

**Standalone agents & retrieval (`src/`):** see [`src/README.md`](src/README.md).

```bash
python src/demo_verifier.py
python src/demo_refiner.py
python src/test_pipeline_offline.py   # full chain, no API key
```

**Useful notebooks:** `notebooks/baseline_single_gemini.ipynb` (single-LLM baseline),
`notebooks/final_check_agent.ipynb`, EDA under `notebooks/exploratory-data-analysis/`,
eval helpers in `notebooks/eval_harness.py` and `src/eval_extractor.py`.

**MedlinePlus index:**

```bash
python src/medlineplus_index.py build
python src/medlineplus_index.py validate
python src/medlineplus_index.py query "high blood pressure"
```

---

## 🏗️ **Project Overview**

This is a Break Through Tech AI Studio challenge project hosted with **Google**. Clinical notes, labs, and discharge instructions are written for clinicians; patients often misunderstand them. Generative models can rewrite that text, but unfaithful or hallucinated clinical details are unsafe.

FaithfulMed asks whether a **pipeline of specialized agents** produces explanations that are both more **readable** and more **faithful** than a single well-prompted LLM — and which agents actually matter for each goal.

Built on Google's stack: **Gemini**, optional **Gemma** / open models via LiteLLM for the leaderboard, and the **Agent Development Kit (ADK)**.

Architecture contracts live in [`docs/architecture.md`](docs/architecture.md).

---

## 📊 **Data Exploration**

Primary datasets (all public; no PHI/PII):

| Dataset | Role |
|---------|------|
| **MedAESQA** | Faithfulness eval + Verifier calibration (human labels) |
| **MTSamples** | Source clinical reports to simplify |
| **PLABA** | Professional ↔ plain-language gold pairs |
| **MedQuAD** | Consumer-health Q&A / tone |
| **Synthea** | Synthetic FHIR stress tests (flatten before LLM use) |
| **MedlinePlus glossary** | RAG knowledge base for plain definitions |

EDA and cleaning notes live under `notebooks/exploratory-data-analysis/` and `data/README.md`.

---

## 🧠 **Model Development**

**Orchestration:** Google ADK `SequentialAgent` + bounded `LoopAgent` (Verifier ↔ Refiner, max 2 rounds), then FinalCheck with its own ≤2 fix rounds and a `needs_review` flag.

**Default model:** `gemini-3.5-flash` at temperature 0 (see `notebooks/adk_pipeline.py`). Model is parameterized for ablations and the multi-model leaderboard.

**Retrieval:** `glossary_lookup.py` — exact match on MedlinePlus synonyms, then optional ChromaDB (`medlineplus_index.py`, embeddings via `gemini-embedding-001`).

**Guarded agents in `src/`:**
- `extractor_agent.py` — SOURCE → typed atoms; rejects invented numbers/facts
- `simplifier.py` — atoms → draft with MedlinePlus RAG + coverage/number/negation checks
- `verifier_agent.py` — LLM verdict + deterministic number/negation guards
- `refiner_agent.py` — targeted fixes; rejects unsafe rewrites
- `readability_agent.py` — ≤8th-grade rewrite with post-pass self-checks
- `final_check_agent.py` — last gate on patient-facing text; fix loop or `needs_review`

---

## 📈 **Results & Key Findings**

*(Fill in as evaluation and ablations land.)*

Target success criteria from the challenge brief:
- ≥80% of outputs at ≤8th-grade FK
- ≥85% factual fidelity on the annotated test set
- <10% clinically meaningful unsupported claims
- ≥15% absolute faithfulness gain vs. single-agent baseline
- Verifier–human agreement ≥80% (κ ≥ 0.6)

Offline wiring (no API key): `python src/test_pipeline_offline.py`.

---

## 🚀 **Next Steps**

- Complete ablation table (drop one agent at a time).
- Calibrate Verifier against MedAESQA human labels.
- Multi-model leaderboard (Gemini + Gemma + one external open model).
- Hosted demo with per-agent traces (HF Spaces or Cloud Run).
- Technical report and final presentation.

---

## 📝 **License**

License TBD — confirm with the Challenge Advisor before publishing.

---

## 📄 **References**

- Challenge brief: `Challenge-Project-Overview.md`
- Architecture: `docs/architecture.md` · walkthrough: `docs/architecture-walkthrough.md`
- Extractor notes: `docs/extractor.md`
- MedAESQA: https://osf.io/ydbzq · https://github.com/deepaknlp/MedAESQA
- MedlinePlus: https://medlineplus.gov/
- Google ADK: https://google.github.io/adk-docs/

---

## 🙏 **Acknowledgements**

Thanks to Challenge Advisor Ram Seshadri, AI Studio Coach Rohil Saraf, Google hosts, and Break Through Tech AI for support on this project.
