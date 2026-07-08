# OncoAgent — Multi-Agent AI System for Oncology Visit Preparation

A small, fully-runnable demonstration of agentic AI system design applied to
health: a pipeline of specialized agents that turns a patient's free-text
notes into a safe, structured, source-grounded visit-preparation summary —
with PII redaction, rule-based safety triage, retrieval-augmented guidance,
and a generator/critic self-revision loop.

> **This is a portfolio/demo project, not a medical device.** The knowledge
> base is synthetic and written for demonstration only — it must never be
> used for real clinical decisions. See [Disclaimer](#disclaimer).

## Why this project

Most "AI agent" demos are generic task-runners. This one is built around a
real, narrow health workflow — the few minutes before an oncology follow-up
visit, where patients are asked to self-report symptoms and often forget
their most important questions — and shows the agent-engineering patterns
that actually matter for that kind of workflow:

- **Tool/agent decomposition** — each agent has one job and a narrow
  interface, so the pipeline is testable and swappable piece by piece.
- **Deterministic safety agent, not a model** — the red-flag triage step is
  rule-based on purpose. When a missed detection has real consequences, an
  auditable rule set beats an opaque model call.
- **Privacy-first ordering** — redaction happens *before* any other agent
  or log sees the text, not as an afterthought.
- **Grounded generation** — the summary agent is only allowed to phrase
  facts the pipeline already extracted; it can't introduce new claims.
- **Generator/critic loop** — a lightweight reflection pattern: the critic
  checks the draft against guardrails (no overreaching claims, urgent cases
  must tell the patient to seek care) and sends it back for revision, bounded
  to a couple of retries.
- **Explainable execution trace** — every agent step is logged, so you can
  see exactly why a given output was produced.
- **Pluggable backends everywhere** — retrieval, redaction, and generation
  all have a working offline default *and* a real Hugging Face / LLM API
  integration point behind an environment variable, so the repo runs out of
  the box on `git clone` but also demonstrates real model integration.

## Architecture

```
 free-text notes
       |
       v
 ┌─────────────────┐   redact PII first, before anything else sees the text
 │ RedactionAgent   │   (regex fallback, or HF NER model via env flag)
 └───────┬─────────┘
         v
 ┌─────────────────┐
 │ IntakeAgent      │   structure symptoms / medications / cancer type
 └───────┬─────────┘
         v
 ┌─────────────────┐   rule-based, not model-based — auditable red-flag
 │ SafetyAgent      │   detection (routine / soon / urgent / emergency)
 └───────┬─────────┘
         v
 ┌─────────────────┐   RAG over a small synthetic guideline knowledge base
 │ LiteratureAgent  │   (TF-IDF by default, or HF sentence-embeddings)
 └───────┬─────────┘
         v
 ┌─────────────────┐ <───────────┐
 │ SummaryAgent     │             │  bounded revision loop
 └───────┬─────────┘             │
         v                        │
 ┌─────────────────┐   fails? ────┘
 │ CriticAgent      │
 └───────┬─────────┘
         v  passes
   VisitSummary + full agent trace
```

All of this is coordinated by `OncoAgentOrchestrator` (`oncoagent/orchestrator.py`),
which is plain Python — no agent framework dependency — so the control flow
is fully visible and easy to reason about or extend.

## Project layout

```
oncoagent/
  schemas.py            Pydantic models shared across agents (FHIR-ish shapes)
  llm_provider.py        Pluggable LLM backend: template / Anthropic / OpenAI
  knowledge_base.py       Retrieval: TF-IDF default, optional HF embeddings
  agents/
    redaction_agent.py    PII redaction: regex default, optional HF NER model
    intake_agent.py       Free text -> structured symptoms/medications
    safety_agent.py        Rule-based red-flag triage
    literature_agent.py    RAG retrieval over the knowledge base
    summary_agent.py       Drafts the grounded visit summary
    critic_agent.py        Reviews drafts against guardrails
  orchestrator.py         Wires the agents together + execution trace
data/
  knowledge_base.json     Synthetic, original visit-prep guidance snippets
  synthetic_patients.json Fictional demo patient cases
cli.py                    Command-line entry point
app.py                    Streamlit UI (same pipeline, browser front end)
tests/test_pipeline.py    Pytest suite (safety triage, redaction, full pipeline)
```

## Running it

```bash
pip install -r requirements.txt

# Run all synthetic demo patients
python cli.py --demo

# Run one, with the full agent trace
python cli.py --demo-id demo-patient-002 --trace

# Run your own text
python cli.py --text "Sudden chest pain and shortness of breath" \
              --cancer-type lung --patient-id p001 --trace

# Tests
pytest tests/ -v

# Live/interactive demo UI (wraps the same pipeline as cli.py)
streamlit run app.py
```

Everything above runs fully offline with no API keys and no external model
downloads.

### Live demo

`app.py` is a thin Streamlit UI over the exact same `OncoAgentOrchestrator`
used by `cli.py` — one pipeline, two front ends. It lets you pick a synthetic
demo patient or type your own (fictional) notes, and shows the redaction,
structured intake, safety flags, retrieved guidance, generated summary, and
full agent trace in the browser.

To get a shareable live link for a GitHub profile: push this repo to GitHub,
then go to [share.streamlit.io](https://share.streamlit.io), sign in with
GitHub, and deploy pointing at `app.py` on your default branch. No secrets or
API keys are required for the default offline mode, so it deploys as-is;
add `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` as Streamlit Cloud "secrets" later
if you want the richer LLM-generated summaries live.

### Enabling real model backends

Each of the three swap points is a single environment variable plus the
relevant optional dependency:

| Capability | Default (offline) | Real model backend |
|---|---|---|
| PII redaction | regex patterns | `ONCOAGENT_USE_HF_NER=1` → Hugging Face token-classification model (e.g. a clinical de-identification model), via `transformers`/`torch` |
| Guideline retrieval | TF-IDF cosine similarity | `ONCOAGENT_USE_HF_EMBEDDINGS=1` → `sentence-transformers/all-MiniLM-L6-v2` |
| Summary generation | deterministic template | Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` → real LLM prose |

All three fall back gracefully to the offline default if the optional
dependency or network access isn't available, so a misconfigured
environment degrades rather than crashes.

## Example: the revision loop in action

For a patient reporting urgent symptoms, the critic agent catches a first
draft that doesn't clearly tell the patient to contact their care team, and
the summary agent is asked to revise:

```
SummaryAgent :: draft -> draft length=1537 chars
CriticAgent  :: review -> passed=False, issues=["Urgency is urgent but the
                draft doesn't clearly instruct the patient to contact their
                care team or emergency services."]
SummaryAgent :: revise_attempt_2 -> draft length=1687 chars
CriticAgent  :: review -> passed=True, issues=[]
```

## Design notes / what I'd do differently for production

- Replace the knowledge base with real, licensed clinical guideline content
  behind a proper retrieval pipeline (chunking, citation-preserving
  metadata), not synthetic snippets.
- Replace regex/TF-IDF defaults with the HF-backed options as the primary
  path once evaluated on real (properly consented, de-identified) data.
- Add a proper evaluation harness for the critic agent (precision/recall on
  a labeled red-flag test set) rather than a hand-written rule list.
- Wire the structured `PatientIntake` output into real FHIR resources
  (Condition, Observation) instead of the current FHIR-ish approximation.

## Disclaimer

This is a personal engineering portfolio project. It is not a certified
medical device, has not been clinically validated, uses only synthetic data,
and must not be used to make real healthcare decisions for any real patient.
Anyone experiencing a possible medical emergency should contact local
emergency services directly.


