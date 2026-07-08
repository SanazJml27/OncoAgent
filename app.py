"""
Streamlit demo UI for OncoAgent.

This is a thin presentation layer over the same pipeline used by cli.py --
it calls OncoAgentOrchestrator directly, so there is exactly one pipeline
implementation behind both the CLI and this UI.

Run locally:
    streamlit run app.py

Deploy for free (recruiter-facing live link):
    Push this repo to GitHub, then create an app at https://share.streamlit.io
    pointing at app.py. No API keys are required for the default (offline)
    mode.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from oncoagent.orchestrator import OncoAgentOrchestrator
from oncoagent.schemas import PipelineResult, UrgencyLevel

DEMO_PATH = Path(__file__).resolve().parent / "data" / "synthetic_patients.json"

URGENCY_COLOR = {
    UrgencyLevel.ROUTINE: "🟢",
    UrgencyLevel.SOON: "🟡",
    UrgencyLevel.URGENT: "🟠",
    UrgencyLevel.EMERGENCY: "🔴",
}

st.set_page_config(page_title="OncoAgent", page_icon="🩺", layout="wide")


@st.cache_resource
def get_orchestrator() -> OncoAgentOrchestrator:
    # Cached so agents (and any real model backends) load once per session,
    # not on every rerun.
    return OncoAgentOrchestrator()


@st.cache_data
def load_demo_patients():
    with open(DEMO_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def render_result(result: PipelineResult, show_trace: bool) -> None:
    urgency = result.summary.urgency
    st.markdown(f"### {URGENCY_COLOR[urgency]} Urgency: `{urgency.value.upper()}`")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Redacted notes")
        st.info(result.intake.redacted_notes or "(empty)")
        st.caption(f"{len(result.intake.redaction_spans)} PII span(s) redacted before any other agent saw this text.")

        st.subheader("Structured intake")
        st.write("**Symptoms:**", ", ".join(result.intake.reported_symptoms) or "(none detected)")
        st.write("**Medications:**", ", ".join(result.intake.medications) or "(none detected)")

        if result.safety_flags:
            st.subheader("Safety flags")
            for f in result.safety_flags:
                st.warning(f"[{f.urgency.value.upper()}] {f.reason} — matched: '{f.matched_phrase}'")

    with col2:
        st.subheader("Retrieved guidance (RAG)")
        for s in result.retrieved_snippets:
            st.write(f"**({s.score:.3f}) [{s.topic}]**")
            st.caption(s.text)

        st.subheader("Visit summary")
        st.success(result.summary.summary_text)

        if result.summary.critic_notes:
            st.subheader("Unresolved critic notes")
            for n in result.summary.critic_notes:
                st.error(n)

        st.caption(f"Revisions requested by critic agent: {result.summary.revision_count}")

    if show_trace:
        st.subheader("Agent execution trace")
        for event in result.trace:
            st.text(f"[{event.timestamp}] {event.agent} :: {event.action} -> {event.detail}")


def main() -> None:
    st.title("🩺 OncoAgent")
    st.caption(
        "A multi-agent pipeline for oncology visit preparation — redaction, "
        "structuring, safety triage, RAG retrieval, and a generator/critic "
        "revision loop. Runs fully offline; see the sidebar for details."
    )

    with st.sidebar:
        st.header("About")
        st.markdown(
            "This is a portfolio/demo project — **not a medical device** and "
            "not for real clinical use. All patient data here is synthetic."
        )
        st.markdown("[View source on GitHub](https://github.com/)")
        st.divider()
        show_trace = st.checkbox("Show agent execution trace", value=False)
        st.divider()
        st.caption(
            "Backends: regex redaction / TF-IDF retrieval / template summary "
            "by default. Set ONCOAGENT_USE_HF_NER, "
            "ONCOAGENT_USE_HF_EMBEDDINGS, or ANTHROPIC_API_KEY / "
            "OPENAI_API_KEY as environment variables to switch to real "
            "model backends."
        )

    orchestrator = get_orchestrator()
    demo_patients = load_demo_patients()

    tab_demo, tab_custom = st.tabs(["Try a demo patient", "Enter your own text"])

    with tab_demo:
        labels = [f"{p['patient_id']} ({p.get('cancer_type', 'n/a')})" for p in demo_patients]
        choice = st.selectbox("Choose a synthetic demo patient", labels)
        selected = demo_patients[labels.index(choice)]

        st.text_area("Raw patient notes (as submitted)", selected["free_text_notes"], height=100, disabled=True)

        if st.button("Run pipeline", key="run_demo"):
            with st.spinner("Running agents..."):
                result = orchestrator.run(
                    selected["patient_id"], selected.get("cancer_type"), selected["free_text_notes"]
                )
            render_result(result, show_trace)

    with tab_custom:
        st.caption("Try your own (fictional!) patient text. Nothing is stored server-side.")
        patient_id = st.text_input("Patient ID", value="custom-patient")
        cancer_type = st.selectbox("Cancer type", ["breast", "lung", "colorectal", "prostate", "other/unspecified"])
        free_text = st.text_area(
            "Patient notes",
            placeholder="e.g. Fatigue and joint aching this week, currently on letrozole.",
            height=120,
        )

        if st.button("Run pipeline", key="run_custom"):
            if not free_text.strip():
                st.error("Please enter some patient notes first.")
            else:
                ct = None if cancer_type == "other/unspecified" else cancer_type
                with st.spinner("Running agents..."):
                    result = orchestrator.run(patient_id, ct, free_text)
                render_result(result, show_trace)


if __name__ == "__main__":
    main()
