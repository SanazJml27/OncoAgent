"""
Summary Agent.

Combines structured intake, retrieved guidance, and safety flags into
a patient-facing visit-preparation summary, via the pluggable LLM
provider (see llm_provider.py). Grounding is enforced by construction:
the prompt only contains facts the pipeline already extracted, so the
model's job is to phrase them clearly, not to introduce new claims.
"""

from __future__ import annotations

from typing import List

from oncoagent.llm_provider import LLMProvider
from oncoagent.schemas import GuidelineSnippet, PatientIntake, SafetyFlag, UrgencyLevel

SYSTEM_PROMPT = (
    "You are a careful clinical visit-preparation assistant. Only use the "
    "facts given to you. Do not invent symptoms, diagnoses, or medical "
    "advice. Always tell the patient to contact their care team directly "
    "for anything urgent. Keep the tone calm and clear."
)


class SummaryAgent:
    name = "SummaryAgent"

    def __init__(self, llm: LLMProvider):
        self._llm = llm

    def run(
        self,
        intake: PatientIntake,
        snippets: List[GuidelineSnippet],
        safety_flags: List[SafetyFlag],
        urgency: UrgencyLevel,
        critic_feedback: str | None = None,
    ) -> str:
        prompt = self._build_prompt(intake, snippets, safety_flags, urgency, critic_feedback)
        return self._llm.complete(SYSTEM_PROMPT, prompt)

    def _build_prompt(
        self,
        intake: PatientIntake,
        snippets: List[GuidelineSnippet],
        safety_flags: List[SafetyFlag],
        urgency: UrgencyLevel,
        critic_feedback: str | None,
    ) -> str:
        lines = [
            f"Patient ID: {intake.patient_id}",
            f"Cancer type: {intake.cancer_type or 'unspecified'}",
            f"Reported symptoms: {', '.join(intake.reported_symptoms) or 'none reported'}",
            f"Current medications: {', '.join(intake.medications) or 'none reported'}",
            f"Overall urgency assessment: {urgency.value}",
        ]

        if safety_flags:
            lines.append("Safety flags detected:")
            for f in safety_flags:
                lines.append(f"  - {f.reason} (matched: '{f.matched_phrase}', level: {f.urgency.value})")

        if snippets:
            lines.append("Relevant background guidance to weave in (do not quote verbatim, paraphrase):")
            for s in snippets:
                lines.append(f"  - [{s.topic}] {s.text}")

        lines.append(
            "\nWrite a short visit-preparation summary for the patient: "
            "1) a one-line urgency note if relevant, 2) a brief summary of "
            "what's been going on, 3) 2-3 suggested discussion questions "
            "for the visit."
        )

        if critic_feedback:
            lines.append(f"\nRevision feedback to address: {critic_feedback}")

        return "\n".join(lines)
