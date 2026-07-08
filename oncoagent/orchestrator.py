"""
Orchestrator.

Coordinates the agent pipeline:

  RedactionAgent -> IntakeAgent -> SafetyAgent -> LiteratureAgent
      -> SummaryAgent <-> CriticAgent (bounded revision loop)

Each step is logged to a trace list for explainability -- a recruiter
(or a real clinician-facing product) benefits a lot from being able to
see *why* the system produced a given output, not just the output.
"""

from __future__ import annotations

from typing import List

from oncoagent.agents.critic_agent import CriticAgent
from oncoagent.agents.intake_agent import IntakeAgent
from oncoagent.agents.literature_agent import LiteratureAgent
from oncoagent.agents.redaction_agent import RedactionAgent
from oncoagent.agents.safety_agent import SafetyAgent
from oncoagent.agents.summary_agent import SummaryAgent
from oncoagent.llm_provider import get_llm_provider
from oncoagent.schemas import (
    AgentTraceEvent,
    PipelineResult,
    VisitSummary,
)

MAX_REVISIONS = 2


class OncoAgentOrchestrator:
    def __init__(self):
        self.redaction_agent = RedactionAgent()
        self.intake_agent = IntakeAgent()
        self.safety_agent = SafetyAgent()
        self.literature_agent = LiteratureAgent()
        self.summary_agent = SummaryAgent(get_llm_provider())
        self.critic_agent = CriticAgent()

    def run(
        self, patient_id: str, cancer_type: str | None, free_text: str
    ) -> PipelineResult:
        trace: List[AgentTraceEvent] = []

        # 1. Redact PII before anything else touches the raw text.
        redacted_text, spans = self.redaction_agent.run(free_text)
        trace.append(
            AgentTraceEvent(
                agent=self.redaction_agent.name,
                action="redact",
                detail=f"{len(spans)} span(s) redacted",
            )
        )

        # 2. Structure the (redacted) intake.
        intake = self.intake_agent.run(patient_id, cancer_type, redacted_text)
        intake.redacted_notes = redacted_text
        intake.redaction_spans = spans
        trace.append(
            AgentTraceEvent(
                agent=self.intake_agent.name,
                action="structure_intake",
                detail=(
                    f"symptoms={intake.reported_symptoms}, "
                    f"medications={intake.medications}"
                ),
            )
        )

        # 3. Safety triage runs on the ORIGINAL text (pre-redaction) so that
        #    redaction of e.g. a date never affects red-flag matching --
        #    but the flags themselves never expose PII, only phrase matches.
        safety_flags = self.safety_agent.run(
            intake.model_copy(update={"free_text_notes": free_text})
        )
        urgency = self.safety_agent.overall_urgency(safety_flags)
        trace.append(
            AgentTraceEvent(
                agent=self.safety_agent.name,
                action="triage",
                detail=f"urgency={urgency.value}, flags={len(safety_flags)}",
            )
        )

        # 4. Retrieve relevant guidance.
        snippets = self.literature_agent.run(intake)
        trace.append(
            AgentTraceEvent(
                agent=self.literature_agent.name,
                action="retrieve",
                detail=f"retrieved {len(snippets)} snippet(s): "
                f"{[s.snippet_id for s in snippets]}",
            )
        )

        # 5. Generate -> critique -> (maybe revise) loop.
        critic_notes: List[str] = []
        critic_feedback = None
        draft = ""
        final_attempt = 1
        for attempt in range(1, MAX_REVISIONS + 2):
            final_attempt = attempt
            draft = self.summary_agent.run(
                intake, snippets, safety_flags, urgency, critic_feedback
            )
            trace.append(
                AgentTraceEvent(
                    agent=self.summary_agent.name,
                    action="draft" if attempt == 1 else f"revise_attempt_{attempt}",
                    detail=f"draft length={len(draft)} chars",
                )
            )

            passed, issues = self.critic_agent.review(draft, urgency, safety_flags)
            trace.append(
                AgentTraceEvent(
                    agent=self.critic_agent.name,
                    action="review",
                    detail=f"passed={passed}, issues={issues}",
                )
            )

            if passed or attempt > MAX_REVISIONS:
                critic_notes = issues
                break

            critic_feedback = "; ".join(issues)

        summary = VisitSummary(
            patient_id=patient_id,
            urgency=urgency,
            summary_text=draft,
            discussion_questions=[
                "What does my current stage/marker trend mean for the plan?",
                "Which of my symptoms are expected side effects vs. worth flagging sooner?",
            ],
            sources_used=[s.snippet_id for s in snippets],
            critic_notes=critic_notes,
            revision_count=final_attempt - 1,
        )

        return PipelineResult(
            intake=intake,
            safety_flags=safety_flags,
            retrieved_snippets=snippets,
            summary=summary,
            trace=trace,
        )
