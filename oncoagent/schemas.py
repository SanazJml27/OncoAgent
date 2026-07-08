"""
Data schemas shared across all agents.

These are deliberately modeled loosely on FHIR resource shapes
(Patient, Condition, Observation) so the pipeline's output could later
be mapped onto real FHIR resources -- without pulling in a full FHIR
library for this demo project.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class UrgencyLevel(str, Enum):
    ROUTINE = "routine"
    SOON = "soon"          # should be seen within days
    URGENT = "urgent"      # should seek care same day
    EMERGENCY = "emergency"  # should call emergency services now


class RedactionSpan(BaseModel):
    """A single span of text that was flagged and redacted."""

    start: int
    end: int
    label: str
    source: str  # "ner_model" or "regex_fallback"


class PatientIntake(BaseModel):
    """Structured representation of what the patient reported."""

    patient_id: str
    cancer_type: Optional[str] = None
    reported_symptoms: List[str] = Field(default_factory=list)
    medications: List[str] = Field(default_factory=list)
    free_text_notes: str = ""
    redacted_notes: str = ""
    redaction_spans: List[RedactionSpan] = Field(default_factory=list)


class GuidelineSnippet(BaseModel):
    """A retrieved piece of (synthetic, non-copyrighted) guidance."""

    snippet_id: str
    cancer_type: str
    topic: str
    text: str
    score: float


class SafetyFlag(BaseModel):
    reason: str
    matched_phrase: str
    urgency: UrgencyLevel


class VisitSummary(BaseModel):
    patient_id: str
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    urgency: UrgencyLevel
    summary_text: str
    discussion_questions: List[str] = Field(default_factory=list)
    sources_used: List[str] = Field(default_factory=list)
    critic_notes: List[str] = Field(default_factory=list)
    revision_count: int = 0


class AgentTraceEvent(BaseModel):
    """One step in the orchestrator's execution log, for explainability."""

    agent: str
    action: str
    detail: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class PipelineResult(BaseModel):
    intake: PatientIntake
    safety_flags: List[SafetyFlag]
    retrieved_snippets: List[GuidelineSnippet]
    summary: VisitSummary
    trace: List[AgentTraceEvent]
