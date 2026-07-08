"""
Safety / Triage Agent.

Deliberately rule-based rather than model-based: for a red-flag
detector where a missed detection could mean a missed emergency, an
explicit, auditable rule set is safer and easier to verify than
relying on model judgment. This is a design choice worth calling out
in an interview -- agents don't have to use an LLM for every step.
"""

from __future__ import annotations

from typing import List

from oncoagent.schemas import PatientIntake, SafetyFlag, UrgencyLevel

EMERGENCY_PATTERNS = [
    "severe chest pain", "chest pain", "difficulty breathing",
    "can't breathe", "cannot breathe", "coughing up blood",
    "severe bleeding", "loss of consciousness", "fainted",
]

URGENT_PATTERNS = [
    "blood streaking", "shortness of breath", "new confusion",
    "high fever", "severe pain", "sudden",
]

SOON_PATTERNS = [
    "worsening", "new numbness", "significant fatigue", "no appetite",
]


class SafetyAgent:
    name = "SafetyAgent"

    def run(self, intake: PatientIntake) -> List[SafetyFlag]:
        text = intake.free_text_notes.lower()
        flags: List[SafetyFlag] = []

        for phrase in EMERGENCY_PATTERNS:
            if phrase in text:
                flags.append(
                    SafetyFlag(
                        reason="Possible medical emergency described",
                        matched_phrase=phrase,
                        urgency=UrgencyLevel.EMERGENCY,
                    )
                )

        if not flags:
            for phrase in URGENT_PATTERNS:
                if phrase in text:
                    flags.append(
                        SafetyFlag(
                            reason="Symptom pattern warrants prompt clinical review",
                            matched_phrase=phrase,
                            urgency=UrgencyLevel.URGENT,
                        )
                    )

        if not flags:
            for phrase in SOON_PATTERNS:
                if phrase in text:
                    flags.append(
                        SafetyFlag(
                            reason="Symptom worth flagging before the next scheduled visit",
                            matched_phrase=phrase,
                            urgency=UrgencyLevel.SOON,
                        )
                    )

        return flags

    def overall_urgency(self, flags: List[SafetyFlag]) -> UrgencyLevel:
        order = [
            UrgencyLevel.EMERGENCY, UrgencyLevel.URGENT,
            UrgencyLevel.SOON, UrgencyLevel.ROUTINE,
        ]
        present = {f.urgency for f in flags}
        for level in order:
            if level in present:
                return level
        return UrgencyLevel.ROUTINE
