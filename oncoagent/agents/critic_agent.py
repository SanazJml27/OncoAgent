"""
Critic Agent.

Implements a lightweight generator-critic loop: the SummaryAgent
drafts, this agent checks the draft against a small set of guardrails,
and the orchestrator re-invokes the SummaryAgent with feedback if a
check fails (bounded to a small number of retries). This is the same
pattern used in more heavyweight "reflection" agent architectures,
just made explicit and auditable instead of another opaque LLM call.
"""

from __future__ import annotations

from typing import List, Tuple

from oncoagent.schemas import SafetyFlag, UrgencyLevel

BANNED_PHRASES = [
    "you have cancer", "you are cured", "diagnosis is", "definitely",
    "guaranteed", "i am a doctor",
]


class CriticAgent:
    name = "CriticAgent"

    def review(
        self, draft: str, urgency: UrgencyLevel, safety_flags: List[SafetyFlag]
    ) -> Tuple[bool, List[str]]:
        issues: List[str] = []
        lowered = draft.lower()

        for phrase in BANNED_PHRASES:
            if phrase in lowered:
                issues.append(
                    f"Draft contains an overreaching clinical claim: '{phrase}'. Remove it."
                )

        if urgency in (UrgencyLevel.EMERGENCY, UrgencyLevel.URGENT):
            if "contact" not in lowered and "call" not in lowered:
                issues.append(
                    "Urgency is "
                    f"{urgency.value} but the draft doesn't clearly instruct the "
                    "patient to contact their care team or emergency services."
                )

        if len(draft.strip()) < 20:
            issues.append("Draft is too short to be a useful summary.")

        passed = len(issues) == 0
        return passed, issues
