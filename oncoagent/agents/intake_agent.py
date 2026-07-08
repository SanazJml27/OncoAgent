"""
Intake Agent.

Turns unstructured patient-reported text into a structured
`PatientIntake` object: symptoms, medications, and cancer type.

For portability, this uses lightweight keyword/pattern extraction
rather than requiring an LLM call, but is written so a real NLP model
(e.g. a clinical entity extraction model from Hugging Face) could
replace `_extract_symptoms` / `_extract_medications` without touching
the rest of the pipeline.
"""

from __future__ import annotations

import re
from typing import List

from oncoagent.schemas import PatientIntake

SYMPTOM_KEYWORDS = [
    "fatigue", "shortness of breath", "cough", "chest pain", "dizzy",
    "dizziness", "sweating", "pain", "ache", "aching", "numbness",
    "bleeding", "blood", "nausea", "vomiting", "appetite", "bowel",
    "forgetful", "memory", "hot flashes", "urinary", "mood",
]

MEDICATION_PATTERN = re.compile(
    r"\b(on|currently on|taking)\s+([a-zA-Z][a-zA-Z0-9\-]{2,})\b", re.IGNORECASE
)

KNOWN_DRUG_HINTS = [
    "letrozole", "osimertinib", "folfox", "pembrolizumab", "tamoxifen",
    "trastuzumab", "docetaxel", "carboplatin", "abiraterone",
]


class IntakeAgent:
    name = "IntakeAgent"

    def run(self, patient_id: str, cancer_type: str | None, free_text: str) -> PatientIntake:
        symptoms = self._extract_symptoms(free_text)
        medications = self._extract_medications(free_text)

        return PatientIntake(
            patient_id=patient_id,
            cancer_type=cancer_type,
            reported_symptoms=symptoms,
            medications=medications,
            free_text_notes=free_text,
        )

    def _extract_symptoms(self, text: str) -> List[str]:
        lowered = text.lower()
        found = []
        for kw in SYMPTOM_KEYWORDS:
            if kw in lowered and kw not in found:
                found.append(kw)
        return found

    def _extract_medications(self, text: str) -> List[str]:
        lowered = text.lower()
        found = set()

        for hint in KNOWN_DRUG_HINTS:
            if hint in lowered:
                found.add(hint)

        for match in MEDICATION_PATTERN.finditer(text):
            candidate = match.group(2).lower()
            if candidate not in {"okay", "doing", "shoulder", "days"}:
                found.add(candidate)

        return sorted(found)
