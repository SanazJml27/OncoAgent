"""
Literature / Guideline Agent.

Retrieves the most relevant visit-prep guidance snippets from the
knowledge base given the patient's cancer type and reported symptoms.
This is the RAG step of the pipeline.
"""

from __future__ import annotations

from typing import List

from oncoagent.knowledge_base import KnowledgeBase
from oncoagent.schemas import GuidelineSnippet, PatientIntake


class LiteratureAgent:
    name = "LiteratureAgent"

    def __init__(self, kb: KnowledgeBase | None = None):
        self._kb = kb or KnowledgeBase()

    def run(self, intake: PatientIntake, top_k: int = 3) -> List[GuidelineSnippet]:
        query_parts = [intake.cancer_type or ""] + intake.reported_symptoms
        query = " ".join(part for part in query_parts if part).strip()
        if not query:
            query = "general oncology visit preparation"

        return self._kb.retrieve(query, cancer_type=intake.cancer_type, top_k=top_k)
