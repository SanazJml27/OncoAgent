"""
Redaction Agent.

Removes personally identifying information from free-text patient
notes before the text is passed to any other agent, logged, or
persisted -- a privacy-by-design pattern that matters a lot for real
health data (and directly informed by EHDS-style data minimisation
thinking).

Two backends:

1. Regex fallback (default, always available): catches common PII
   patterns -- names following "My name is"/"Patient", emails, phone
   numbers, national ID-like numbers, and explicit dates.

2. Hugging Face NER backend (enabled via ONCOAGENT_USE_HF_NER=1):
   uses a token-classification model (e.g. a biomedical/clinical
   de-identification model such as "obi/deid_roberta_i2b2") to catch
   entities the regex patterns would miss. Requires network access to
   huggingface.co and the `transformers` + `torch` extras.
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple

from oncoagent.schemas import RedactionSpan

REGEX_PATTERNS: List[Tuple[str, str]] = [
    (r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "EMAIL"),
    (r"\b(?:\+?\d{1,3}[\s-]?)?\(?\d{2,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4}\b", "PHONE"),
    (r"\b\d{6}[-+A]\d{3}[0-9A-Z]\b", "NATIONAL_ID"),  # Finnish HETU-style
    (r"\b\d{4}-\d{2}-\d{2}\b", "DATE"),
    (r"\b(?:My name is|Patient)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)", "NAME"),
]


class RedactionAgent:
    name = "RedactionAgent"

    def __init__(self):
        self._use_hf = os.getenv("ONCOAGENT_USE_HF_NER") == "1"
        self._pipeline = None
        if self._use_hf:
            try:
                self._init_hf_pipeline()
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[RedactionAgent] Falling back to regex redaction "
                    f"(HF NER backend unavailable: {exc})"
                )
                self._use_hf = False

    def _init_hf_pipeline(self) -> None:
        from transformers import pipeline  # optional dependency

        self._pipeline = pipeline(
            "token-classification",
            model="obi/deid_roberta_i2b2",
            aggregation_strategy="simple",
        )

    def run(self, text: str) -> Tuple[str, List[RedactionSpan]]:
        if self._use_hf and self._pipeline is not None:
            return self._redact_with_hf(text)
        return self._redact_with_regex(text)

    def _redact_with_regex(self, text: str) -> Tuple[str, List[RedactionSpan]]:
        spans: List[RedactionSpan] = []
        redacted = text

        # Process name pattern separately since it has a capture group.
        for pattern, label in REGEX_PATTERNS:
            for match in re.finditer(pattern, redacted):
                start, end = match.span(1) if match.groups() else match.span()
                spans.append(
                    RedactionSpan(
                        start=start, end=end, label=label, source="regex_fallback"
                    )
                )

        # Apply redactions right-to-left so earlier offsets stay valid.
        for span in sorted(spans, key=lambda s: s.start, reverse=True):
            redacted = redacted[: span.start] + f"[REDACTED:{span.label}]" + redacted[span.end :]

        return redacted, spans

    def _redact_with_hf(self, text: str) -> Tuple[str, List[RedactionSpan]]:
        entities = self._pipeline(text)
        spans = [
            RedactionSpan(
                start=e["start"], end=e["end"], label=e["entity_group"],
                source="ner_model",
            )
            for e in entities
        ]
        redacted = text
        for span in sorted(spans, key=lambda s: s.start, reverse=True):
            redacted = (
                redacted[: span.start] + f"[REDACTED:{span.label}]" + redacted[span.end :]
            )
        return redacted, spans
