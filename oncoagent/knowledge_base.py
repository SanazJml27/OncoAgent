"""
Retrieval layer for the Literature/Guideline Agent.

Two retrieval backends are supported:

1. TF-IDF + cosine similarity (default). Pure scikit-learn/numpy, no
   internet access required at runtime -- so the whole project runs
   out of the box on a fresh clone.

2. Sentence-embedding retrieval using a Hugging Face model
   (sentence-transformers/all-MiniLM-L6-v2), enabled by setting
   ONCOAGENT_USE_HF_EMBEDDINGS=1. This demonstrates the integration
   point for swapping in a real HF model; it downloads weights on
   first use, so it needs network access to huggingface.co.

Both backends implement the same `retrieve()` interface so the rest of
the pipeline doesn't care which one is active.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from oncoagent.schemas import GuidelineSnippet

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"


class KnowledgeBase:
    def __init__(self, data_path: Path = DATA_PATH):
        with open(data_path, "r", encoding="utf-8") as f:
            self._records = json.load(f)
        self._texts = [r["text"] for r in self._records]

        self._use_hf = os.getenv("ONCOAGENT_USE_HF_EMBEDDINGS") == "1"
        self._hf_model = None
        self._hf_embeddings = None

        if self._use_hf:
            try:
                self._init_hf_backend()
            except Exception as exc:  # noqa: BLE001 - deliberate broad fallback
                print(
                    f"[KnowledgeBase] Falling back to TF-IDF retrieval "
                    f"(HF embedding backend unavailable: {exc})"
                )
                self._use_hf = False

        if not self._use_hf:
            self._vectorizer = TfidfVectorizer(stop_words="english")
            self._doc_matrix = self._vectorizer.fit_transform(self._texts)

    def _init_hf_backend(self) -> None:
        from sentence_transformers import SentenceTransformer  # optional dep

        self._hf_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self._hf_embeddings = self._hf_model.encode(
            self._texts, convert_to_numpy=True, normalize_embeddings=True
        )

    def retrieve(
        self, query: str, cancer_type: str | None = None, top_k: int = 3
    ) -> List[GuidelineSnippet]:
        candidate_idx = list(range(len(self._records)))
        if cancer_type:
            candidate_idx = [
                i
                for i, r in enumerate(self._records)
                if r["cancer_type"] in (cancer_type, "general")
            ]
            if not candidate_idx:
                candidate_idx = list(range(len(self._records)))

        if self._use_hf and self._hf_model is not None:
            query_vec = self._hf_model.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True
            )
            sims = cosine_similarity(query_vec, self._hf_embeddings[candidate_idx])[0]
        else:
            query_vec = self._vectorizer.transform([query])
            sims = cosine_similarity(query_vec, self._doc_matrix[candidate_idx])[0]

        ranked = sorted(
            zip(candidate_idx, sims), key=lambda pair: pair[1], reverse=True
        )[:top_k]

        results = []
        for idx, score in ranked:
            r = self._records[idx]
            results.append(
                GuidelineSnippet(
                    snippet_id=r["snippet_id"],
                    cancer_type=r["cancer_type"],
                    topic=r["topic"],
                    text=r["text"],
                    score=float(round(score, 4)),
                )
            )
        return results
