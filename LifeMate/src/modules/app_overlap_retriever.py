"""Hybrid BM25 + Installed App Overlap Retriever for Few-Shot POMDP Trajectory Grounding."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.modules.retriever_bm25 import RetrieverBM25


class AppOverlapRetriever:
    """Combines lexical BM25 similarity with Jaccard app-overlap to select few-shot trajectories."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        ret_cfg = self.config.get("retrieval", self.config)
        self.enable_rag = bool(ret_cfg.get("enable_rag", True))
        self.index_path = str(ret_cfg.get("index_path", "artifacts/rag_trajectories.json"))
        self.top_k = int(ret_cfg.get("top_k", 2))
        self.bm25_weight = float(ret_cfg.get("bm25_weight", 0.55))
        self.app_overlap_weight = float(ret_cfg.get("app_overlap_weight", 0.45))
        self.min_threshold = float(ret_cfg.get("min_similarity_threshold", 0.15))

        self.bm25 = RetrieverBM25()
        if self.index_path and Path(self.index_path).exists():
            self.bm25.load_index(self.index_path)

    def fit_records(self, records: List[Dict[str, Any]]) -> None:
        """Fit retriever directly on a list of training trajectory dictionaries."""
        self.bm25.fit(records)

    @staticmethod
    def _compute_app_overlap(
        query_apps: Set[str], doc_installed_apps: Set[str], doc_used_apps: Set[str]
    ) -> float:
        """Compute compatibility score between sample's installed apps and trajectory's used apps."""
        if not query_apps:
            return 0.0
        # If trajectory used an app that is NOT installed in the current query sample, penalize
        if doc_used_apps and not doc_used_apps.issubset(query_apps):
            return 0.0
        if doc_used_apps and doc_used_apps.issubset(query_apps):
            return 1.0
        union = query_apps | doc_installed_apps
        if not union:
            return 0.0
        return len(query_apps & doc_installed_apps) / float(len(union))

    def retrieve(
        self,
        utterance: str,
        installed_apps: List[str],
        candidate_apps: Optional[List[str]] = None,
        exclude_sample_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k in-context POMDP demonstrations without any test label leakage."""
        if not self.enable_rag or self.top_k <= 0 or not self.bm25.corpus:
            return []

        bm25_scores = self.bm25.score_all(utterance)
        q_installed = set(installed_apps)
        q_candidates = set(candidate_apps) if candidate_apps else q_installed

        scored_items: List[tuple[float, int, Dict[str, Any]]] = []
        for idx, doc in enumerate(self.bm25.corpus):
            doc_id = str(doc.get("id", ""))
            if exclude_sample_id and doc_id == exclude_sample_id:
                continue

            doc_installed = set(doc.get("installed_apps") or [])
            doc_used = set(doc.get("used_apps") or [])

            app_score = self._compute_app_overlap(q_installed, doc_installed, doc_used)
            if q_candidates and doc_used and doc_used.issubset(q_candidates):
                app_score = min(1.0, app_score + 0.25)

            combined = (
                self.bm25_weight * bm25_scores[idx]
                + self.app_overlap_weight * app_score
            )
            if combined >= self.min_threshold:
                scored_items.append((combined, idx, doc))

        scored_items.sort(key=lambda x: (-x[0], x[1]))
        results: List[Dict[str, Any]] = []
        for score, _, doc in scored_items[: self.top_k]:
            copied = dict(doc)
            copied["retrieval_score"] = round(score, 4)
            results.append(copied)
        return results