"""Offline BM25 Retriever for Vietnamese Spoken Utterances."""

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional


class RetrieverBM25:
    """BM25 retriever over training trajectories with NFC Vietnamese word tokenization."""

    STOPWORDS = {
        "tôi",
        "mình",
        "cho",
        "của",
        "và",
        "là",
        "có",
        "được",
        "này",
        "đó",
        "với",
        "nhé",
        "nha",
        "ạ",
        "ơi",
        "hãy",
        "giúp",
        "nhờ",
        "vào",
        "lúc",
    }

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0

    @classmethod
    def tokenize(cls, text: str) -> List[str]:
        """Tokenize Vietnamese text into normalized lowercase word tokens."""
        if not text:
            return []
        norm = unicodedata.normalize("NFC", text.lower())
        tokens = re.findall(r"\w+", norm, flags=re.UNICODE)
        return [t for t in tokens if t not in cls.STOPWORDS and len(t) > 1]

    def fit(self, documents: List[Dict[str, Any]]) -> None:
        """Build BM25 inverted index over trajectory records."""
        self.corpus = list(documents)
        self.tokenized_corpus = []
        self.doc_freqs = {}
        self.doc_len = []

        for doc in self.corpus:
            text = str(doc.get("utterance") or doc.get("audio_text") or "")
            tokens = self.tokenize(text)
            self.tokenized_corpus.append(tokens)
            self.doc_len.append(len(tokens))
            for token in set(tokens):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        n_docs = len(self.corpus)
        self.avgdl = (sum(self.doc_len) / float(n_docs)) if n_docs > 0 else 1.0
        self.idf = {}
        for token, df in self.doc_freqs.items():
            self.idf[token] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    def save_index(self, path: str | Path) -> None:
        """Persist corpus to JSON file for fast offline loading."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.corpus, f, ensure_ascii=False, indent=2)

    def load_index(self, path: str | Path) -> bool:
        """Load trajectory corpus from JSON file and rebuild in-memory BM25 statistics."""
        in_path = Path(path)
        if not in_path.exists():
            return False
        with open(in_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        if isinstance(docs, list):
            self.fit(docs)
            return True
        return False

    def score_all(self, query: str) -> List[float]:
        """Compute normalized BM25 scores in [0, 1] for all documents in the corpus."""
        if not self.corpus:
            return []
        q_tokens = self.tokenize(query)
        if not q_tokens:
            return [0.0] * len(self.corpus)

        raw_scores: List[float] = []
        for idx, doc_tokens in enumerate(self.tokenized_corpus):
            dl = self.doc_len[idx]
            score = 0.0
            if dl > 0:
                tf_map: Dict[str, int] = {}
                for t in doc_tokens:
                    tf_map[t] = tf_map.get(t, 0) + 1
                for qt in q_tokens:
                    if qt not in tf_map:
                        continue
                    tf = tf_map[qt]
                    idf_val = self.idf.get(qt, 0.0)
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                    score += idf_val * (tf * (self.k1 + 1.0)) / denom
            raw_scores.append(score)

        max_score = max(raw_scores) if raw_scores else 0.0
        if max_score <= 1e-9:
            return [0.0] * len(self.corpus)
        return [s / max_score for s in raw_scores]

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Retrieve top-k highest scoring documents by BM25 similarity."""
        scores = self.score_all(query)
        ranked = sorted(
            enumerate(scores), key=lambda x: (-x[1], x[0])
        )
        results: List[Dict[str, Any]] = []
        for idx, sc in ranked[: max(0, top_k)]:
            if sc <= 0.0:
                continue
            item = dict(self.corpus[idx])
            item["bm25_score"] = round(sc, 4)
            results.append(item)
        return results