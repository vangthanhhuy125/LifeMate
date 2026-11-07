import json
import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any

class NativeBM25Okapi:
    """Hiện thực hóa thuật toán Okapi BM25 độc lập, không phụ thuộc thư viện ngoài."""
    def __init__(self, tokenized_corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(tokenized_corpus)
        self.doc_len = [len(doc) for doc in tokenized_corpus]
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0.0
        self.doc_freqs = []
        self.idf = {}
        df = {}
        for doc in tokenized_corpus:
            frequencies = Counter(doc)
            self.doc_freqs.append(frequencies)
            for token in frequencies.keys():
                df[token] = df.get(token, 0) + 1
        for token, freq in df.items():
            self.idf[token] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.corpus_size
        for q in query_tokens:
            if q not in self.idf:
                continue
            q_idf = self.idf[q]
            for idx, doc_freq in enumerate(self.doc_freqs):
                tf = doc_freq.get(q, 0)
                if tf > 0:
                    denom = tf + self.k1 * (1.0 - self.b + self.b * self.doc_len[idx] / (self.avgdl + 1e-8))
                    scores[idx] += q_idf * (tf * (self.k1 + 1.0)) / denom
        return scores

class AppOverlapTrajectoryRetriever:
    """
    [Contribution 3 - Grounding đa phương thức qua Few-Shot RAG nội bộ]:
    Truy xuất quỹ đạo mẫu (trajectories) từ tập huấn luyện kết hợp điểm BM25 ngữ nghĩa
    và hệ số tương đồng tập ứng dụng khả dụng (App-Overlap Weighting).
    """
    def __init__(self, top_k: int = 3, app_overlap_boost: float = 1.35):
        self.top_k = top_k
        self.app_overlap_boost = app_overlap_boost
        self.records: List[Dict[str, Any]] = []
        self.bm25 = None

    @staticmethod
    def tokenize_vi(text: str) -> List[str]:
        return re.findall(r'\w+', str(text).lower())

    def build_and_save_index(self, train_jsonl_path: str, save_index_path: str = None) -> int:
        records = []
        tokenized_docs = []
        with open(train_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                raw_text = item.get("audio_text", "")
                utterance = raw_text[-1] if isinstance(raw_text, list) and raw_text else str(raw_text)
                gold_calls = item.get("label", {}).get("tool_calls", [])
                installed_apps = list(item.get("context", {}).get("installed_apps", {}).keys())

                if utterance:
                    records.append({
                        "id": item.get("id"),
                        "utterance": utterance,
                        "installed_apps": installed_apps,
                        "gold_tool_calls": gold_calls
                    })
                    tokenized_docs.append(self.tokenize_vi(utterance))

        self.records = records
        if tokenized_docs:
            self.bm25 = NativeBM25Okapi(tokenized_docs)

        if save_index_path:
            Path(save_index_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_index_path, "wb") as out:
                pickle.dump((self.records, self.bm25), out)
        return len(self.records)

    def load_index(self, index_path: str) -> bool:
        p = Path(index_path)
        if p.exists():
            with open(p, "rb") as f:
                self.records, self.bm25 = pickle.load(f)
            return True
        return False

    def retrieve_top_k(self, utterance_A: str, active_app_ids: List[str]) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.records:
            return []

        query_tokens = self.tokenize_vi(utterance_A)
        base_scores = self.bm25.get_scores(query_tokens)
        active_set = set(active_app_ids)

        ranked = []
        for idx, base_s in enumerate(base_scores):
            rec = self.records[idx]
            rec_apps = set(rec.get("installed_apps", []))
            overlap_count = len(active_set & rec_apps)
            multiplier = (self.app_overlap_boost ** overlap_count) if overlap_count > 0 else 1.0
            ranked.append((base_s * multiplier, rec))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in ranked[:self.top_k]]
