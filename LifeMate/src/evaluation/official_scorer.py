"""Exact implementation of the Zalo AI Challenge 2026 LifeMate Official Scoring Formula."""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple


class OfficialScorer:
    """Computes official State Match, Tool-Call Match (with token-F1 > 0.6 for fuzzy fields), and Final Score."""

    def __init__(self, fuzzy_params_by_tool: Optional[Dict[str, Set[str]]] = None) -> None:
        self.fuzzy_params_by_tool: Dict[str, Set[str]] = fuzzy_params_by_tool or {}

    @classmethod
    def from_catalog(cls, catalog_data: Dict[str, Any]) -> "OfficialScorer":
        """Extract all `x-match: "fuzzy"` parameter names per tool from `<split>_tools_catalog.json`."""
        fuzzy_map: Dict[str, Set[str]] = {}
        for app in catalog_data.get("apps", []):
            if not isinstance(app, dict):
                continue
            for tool in app.get("tools", []):
                if not isinstance(tool, dict):
                    continue
                t_name = str(tool.get("name", ""))
                props = (tool.get("parameters") or {}).get("properties") or {}
                for p_name, p_def in props.items():
                    if isinstance(p_def, dict) and p_def.get("x-match") == "fuzzy":
                        fuzzy_map.setdefault(t_name, set()).add(str(p_name))
        return cls(fuzzy_params_by_tool=fuzzy_map)

    @staticmethod
    def normalize_text(text: str) -> str:
        """Lowercase and normalize text to Unicode NFC as specified in the competition rules."""
        return unicodedata.normalize("NFC", str(text).lower().strip())

    @classmethod
    def tokenize_for_f1(cls, text: str) -> List[str]:
        """Split normalized text into alphanumeric tokens for fuzzy F1 computation."""
        norm = cls.normalize_text(text)
        return re.findall(r"\w+", norm, flags=re.UNICODE)

    @classmethod
    def compute_token_f1(cls, pred_str: str, gold_str: str) -> float:
        """Compute token-level F1 between predicted and gold string after lowercasing and Unicode NFC."""
        pred_tokens = cls.tokenize_for_f1(pred_str)
        gold_tokens = cls.tokenize_for_f1(gold_str)

        if not pred_tokens and not gold_tokens:
            return 1.0
        if not pred_tokens or not gold_tokens:
            return 0.0

        gold_counts: Dict[str, int] = {}
        for t in gold_tokens:
            gold_counts[t] = gold_counts.get(t, 0) + 1

        matched = 0
        for t in pred_tokens:
            if gold_counts.get(t, 0) > 0:
                matched += 1
                gold_counts[t] -= 1

        if matched == 0:
            return 0.0
        precision = matched / float(len(pred_tokens))
        recall = matched / float(len(gold_tokens))
        return (2.0 * precision * recall) / (precision + recall)

    def arguments_match(
        self,
        tool_name: str,
        pred_args: Dict[str, Any],
        gold_args: Dict[str, Any],
    ) -> bool:
        """Check whether every non-fuzzy argument is identical and every fuzzy argument has token-F1 > 0.6."""
        if not isinstance(pred_args, dict) or not isinstance(gold_args, dict):
            return False

        if set(pred_args.keys()) != set(gold_args.keys()):
            return False

        fuzzy_fields = self.fuzzy_params_by_tool.get(tool_name, set())

        for key, gold_val in gold_args.items():
            pred_val = pred_args.get(key)
            if key in fuzzy_fields and isinstance(gold_val, str) and isinstance(pred_val, str):
                f1 = self.compute_token_f1(pred_val, gold_val)
                if f1 <= 0.6:
                    return False
            else:
                if isinstance(gold_val, str) and isinstance(pred_val, str):
                    if unicodedata.normalize("NFC", pred_val) != unicodedata.normalize("NFC", gold_val):
                        return False
                elif tool_name == "sys.ask_missing_info" and key == "params":
                    if isinstance(gold_val, list) and isinstance(pred_val, list):
                        if sorted(str(x) for x in pred_val) != sorted(str(x) for x in gold_val):
                            return False
                    elif pred_val != gold_val:
                        return False
                elif pred_val != gold_val:
                    return False

        return True

    def calls_match(self, pred_call: Dict[str, Any], gold_call: Dict[str, Any]) -> bool:
        """Return True if a submitted tool call matches a gold reference tool call."""
        pred_name = str(pred_call.get("name", ""))
        gold_name = str(gold_call.get("name", ""))
        if not pred_name or pred_name != gold_name:
            return False
        return self.arguments_match(
            tool_name=gold_name,
            pred_args=pred_call.get("arguments") or {},
            gold_args=gold_call.get("arguments") or {},
        )

    def match_tool_call_lists(
        self,
        submitted_calls: List[Dict[str, Any]],
        gold_calls: List[Dict[str, Any]],
    ) -> Tuple[int, int, int, float, float, float]:
        """Match submitted calls to gold calls (at most 1-to-1, order-independent).

        Returns:
            Tuple of (M, U, G_len, P, R, F1).
        """
        g_len = len(gold_calls)
        s_len = len(submitted_calls)
        matched_gold_indices: Set[int] = set()
        m_count = 0

        for sub_call in submitted_calls:
            for g_idx, g_call in enumerate(gold_calls):
                if g_idx in matched_gold_indices:
                    continue
                if self.calls_match(sub_call, g_call):
                    matched_gold_indices.add(g_idx)
                    m_count += 1
                    break

        u_count = s_len - m_count
        precision = (m_count / float(m_count + u_count)) if (m_count + u_count) > 0 else 0.0
        recall = (m_count / float(g_len)) if g_len > 0 else 0.0
        f1 = (
            (2.0 * precision * recall) / (precision + recall)
            if (precision + recall) > 0.0
            else 0.0
        )
        return m_count, u_count, g_len, precision, recall, f1

    def score_sample(
        self,
        submitted_calls: List[Dict[str, Any]],
        submitted_hash: str,
        gold_calls: List[Dict[str, Any]],
        gold_hash: str,
    ) -> Dict[str, Any]:
        """Score a single sample using the exact competition formula."""
        # Case 1: Out-of-scope sample (gold list is empty)
        if len(gold_calls) == 0:
            is_correct_oos = len(submitted_calls) == 0 and (submitted_hash == gold_hash)
            score = 1.0 if len(submitted_calls) == 0 else 0.0
            return {
                "score": score,
                "f1_ignore_hash": score,
                "state_match": submitted_hash == gold_hash,
                "is_out_of_scope": True,
                "M": 0,
                "U": len(submitted_calls),
                "G": 0,
                "precision": 1.0 if is_correct_oos else 0.0,
                "recall": 1.0 if is_correct_oos else 0.0,
            }

        # Case 2: Non-empty gold list
        m_count, u_count, g_len, precision, recall, f1 = self.match_tool_call_lists(
            submitted_calls=submitted_calls,
            gold_calls=gold_calls,
        )
        state_match = bool(submitted_hash == gold_hash)

        if not state_match or m_count == 0:
            official_score = 0.0
        else:
            official_score = f1

        return {
            "score": round(official_score, 6),
            "f1_ignore_hash": round(f1, 6),
            "state_match": state_match,
            "is_out_of_scope": False,
            "M": m_count,
            "U": u_count,
            "G": g_len,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
        }

    def score_dataset(
        self,
        predictions: Dict[str, Dict[str, Any]],
        ground_truths: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute dataset-wide Final Score = (100 / N) * sum(Score_i) and tie-breaker F1."""
        n_samples = len(ground_truths)
        if n_samples == 0:
            return {"final_score": 0.0, "tie_breaker_f1": 0.0, "state_match_rate": 0.0, "N": 0}

        total_score = 0.0
        total_f1 = 0.0
        state_matches = 0
        per_sample: Dict[str, Dict[str, Any]] = {}

        for sample_id, gt in ground_truths.items():
            pred = predictions.get(sample_id, {"tool_calls": [], "state_hash": ""})
            res = self.score_sample(
                submitted_calls=pred.get("tool_calls") or [],
                submitted_hash=str(pred.get("state_hash", "")),
                gold_calls=gt.get("tool_calls") or [],
                gold_hash=str(gt.get("state_hash", "")),
            )
            total_score += res["score"]
            total_f1 += res["f1_ignore_hash"]
            if res["state_match"]:
                state_matches += 1
            per_sample[sample_id] = res

        return {
            "final_score": round((100.0 / n_samples) * total_score, 4),
            "tie_breaker_f1": round((100.0 / n_samples) * total_f1, 4),
            "state_match_rate": round((100.0 / n_samples) * state_matches, 4),
            "N": n_samples,
            "per_sample": per_sample,
        }