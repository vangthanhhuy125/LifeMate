import re
import unicodedata
from collections import Counter
from typing import List, Dict, Any, Set

class OfficialLifeMateScorer:
    """
    Hiện thực hóa chính xác 100% công thức chấm điểm của đề bài LifeMate:
    - Nếu G rỗng (Out-of-scope): Score = 1 nếu danh sách nộp rỗng, ngược lại = 0.
    - Nếu G khác rỗng:
      + Nếu hash_sys != hash_gold hoặc M == 0 -> Score = 0.
      + Ngược lại: P = M / (M + U), R = M / |G|, Score = 2*P*R / (P + R).
    - Hỗ trợ trích xuất riêng Precision P, Recall R, tỷ lệ khớp State Hash và
      điểm Tie-breaking F1 (bỏ qua kiểm tra state_hash) phục vụ bảng kết quả bài báo.
    """
    @staticmethod
    def normalize_tokens(text: Any) -> List[str]:
        nfc_lower = unicodedata.normalize("NFC", str(text)).lower()
        return re.findall(r'\w+', nfc_lower)

    @classmethod
    def token_f1(cls, pred_val: Any, gold_val: Any) -> float:
        t_pred = cls.normalize_tokens(pred_val)
        t_gold = cls.normalize_tokens(gold_val)
        if not t_pred and not t_gold:
            return 1.0
        if not t_pred or not t_gold:
            return 0.0
        common = Counter(t_pred) & Counter(t_gold)
        same = sum(common.values())
        if same == 0:
            return 0.0
        p = same / len(t_pred)
        r = same / len(t_gold)
        return (2.0 * p * r) / (p + r)

    @classmethod
    def match_call(cls, sys_c: Dict[str, Any], gold_c: Dict[str, Any], fuzzy_keys: Set[str]) -> bool:
        if sys_c.get("name") != gold_c.get("name"):
            return False
        s_args = sys_c.get("arguments", {})
        g_args = gold_c.get("arguments", {})
        if set(s_args.keys()) != set(g_args.keys()):
            return False
        for k, g_val in g_args.items():
            s_val = s_args.get(k)
            if k in fuzzy_keys:
                if cls.token_f1(s_val, g_val) <= 0.6:
                    return False
            else:
                if s_val != g_val:
                    return False
        return True

    @classmethod
    def score_instance(
        cls,
        sys_calls: List[Dict[str, Any]],
        gold_calls: List[Dict[str, Any]],
        sys_hash: str,
        gold_hash: str,
        fuzzy_map: Dict[str, List[str]] = None,
        ignore_hash: bool = False
    ) -> Dict[str, Any]:
        if fuzzy_map is None:
            fuzzy_map = {}

        state_ok = (sys_hash == gold_hash)

        if len(gold_calls) == 0:
            sc = 1.0 if len(sys_calls) == 0 else 0.0
            return {
                "score": sc,
                "P": sc,
                "R": sc,
                "M": 0,
                "U": len(sys_calls),
                "G": 0,
                "state_matched": state_ok
            }

        # Đếm số lệnh khớp M (không xét thứ tự, mỗi lệnh khớp tối đa 1 lần)
        matched_g = set()
        M = 0
        for sc_item in sys_calls:
            f_keys = set(fuzzy_map.get(sc_item.get("name", ""), []))
            for g_idx, gc_item in enumerate(gold_calls):
                if g_idx not in matched_g and cls.match_call(sc_item, gc_item, f_keys):
                    matched_g.add(g_idx)
                    M += 1
                    break

        U = len(sys_calls) - M
        G = len(gold_calls)
        P = M / (M + U) if (M + U) > 0 else 0.0
        R = M / G if G > 0 else 0.0
        f1 = (2.0 * P * R) / (P + R) if (P + R) > 0 else 0.0

        if (not ignore_hash and not state_ok) or M == 0:
            final_s = 0.0
        else:
            final_s = f1

        return {
            "score": final_s,
            "P": P,
            "R": R,
            "f1_unconstrained": f1,
            "M": M,
            "U": U,
            "G": G,
            "state_matched": state_ok
        }
