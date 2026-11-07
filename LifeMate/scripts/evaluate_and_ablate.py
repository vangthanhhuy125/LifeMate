import json
import argparse
from src.evaluation.official_scorer import OfficialLifeMateScorer
from src.modules.dynamic_schema_pruner import DynamicSchemaPruner

def main():
    parser = argparse.ArgumentParser(description="Evaluate Submission & Generate Research Ablation Metrics")
    parser.add_argument("--pred", type=str, required=True, help="Đường dẫn file submission.jsonl")
    parser.add_argument("--gold", type=str, required=True, help="Đường dẫn file jsonl có nhãn chuẩn")
    parser.add_argument("--catalog", type=str, required=True, help="Đường dẫn file <split>_tools_catalog.json")
    args = parser.parse_args()

    with open(args.catalog, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    fuzzy_map = DynamicSchemaPruner.extract_fuzzy_argument_map(catalog)

    preds = {}
    with open(args.pred, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                preds[obj["id"]] = obj

    golds = {}
    with open(args.gold, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                golds[obj["id"]] = {
                    "tool_calls": obj.get("label", {}).get("tool_calls", []),
                    "state_hash": obj.get("label", {}).get("state_hash", "")
                }

    N = len(golds)
    sum_score, sum_tie_f1, sum_P, sum_R, total_U, state_hits = 0.0, 0.0, 0.0, 0.0, 0, 0

    for sid, g_item in golds.items():
        p_item = preds.get(sid, {"tool_calls": [], "state_hash": ""})
        res = OfficialLifeMateScorer.score_instance(
            sys_calls=p_item.get("tool_calls", []),
            gold_calls=g_item["tool_calls"],
            sys_hash=p_item.get("state_hash", ""),
            gold_hash=g_item["state_hash"],
            fuzzy_map=fuzzy_map,
            ignore_hash=False
        )
        sum_score += res["score"]
        sum_tie_f1 += res["f1_unconstrained"]
        sum_P += res["P"]
        sum_R += res["R"]
        total_U += res["U"]
        if res["state_matched"]:
            state_hits += 1

    print("==================================================================")
    print(f"BÁO CÁO KẾT QUẢ THỰC NGHIỆM KHOA HỌC (N = {N} samples)")
    print("------------------------------------------------------------------")
    print(f"1. Final Official Score (100/N * Sum) : {(100.0 / N) * sum_score:.2f}")
    print(f"2. Final State Hash Match Rate (%)    : {(100.0 / N) * state_hits:.2f}%")
    print(f"3. Tie-Breaking Tool Call F1 (%)      : {(100.0 / N) * sum_tie_f1:.2f}")
    print(f"4. Macro Tool Precision P = M/(M+U)   : {(100.0 / N) * sum_P:.2f}%")
    print(f"5. Macro Tool Recall R = M/|G|        : {(100.0 / N) * sum_R:.2f}%")
    print(f"6. Total Unmatched/Rejected Calls (U) : {total_U}")
    print("==================================================================")

if __name__ == "__main__":
    main()
