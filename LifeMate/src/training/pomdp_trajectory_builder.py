import json
from pathlib import Path
from typing import Dict, Any, List
from src.modules.dynamic_schema_pruner import DynamicSchemaPruner

class POMDPTrajectoryBuilder:
    """
    Chuyển đổi dữ liệu `train.jsonl` thành các cặp trạng thái - hành động POMDP:
      (C, S(K), H_{k-1}, A) -> a_k
    Bao gồm cả:
      - Các bước gọi `read` lấy ID `x-ref` rồi mới gọi `write`.
      - Các bước gọi `sys.ask_missing_info` (`ASK_MISSING`).
      - Các mẫu ngoài phạm vi (`TERMINATE` ngay tại k=1).
    """
    @classmethod
    def build_sft_dataset(
        cls,
        train_jsonl_path: str,
        catalog_path: str,
        output_sft_jsonl: str
    ) -> int:
        with open(catalog_path, "r", encoding="utf-8") as f:
            full_catalog = json.load(f)

        sft_samples: List[Dict[str, Any]] = []
        with open(train_jsonl_path, "r", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                item = json.loads(line)
                sample_id = item.get("id")
                ctx = item.get("context", {})
                history = item.get("history", [])
                raw_text = item.get("audio_text", "")
                utterance_A = raw_text[-1] if isinstance(raw_text, list) and raw_text else str(raw_text)

                pruned_SK, _ = DynamicSchemaPruner.prune_and_analyze(
                    full_catalog, ctx.get("installed_apps", {}), enable_pruning=True
                )

                label = item.get("label", {})
                gold_calls = label.get("tool_calls", [])
                gold_results = label.get("tool_results", [])

                interaction_H = []
                for idx, g_call in enumerate(gold_calls):
                    t_name = g_call.get("name", "")
                    t_args = g_call.get("arguments", {})

                    if t_name == "sys.ask_missing_info":
                        target_action = {
                            "thought": "Phát hiện thiếu tham số bắt buộc, kích hoạt AskMissing.",
                            "action_type": "ASK_MISSING",
                            "tool_name": t_args.get("tool_name", ""),
                            "missing_params": t_args.get("params", [])
                        }
                    else:
                        target_action = {
                            "thought": f"Thực thi công cụ bước {idx + 1} theo lược đồ S(K).",
                            "action_type": "CALL",
                            "tool_name": t_name,
                            "arguments": t_args
                        }

                    sft_samples.append({
                        "id": f"{sample_id}_step_{idx+1}",
                        "utterance_A": utterance_A,
                        "context_C": ctx,
                        "dialogue_history": history,
                        "pruned_schema_SK": pruned_SK,
                        "history_H": list(interaction_H),
                        "target_json": json.dumps(target_action, ensure_ascii=False)
                    })

                    obs = gold_results[idx] if idx < len(gold_results) else {"ok": True}
                    interaction_H.append({
                        "step": idx + 1,
                        "action": g_call,
                        "observation_o_k": obs
                    })

                # Nếu mẫu không kết thúc bằng sys.ask_missing_info, thêm bước TERMINATE
                if not gold_calls or gold_calls[-1].get("name") != "sys.ask_missing_info":
                    term_action = {
                        "thought": "Hoàn tất chu trình hoặc yêu cầu nằm ngoài phạm vi hỗ trợ.",
                        "action_type": "TERMINATE"
                    }
                    sft_samples.append({
                        "id": f"{sample_id}_terminate",
                        "utterance_A": utterance_A,
                        "context_C": ctx,
                        "dialogue_history": history,
                        "pruned_schema_SK": pruned_SK,
                        "history_H": list(interaction_H),
                        "target_json": json.dumps(term_action, ensure_ascii=False)
                    })

        Path(output_sft_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(output_sft_jsonl, "w", encoding="utf-8") as fout:
            for s in sft_samples:
                fout.write(json.dumps(s, ensure_ascii=False) + "\n")
        return len(sft_samples)
