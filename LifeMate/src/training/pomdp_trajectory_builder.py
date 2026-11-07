"""Converts labeled `train.jsonl` samples into step-by-step POMDP SFT trajectories and RAG indices."""

import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.modules.dynamic_schema_pruner import DynamicSchemaPruner
from src.modules.nlu_normalizers import NLUNormalizer
from src.utils.pomdp_types import SampleInput


class POMDPTrajectoryBuilder:
    """Builds strict leak-free RAG indices and multi-step POMDP SFT datasets from `train.jsonl`."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.normalizer = NLUNormalizer(config)
        self.pruner = DynamicSchemaPruner(config)

    @staticmethod
    def _norm(text: str) -> str:
        return unicodedata.normalize("NFC", str(text).strip())

    def extract_rag_record(self, sample: SampleInput) -> Dict[str, Any]:
        """Extract a compact Few-Shot RAG demonstration record from a labeled training sample."""
        label = sample.label or {}
        gold_calls = label.get("tool_calls") or []
        used_apps: List[str] = []
        for tc in gold_calls:
            if isinstance(tc, dict) and tc.get("name"):
                app = str(tc["name"]).split(".")[0]
                if app != "sys" and app not in used_apps:
                    used_apps.append(app)

        nlu_res = self.normalizer.normalize(sample.audio_text or "", sample.context)
        return {
            "id": sample.sample_id,
            "utterance": nlu_res["normalized_text"],
            "installed_apps": sample.installed_app_names,
            "used_apps": used_apps,
            "gold_tool_calls": gold_calls,
            "is_out_of_scope": len(gold_calls) == 0,
        }

    def build_sft_steps_for_sample(self, sample: SampleInput) -> List[Dict[str, Any]]:
        """Decompose a single training sample's gold tool calls & results into POMDP step transitions."""
        label = sample.label or {}
        gold_calls: List[Dict[str, Any]] = list(label.get("tool_calls") or [])
        gold_results: List[ Any] = list(label.get("tool_results") or [])

        nlu_res = self.normalizer.normalize(sample.audio_text or "", sample.context)
        norm_utt = nlu_res["normalized_text"]
        clues = nlu_res["clues"]
        prune_out = self.pruner.prune_for_sample(sample.context, norm_utt, clues)

        sft_examples: List[Dict[str, Any]] = []
        step_history: List[Dict[str, Any]] = []

        for step_idx, tc in enumerate(gold_calls, start=1):
            t_name = str(tc.get("name", ""))
            t_args = tc.get("arguments") or {}
            act_type = "ASK_MISSING" if t_name == "sys.ask_missing_info" else "CALL_TOOL"

            target_json = {
                "thought": f"Bước {step_idx}: Thực hiện lệnh '{t_name}' dựa trên ngữ cảnh và lịch sử quan sát.",
                "action_type": act_type,
                "tool_name": t_name,
                "arguments": t_args,
            }

            sft_examples.append(
                {
                    "sample_id": sample.sample_id,
                    "step_idx": step_idx,
                    "utterance": norm_utt,
                    "context": sample.context,
                    "pruned_schemas": prune_out["schemas_json"],
                    "history_so_far": list(step_history),
                    "target": target_json,
                }
            )

            obs_res = gold_results[step_idx - 1] if (step_idx - 1) < len(gold_results) else {"ok": True}
            step_history.append(
                {
                    "step_idx": step_idx,
                    "tool_call": tc,
                    "tool_result": obs_res,
                }
            )

        # Add final TERMINATE transition if trajectory did not end with sys.ask_missing_info
        if not gold_calls or gold_calls[-1].get("name") != "sys.ask_missing_info":
            term_step = len(gold_calls) + 1
            sft_examples.append(
                {
                    "sample_id": sample.sample_id,
                    "step_idx": term_step,
                    "utterance": norm_utt,
                    "context": sample.context,
                    "pruned_schemas": prune_out["schemas_json"],
                    "history_so_far": list(step_history),
                    "target": {
                        "thought": "Đã hoàn thành toàn bộ yêu cầu hoặc yêu cầu nằm ngoài phạm vi hỗ trợ.",
                        "action_type": "TERMINATE",
                        "tool_name": None,
                        "arguments": {},
                    },
                }
            )

        return sft_examples

    def process_train_split(
        self,
        train_jsonl_path: str | Path,
        catalog_path: str | Path,
        val_ratio: float = 0.1,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[SampleInput]]:
        """Process `train.jsonl`, splitting into train/val to prevent any evaluation leakage."""
        if Path(catalog_path).exists():
            self.pruner.load_catalog_file(catalog_path)

        samples: List[SampleInput] = []
        with open(train_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                samples.append(SampleInput.from_dict(json.loads(line)))

        n_total = len(samples)
        n_val = max(1, int(n_total * val_ratio)) if n_total > 10 else 0
        train_samples = samples[: n_total - n_val]
        val_samples = samples[n_total - n_val :]

        rag_records: List[Dict[str, Any]] = []
        sft_records: List[Dict[str, Any]] = []

        for s in train_samples:
            rag_records.append(self.extract_rag_record(s))
            sft_records.extend(self.build_sft_steps_for_sample(s))

        return rag_records, sft_records, val_samples