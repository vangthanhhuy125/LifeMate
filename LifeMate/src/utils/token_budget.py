"""Token budget manager ensuring context length strictly stays within <= 16,384 tokens."""

import json
from typing import Any, Dict, List


class TokenBudgetManager:
    """Tracks and trims prompt components so total context never exceeds 16K tokens."""

    def __init__(
        self,
        max_context_tokens: int = 16384,
        safe_prompt_tokens: int = 12288,
        reserved_generation_tokens: int = 512,
    ) -> None:
        self.max_context_tokens = min(int(max_context_tokens), 16384)
        self.safe_prompt_tokens = min(
            int(safe_prompt_tokens),
            self.max_context_tokens - int(reserved_generation_tokens),
        )
        self.reserved_generation_tokens = int(reserved_generation_tokens)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Conservative token estimator for Vietnamese + JSON text (~2.8 chars/token)."""
        if not text:
            return 0
        char_estimate = int(len(text) / 2.6) + 1
        word_estimate = int(len(text.split()) * 1.45) + 1
        return max(char_estimate, word_estimate)

    def fit_components(
        self,
        system_prompt: str,
        context_block: str,
        schemas: List[Dict[str, Any]],
        few_shot_examples: List[Dict[str, Any]],
        step_history: List[str],
    ) -> Dict[str, Any]:
        """Trim few-shot examples, step history, and verbose schema fields to fit budget."""
        base_tokens = self.estimate_tokens(system_prompt) + self.estimate_tokens(context_block)
        remaining = max(0, self.safe_prompt_tokens - base_tokens)

        trimmed_schemas = list(schemas)
        schema_str = json.dumps(trimmed_schemas, ensure_ascii=False)
        schema_tokens = self.estimate_tokens(schema_str)

        if schema_tokens > int(remaining * 0.65):
            compact_schemas = []
            for s in trimmed_schemas:
                compact_s = dict(s)
                if "description" in compact_s and len(str(compact_s["description"])) > 80:
                    compact_s["description"] = str(compact_s["description"])[:77] + "..."
                compact_schemas.append(compact_s)
            trimmed_schemas = compact_schemas
            schema_str = json.dumps(trimmed_schemas, ensure_ascii=False)
            schema_tokens = self.estimate_tokens(schema_str)

        remaining = max(0, remaining - schema_tokens)

        kept_history: List[str] = []
        used_history_tokens = 0
        for entry in reversed(step_history):
            t_cost = self.estimate_tokens(entry)
            if used_history_tokens + t_cost <= int(remaining * 0.55):
                kept_history.insert(0, entry)
                used_history_tokens += t_cost
            else:
                truncated_entry = entry[:400] + "...[truncated]"
                t_trunc = self.estimate_tokens(truncated_entry)
                if used_history_tokens + t_trunc <= int(remaining * 0.55):
                    kept_history.insert(0, truncated_entry)
                    used_history_tokens += t_trunc
                break

        remaining = max(0, remaining - used_history_tokens)

        kept_examples: List[Dict[str, Any]] = []
        used_example_tokens = 0
        for ex in few_shot_examples:
            ex_str = json.dumps(ex, ensure_ascii=False)
            t_cost = self.estimate_tokens(ex_str)
            if used_example_tokens + t_cost <= remaining:
                kept_examples.append(ex)
                used_example_tokens += t_cost
            else:
                break

        total_estimated = (
            base_tokens + schema_tokens + used_history_tokens + used_example_tokens
        )

        return {
            "schemas": trimmed_schemas,
            "few_shot_examples": kept_examples,
            "step_history": kept_history,
            "estimated_prompt_tokens": total_estimated,
            "within_budget": total_estimated <= self.safe_prompt_tokens,
        }