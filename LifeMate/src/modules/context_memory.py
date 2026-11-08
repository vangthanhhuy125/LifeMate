"""Conversation Context and Multi-Turn Working Memory Manager."""

import unicodedata
from typing import Any, Dict, List, Optional

from src.utils.pomdp_types import BeliefState, SampleInput


class ContextMemoryManager:
    """Manages multi-turn history, coreference resolution, and initial POMDP BeliefState."""

    def __init__(self, nlu_normalizer: Optional[Any] = None) -> None:
        self.nlu_normalizer = nlu_normalizer

    @staticmethod
    def _normalize(text: str) -> str:
        """Standardize text to Unicode NFC and trim whitespace."""
        return unicodedata.normalize("NFC", str(text).strip())

    def summarize_history(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Distill earlier conversation turns into historical actions and unresolved context."""
        historical_utterances: List[str] = []
        historical_entities: Dict[str, Any] = {}
        pending_missing_tool: Optional[str] = None
        pending_missing_params: List[str] = []

        for turn in history:
            if not isinstance(turn, dict):
                continue

            role = str(turn.get("role", "")).lower()
            content = self._normalize(
                turn.get("content")
                or turn.get("text")
                or turn.get("audio_text")
                or ""
            )
            if content and role in ("user", "human", ""):
                historical_utterances.append(content)

            # Inspect any tool calls recorded in history turns
            t_calls = turn.get("tool_calls") or []
            for tc in t_calls:
                if not isinstance(tc, dict):
                    continue
                t_name = tc.get("name", "")
                t_args = tc.get("arguments") or {}

                if t_name == "sys.ask_missing_info":
                    pending_missing_tool = t_args.get("tool_name")
                    pending_missing_params = list(t_args.get("params") or [])
                else:
                    for arg_k, arg_v in t_args.items():
                        if arg_v is not None and arg_v != "":
                            historical_entities[arg_k] = arg_v

        return {
            "historical_utterances": historical_utterances,
            "historical_entities": historical_entities,
            "pending_missing_tool": pending_missing_tool,
            "pending_missing_params": pending_missing_params,
        }

    def initialize_belief_state(
        self,
        sample: SampleInput,
        raw_utterance: str,
        normalized_utterance: str,
        extracted_clues: Dict[str, Any],
        candidate_apps: List[str],
        is_out_of_scope_likely: bool = False,
    ) -> BeliefState:
        """Create the initial POMDP BeliefState b_0 combining current turn and history memory."""
        ctx = sample.context
        current_time = str(ctx.get("current_time", "2026-11-16T17:00:00+07:00"))
        timezone = str(ctx.get("timezone", "Asia/Ho_Chi_Minh"))

        # Resolve current location
        loc = ctx.get("location") or ctx.get("current_location")
        if isinstance(loc, dict):
            loc_str = self._normalize(loc.get("address") or loc.get("name") or "")
        elif isinstance(loc, str):
            loc_str = self._normalize(loc)
        else:
            loc_str = None

        # Resolve user profile
        raw_profile = ctx.get("user_profile") or {}
        normalized_profile = {
            str(k): self._normalize(v) if isinstance(v, str) else v
            for k, v in raw_profile.items()
        }

        # Contextual inference: if location missing but user leaving work for home
        if (
            not loc_str
            and "work_address" in normalized_profile
            and "sắp về nhà" in normalized_utterance.lower()
        ):
            loc_str = str(normalized_profile["work_address"])

        history_summary = self.summarize_history(sample.history)
        merged_clues = dict(extracted_clues)

        # Carry over pending missing info context from previous turns
        if history_summary["pending_missing_tool"]:
            merged_clues["pending_missing_tool"] = history_summary["pending_missing_tool"]
            merged_clues["pending_missing_params"] = history_summary["pending_missing_params"]

        # Merge historical entities if not already overwritten
        for hist_k, hist_v in history_summary["historical_entities"].items():
            if hist_k not in merged_clues:
                merged_clues[f"history_{hist_k}"] = hist_v

        return BeliefState(
            sample_id=sample.sample_id,
            raw_utterance=raw_utterance,
            normalized_utterance=normalized_utterance,
            current_time=current_time,
            timezone=timezone,
            current_location=loc_str,
            user_profile=normalized_profile,
            installed_apps=sample.installed_app_names,
            conversation_history=sample.history,
            extracted_clues=merged_clues,
            candidate_apps=candidate_apps,
            is_out_of_scope_likely=is_out_of_scope_likely,
        )