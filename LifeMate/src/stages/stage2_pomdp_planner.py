"""Stage 2: State-Grounded POMDP Policy Planner with Expected Utility & LLM Integration."""

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.pomdp_types import (
    ActionType,
    BeliefState,
    POMDPAction,
    ToolSpec,
    UtilityEstimate,
)
from src.utils.token_budget import TokenBudgetManager


class Stage2POMDPPlanner:
    """Computes POMDP action decisions over BeliefState $b_k$ via LLM and Utility Grounding."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        plan_cfg = config.get("pomdp_planner", {})
        hw_cfg = config.get("hardware_constraints", {})

        self.model_name_or_path = str(
            plan_cfg.get("llm_model_name_or_path", "models/Qwen2.5-3B-Instruct")
        )
        self.lora_adapter_path = str(plan_cfg.get("lora_adapter_path", "artifacts/lora_adapter"))
        self.backend = str(plan_cfg.get("backend", "heuristic_or_hf"))
        self.temperature = float(plan_cfg.get("temperature", 0.0))
        self.do_sample = bool(plan_cfg.get("do_sample", False))
        self.max_new_tokens = int(plan_cfg.get("max_new_tokens", 384))
        self.enable_constrained_decoding = bool(
            plan_cfg.get("enable_constrained_decoding", True)
        )
        self.info_gain_read_bonus = float(plan_cfg.get("info_gain_read_bonus", 0.35))
        self.missing_param_ask_threshold = float(
            plan_cfg.get("missing_param_ask_threshold", 0.70)
        )

        self.token_budget = TokenBudgetManager(
            max_context_tokens=int(hw_cfg.get("max_context_tokens", 16384)),
            safe_prompt_tokens=int(hw_cfg.get("safe_prompt_tokens", 12288)),
            reserved_generation_tokens=self.max_new_tokens,
        )

        self.system_prompt = self._load_text_file(
            plan_cfg.get("system_prompt_path", "configs/prompts/policy_system_prompt.txt")
        )
        self.policy_template = self._load_text_file(
            plan_cfg.get("policy_prompt_path", "configs/prompts/pomdp_policy_prompt.txt")
        )

        self._llm_model = None
        self._llm_tokenizer = None
        self._llm_load_attempted = False

    @staticmethod
    def _load_text_file(path_str: str) -> str:
        path = Path(path_str)
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        return ""

    @staticmethod
    def _norm(text: str) -> str:
        return unicodedata.normalize("NFC", str(text).strip())

    def _lazy_init_llm(self) -> None:
        """Lazily load local HuggingFace LLM (+ optional LoRA adapter) when weights exist on disk."""
        if self._llm_load_attempted or self.backend == "heuristic_only":
            return
        self._llm_load_attempted = True

        model_path = Path(self.model_name_or_path)
        if not model_path.exists():
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._llm_tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), local_files_only=True, trust_remote_code=True
            )
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                local_files_only=True,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
            )

            lora_path = Path(self.lora_adapter_path)
            if lora_path.exists() and (lora_path / "adapter_config.json").exists():
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, str(lora_path), is_trainable=False)

            model.eval()
            self._llm_model = model
        except Exception:
            self._llm_model = None
            self._llm_tokenizer = None

    def build_step_prompt(
        self,
        belief_state: BeliefState,
        pruned_tools: List[ToolSpec],
        few_shot_examples: List[Dict[str, Any]],
        step_idx: int,
    ) -> Dict[str, Any]:
        """Construct the token-budgeted POMDP prompt for step $k$."""
        schemas_list = [
            {
                "name": t.name,
                "kind": t.kind,
                "description": t.description,
                "parameters": t.parameters,
                "required": t.required_params,
                "x_ref": t.xref_params,
                "x_one_of": t.one_of_groups,
            }
            for t in pruned_tools
        ]

        history_lines: List[str] = []
        for s in belief_state.steps:
            status = "OK" if s.observation.ok else f"ERROR ({s.observation.error})"
            history_lines.append(
                f"Step {s.step_idx}: Action={s.action.tool_name}({json.dumps(s.action.arguments, ensure_ascii=False)}) -> {status} | Obs={json.dumps(s.observation.result, ensure_ascii=False)}"
            )

        context_summary = json.dumps(
            {
                "current_time": belief_state.current_time,
                "timezone": belief_state.timezone,
                "current_location": belief_state.current_location,
                "user_profile": belief_state.user_profile,
                "installed_apps": belief_state.installed_apps,
            },
            ensure_ascii=False,
        )

        budgeted = self.token_budget.fit_components(
            system_prompt=self.system_prompt,
            context_block=context_summary,
            schemas=schemas_list,
            few_shot_examples=few_shot_examples,
            step_history=history_lines,
        )

        prompt_body = self.policy_template.format(
            current_time=belief_state.current_time,
            timezone=belief_state.timezone,
            current_location=belief_state.current_location or "Không rõ",
            user_profile=json.dumps(belief_state.user_profile, ensure_ascii=False),
            installed_apps=json.dumps(belief_state.installed_apps, ensure_ascii=False),
            few_shot_examples=json.dumps(budgeted["few_shot_examples"], ensure_ascii=False, indent=2),
            pruned_schemas=json.dumps(budgeted["schemas"], ensure_ascii=False, indent=2),
            raw_utterance=belief_state.raw_utterance,
            normalized_utterance=belief_state.normalized_utterance,
            extracted_clues=json.dumps(belief_state.extracted_clues, ensure_ascii=False),
            known_entities=json.dumps(belief_state.known_entities, ensure_ascii=False),
            step_history="\n".join(budgeted["step_history"]) if budgeted["step_history"] else "Chưa có bước thực thi nào.",
            step_idx=step_idx,
        )

        return {
            "system_prompt": self.system_prompt,
            "user_prompt": prompt_body,
            "estimated_tokens": budgeted["estimated_prompt_tokens"],
            "within_budget": budgeted["within_budget"],
        }

    def _generate_with_llm(self, system_prompt: str, user_prompt: str) -> Optional[POMDPAction]:
        """Run deterministic greedy decoding on the local LLM and parse the JSON decision."""
        self._lazy_init_llm()
        if self._llm_model is None or self._llm_tokenizer is None:
            return None

        try:
            import torch

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            if hasattr(self._llm_tokenizer, "apply_chat_template"):
                full_text = self._llm_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                full_text = f"{system_prompt}\n\n{user_prompt}\n\nJSON:"

            inputs = self._llm_tokenizer(full_text, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to(self._llm_model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._llm_model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=self._llm_tokenizer.eos_token_id,
                )
            gen_ids = outputs[0][inputs["input_ids"].shape[1] :]
            raw_output = self._llm_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            return self.parse_llm_json_output(raw_output)
        except Exception:
            return None

    @classmethod
    def parse_llm_json_output(cls, raw_output: str) -> Optional[POMDPAction]:
        """Extract and validate a structured `POMDPAction` from LLM text output."""
        if not raw_output:
            return None
        match = re.search(r"\{.*\}", raw_output, flags=re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
            if not isinstance(data, dict):
                return None
            act_str = str(data.get("action_type", "CALL_TOOL")).upper()
            tool_name = data.get("tool_name") or data.get("name")
            args = data.get("arguments") or {}
            thought = str(data.get("thought", ""))

            if act_str == "TERMINATE" or not tool_name:
                return POMDPAction(action_type=ActionType.TERMINATE, thought=thought)
            if act_str == "ASK_MISSING" or tool_name == "sys.ask_missing_info":
                return POMDPAction(
                    action_type=ActionType.ASK_MISSING,
                    tool_name="sys.ask_missing_info",
                    arguments=dict(args) if isinstance(args, dict) else {},
                    thought=thought,
                )
            return POMDPAction(
                action_type=ActionType.CALL_TOOL,
                tool_name=str(tool_name),
                arguments=dict(args) if isinstance(args, dict) else {},
                thought=thought,
            )
        except Exception:
            return None

    def _match_entity_from_belief(
        self,
        xref_target: str,
        belief_state: BeliefState,
        utterance_lower: str,
    ) -> List[Dict[str, Any]]:
        """Match observed entities in `belief_state.known_entities` against utterance & clues."""
        candidates: List[Dict[str, Any]] = []
        if xref_target in belief_state.known_entities:
            candidates.extend(belief_state.known_entities[xref_target])
        short_key = xref_target.split(".")[-1]
        if short_key in belief_state.known_entities and not candidates:
            candidates.extend(belief_state.known_entities[short_key])
        if not candidates:
            app_prefix = xref_target.split(".")[0]
            for k, items in belief_state.known_entities.items():
                if k.startswith(f"{app_prefix}."):
                    candidates.extend(items)

        if not candidates:
            return []

        clues = belief_state.extracted_clues
        target_room = clues.get("room")
        target_floor = clues.get("floor")
        target_dev_types = clues.get("device_types") or []
        target_recipient = (
            clues.get("recipient_name") or clues.get("recipient_relation_or_name") or ""
        ).lower()

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for ent in candidates:
            score = 0.0
            ent_name = str(ent.get("name") or ent.get("title") or ent.get("label") or "").lower()
            ent_room = str(ent.get("room") or "").lower()
            ent_type = str(ent.get("type") or "").lower()
            ent_floor = ent.get("floor")
            ent_rel = str(ent.get("relation") or "").lower()

            if ent_name and ent_name in utterance_lower:
                score += 3.0
            if target_room:
                if ent_room == target_room.lower() or target_room.lower() in ent_name:
                    score += 2.5
                elif ent_room and ent_room != target_room.lower():
                    score -= 2.0
            if target_floor is not None and ent_floor is not None:
                if int(ent_floor) == int(target_floor):
                    score += 2.5
                else:
                    score -= 2.0
            if target_dev_types:
                if ent_type in target_dev_types:
                    score += 2.0
                elif any(dt in ent_name for dt in target_dev_types):
                    score += 1.5
            if target_recipient:
                if target_recipient == ent_name or target_recipient == ent_rel or target_recipient in ent_name:
                    score += 3.0

            scored.append((score, ent))

        scored.sort(key=lambda x: -x[0])
        if not scored:
            return []

        # If user asked for "hết" / "tất cả" (e.g. "Tắt hết đèn tầng một"), return all positive matches
        if any(w in utterance_lower for w in ("tất cả", "hết", "toàn bộ")):
            positives = [item for sc, item in scored if sc > 0.0]
            if positives:
                return positives

        best_score, best_ent = scored[0]
        if best_score > 0.0 or len(candidates) == 1:
            return [best_ent]
        return []

    def _synthesize_grounded_arguments(
        self,
        tool_spec: ToolSpec,
        belief_state: BeliefState,
    ) -> Tuple[Dict[str, Any], List[str], bool]:
        """Ground tool arguments from `belief_state` clues, profile, and observed `x-ref` entities.

        Returns:
            Tuple of (arguments_dict, missing_required_params, needs_read_first).
        """
        args: Dict[str, Any] = {}
        utterance = belief_state.normalized_utterance
        lower = utterance.lower()
        clues = belief_state.extracted_clues
        props = tool_spec.parameters.get("properties") or {}

        needs_read_first = False

        # 1. Resolve x-ref parameters from observed environment entities
        for param_name, xref_target in tool_spec.xref_params.items():
            matched_entities = self._match_entity_from_belief(xref_target, belief_state, lower)
            if not matched_entities:
                valid_ids = belief_state.get_valid_ids_for_xref(xref_target)
                if not valid_ids:
                    needs_read_first = True
                continue

            # Pick the first matched entity whose ID hasn't already been used with this exact write tool
            chosen_id: Optional[str] = None
            for ent in matched_entities:
                cand_id = str(ent.get("id") or ent.get(param_name) or "")
                if not cand_id:
                    continue
                already_called = any(
                    s.was_committed_to_env
                    and s.observation.ok
                    and s.action.tool_name == tool_spec.name
                    and s.action.arguments.get(param_name) == cand_id
                    for s in belief_state.steps
                )
                if not already_called:
                    chosen_id = cand_id
                    break

            if chosen_id:
                args[param_name] = chosen_id

        # 2. Populate domain parameters according to property definitions
        for prop_name, prop_def in props.items():
            if prop_name in args or not isinstance(prop_def, dict):
                continue
            p_lower = prop_name.lower()
            enums = prop_def.get("enum") or []

            if p_lower in ("power", "state", "status") and "power" in clues:
                val = clues["power"]
                if not enums or val in enums:
                    args[prop_name] = val
            elif p_lower in ("temperature", "temp") and "temperature" in clues:
                args[prop_name] = int(clues["temperature"])
            elif p_lower == "mode":
                if "ac_mode" in clues and (not enums or clues["ac_mode"] in enums):
                    args[prop_name] = clues["ac_mode"]
                elif "cool" in enums and ("điều hòa" in lower or "máy lạnh" in lower or "ac" in lower):
                    args[prop_name] = "cool"
            elif p_lower in ("brightness", "level", "volume") and "percentage_value" in clues:
                args[prop_name] = int(clues["percentage_value"])
            elif p_lower in ("from", "pickup", "origin", "from_address", "pickup_address"):
                if "pickup_address" in clues:
                    args[prop_name] = clues["pickup_address"]
                elif belief_state.current_location:
                    args[prop_name] = belief_state.current_location
            elif p_lower in ("to", "destination", "dest", "to_address", "dropoff_address", "address"):
                if "destination_address" in clues:
                    args[prop_name] = clues["destination_address"]
            elif p_lower in ("time", "datetime", "remind_at", "alarm_time", "start_time"):
                if "resolved_time_iso" in clues:
                    args[prop_name] = clues["resolved_time_iso"]
            elif p_lower in ("amount", "money", "value") and "amount" in clues:
                args[prop_name] = clues["amount"]
            elif p_lower in ("title", "content", "message", "note", "subject"):
                if "reminder_title" in clues:
                    args[prop_name] = clues["reminder_title"]

            # Carry over historical entities if relevant
            hist_key = f"history_{prop_name}"
            if prop_name not in args and hist_key in clues:
                args[prop_name] = clues[hist_key]

        # 3. Check missing required & x-one-of parameters
        missing: List[str] = []
        for req in tool_spec.required_params:
            if req not in args or args[req] is None or args[req] == "":
                missing.append(req)
        for group in tool_spec.one_of_groups:
            if group and not any(g in args and args[g] not in (None, "") for g in group):
                for g in group:
                    if g not in missing:
                        missing.append(g)

        return args, missing, needs_read_first

    def _select_tool_matches_for_utterance(
        self, pruned_tools: List[ToolSpec], belief_state: BeliefState
    ) -> List[ToolSpec]:
        """Rank write and read tools by semantic alignment with the utterance and clues."""
        lower = belief_state.normalized_utterance.lower()
        clues = belief_state.extracted_clues
        candidate_apps = set(belief_state.candidate_apps)

        scored: List[Tuple[float, ToolSpec]] = []
        for spec in pruned_tools:
            if spec.is_meta:
                continue
            if candidate_apps and spec.app not in candidate_apps:
                continue

            score = 0.0
            t_name = spec.name.lower()

            # Smart home matching
            if spec.app == "smart_home":
                dev_types = clues.get("device_types") or []
                if ("ac" in dev_types or "temperature" in clues) and ("climate" in t_name or "ac" in t_name):
                    score += 4.0
                if "light" in dev_types and "light" in t_name:
                    score += 4.0
                if "fan" in dev_types and "fan" in t_name:
                    score += 4.0
                if "curtain" in dev_types and "curtain" in t_name:
                    score += 4.0
                if "tv" in dev_types and "tv" in t_name:
                    score += 4.0

            # Ride-hailing matching
            if spec.app == "ride" and any(w in lower for w in ("đặt xe", "book xe", "gọi xe", "đón tôi")):
                if "book" in t_name or "create" in t_name or "order" in t_name:
                    score += 4.5

            # Alarm & Reminder matching
            if spec.app == "alarm" and "báo thức" in lower:
                if "create" in t_name or "set" in t_name or "add" in t_name:
                    score += 4.0
            if spec.app == "reminder" and "nhắc" in lower:
                if "create" in t_name or "set" in t_name or "add" in t_name:
                    score += 4.0

            # Banking matching
            if spec.app == "banking":
                if any(w in lower for w in ("chuyển tiền", "chuyển khoản", "chuyển cho")) and "transfer" in t_name:
                    score += 4.0
                if any(w in lower for w in ("hóa đơn", "tiền điện", "tiền nước", "thanh toán")) and ("bill" in t_name or "pay" in t_name):
                    score += 4.0

            # Music matching
            if spec.app == "music" and any(w in lower for w in ("playlist", "bài hát", "nhạc")):
                if "playlist" in lower and "playlist" in t_name:
                    score += 4.0
                elif "play" in t_name:
                    score += 3.0

            # Generic lexical overlap with tool description
            for word in re.findall(r"\w+", spec.description.lower()):
                if len(word) > 2 and word in lower:
                    score += 0.35

            if score > 0.0:
                scored.append((score, spec))

        scored.sort(key=lambda x: -x[0])
        return [spec for _, spec in scored]

    def _compute_pomdp_utility_decision(
        self,
        belief_state: BeliefState,
        pruned_tools: List[ToolSpec],
    ) -> POMDPAction:
        """Compute optimal POMDP action $a_k = \\arg\\max_a Q(b_k, a)$ over candidate tools."""
        if belief_state.is_out_of_scope_likely:
            return POMDPAction(
                action_type=ActionType.TERMINATE,
                thought="Yêu cầu nằm ngoài phạm vi các ứng dụng đã cài đặt (Out-of-Scope).",
                utility_score=1.0,
            )

        tools_by_app: Dict[str, List[ToolSpec]] = {}
        for t in pruned_tools:
            tools_by_app.setdefault(t.app, []).append(t)

        ranked_tools = self._select_tool_matches_for_utterance(pruned_tools, belief_state)
        estimates: List[UtilityEstimate] = []

        for rank_idx, spec in enumerate(ranked_tools):
            base_goal_align = max(0.2, 1.0 - rank_idx * 0.12)

            if spec.is_read:
                already_read = spec.name in belief_state.observed_read_tools
                if already_read:
                    continue
                est = UtilityEstimate(
                    candidate_action=POMDPAction(
                        action_type=ActionType.CALL_TOOL,
                        tool_name=spec.name,
                        arguments={},
                        thought=f"Gọi công cụ đọc '{spec.name}' để thu thập trạng thái môi trường.",
                    ),
                    goal_alignment=base_goal_align * 0.6,
                    information_gain=self.info_gain_read_bonus,
                    schema_validity_prob=1.0,
                )
                estimates.append(est)
                continue

            # Evaluate write tool
            args, missing_params, needs_read_first = self._synthesize_grounded_arguments(
                spec, belief_state
            )

            # Avoid repeating the exact same write tool call
            already_executed = any(
                s.was_committed_to_env
                and s.observation.ok
                and s.action.tool_name == spec.name
                and s.action.arguments == args
                for s in belief_state.steps
            )
            if already_executed:
                continue

            if needs_read_first:
                # Find the read tool in the same app or x-ref target app to gain information first
                for xref_target in spec.xref_params.values():
                    ref_app = xref_target.split(".")[0]
                    for r_spec in tools_by_app.get(ref_app, []):
                        if r_spec.is_read and r_spec.name not in belief_state.observed_read_tools:
                            read_act = POMDPAction(
                                action_type=ActionType.CALL_TOOL,
                                tool_name=r_spec.name,
                                arguments={},
                                thought=f"Cần lấy danh sách thực thể '{xref_target}' qua '{r_spec.name}' trước khi gọi '{spec.name}'.",
                            )
                            estimates.append(
                                UtilityEstimate(
                                    candidate_action=read_act,
                                    goal_alignment=base_goal_align,
                                    information_gain=self.info_gain_read_bonus + 0.5,
                                    schema_validity_prob=1.0,
                                )
                            )
                continue

            if not missing_params:
                write_act = POMDPAction(
                    action_type=ActionType.CALL_TOOL,
                    tool_name=spec.name,
                    arguments=args,
                    thought=f"Đã xác thực đầy đủ tham số cho '{spec.name}', tiến hành thực thi.",
                )
                estimates.append(
                    UtilityEstimate(
                        candidate_action=write_act,
                        goal_alignment=base_goal_align + 0.4,
                        information_gain=0.1,
                        schema_validity_prob=1.0,
                    )
                )
            else:
                # Missing required parameters that cannot be obtained from read tools -> Ask user
                if not belief_state.asked_missing_info and not belief_state.executed_write_tools:
                    ask_act = POMDPAction(
                        action_type=ActionType.ASK_MISSING,
                        tool_name="sys.ask_missing_info",
                        arguments={
                            "tool_name": spec.name,
                            "params": missing_params,
                        },
                        thought=f"Thiếu tham số bắt buộc {missing_params} cho công cụ '{spec.name}', gọi sys.ask_missing_info.",
                    )
                    estimates.append(
                        UtilityEstimate(
                            candidate_action=ask_act,
                            goal_alignment=self.missing_param_ask_threshold,
                            information_gain=0.25,
                            schema_validity_prob=1.0,
                        )
                    )

        if not estimates:
            return POMDPAction(
                action_type=ActionType.TERMINATE,
                thought="Tất cả các tác vụ trong yêu cầu đã hoàn tất hoặc không còn hành động khả thi.",
                utility_score=1.0,
            )

        estimates.sort(key=lambda e: -e.total_utility)
        best = estimates[0]
        best.candidate_action.utility_score = round(best.total_utility, 4)
        return best.candidate_action

    def plan_next_action(
        self,
        belief_state: BeliefState,
        pruned_tools: List[ToolSpec],
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
        step_idx: int = 1,
    ) -> POMDPAction:
        """Select the next POMDP action $a_k$ combining LLM policy and Expected Utility Grounding."""
        if belief_state.is_out_of_scope_likely and not belief_state.candidate_apps:
            return POMDPAction(
                action_type=ActionType.TERMINATE,
                thought="Out-of-scope request: no installed app matches user intent.",
                utility_score=1.0,
            )

        prompt_bundle = self.build_step_prompt(
            belief_state=belief_state,
            pruned_tools=pruned_tools,
            few_shot_examples=few_shot_examples or [],
            step_idx=step_idx,
        )

        # Try local LLM policy first if weights are loaded
        llm_action = self._generate_with_llm(
            system_prompt=prompt_bundle["system_prompt"],
            user_prompt=prompt_bundle["user_prompt"],
        )
        if llm_action is not None:
            return llm_action

        # Fall back to deterministic POMDP Expected Utility planner
        return self._compute_pomdp_utility_decision(
            belief_state=belief_state,
            pruned_tools=pruned_tools,
        )