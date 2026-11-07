"""Stage 3: Closed-Loop POMDP Tool Executor with Validate-Before-Call Self-Correction."""

import time
from typing import Any, Dict, List, Optional

from src.modules.validator_guard import ValidatorGuard
from src.stages.stage2_pomdp_planner import Stage2POMDPPlanner
from src.utils.pomdp_types import (
    ActionType,
    BeliefState,
    Observation,
    POMDPAction,
    ToolSpec,
    TransitionStep,
)


class Stage3ClosedLoopExecutor:
    """Executes the POMDP perception-planning-validation-execution loop on a single `ToolEnv`."""

    def __init__(
        self,
        config: Dict[str, Any],
        planner: Stage2POMDPPlanner,
        validator: Optional[ValidatorGuard] = None,
    ) -> None:
        self.config = config
        self.planner = planner
        self.validator = validator or ValidatorGuard(config)

        exec_cfg = config.get("execution", {})
        hw_cfg = config.get("hardware_constraints", {})
        self.max_steps = int(exec_cfg.get("max_steps", 6))
        self.max_retries = int(exec_cfg.get("max_self_correction_retries", 2))
        self.max_sample_time_sec = float(hw_cfg.get("max_sample_time_sec", 28.0))

    def execute_episode(
        self,
        env: Any,
        belief_state: BeliefState,
        pruned_tools: List[ToolSpec],
        few_shot_examples: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run closed-loop interaction with `env` until task completion or budget exhaustion.

        IMPORTANT RULE COMPLIANCE:
        - `env` is created ONCE outside/before this loop and passed in.
        - `env.validate()` is called first to catch schema errors without polluting the call log.
        - `env.call()` is only invoked after validation passes.
        - Final submission outputs come directly from `env.get_tool_calls()` and `env.get_state_hash()`.
        """
        start_time = time.monotonic()
        tools_map: Dict[str, ToolSpec] = {t.name: t for t in pruned_tools}
        examples = few_shot_examples or []

        # Fast-path for strict out-of-scope samples when no installed apps can handle the request
        if belief_state.is_out_of_scope_likely and not belief_state.candidate_apps:
            belief_state.terminated = True
            return {
                "tool_calls": env.get_tool_calls(),
                "state_hash": env.get_state_hash(),
                "belief_state": belief_state,
                "elapsed_sec": round(time.monotonic() - start_time, 4),
            }

        step_idx = 0
        while step_idx < self.max_steps and not belief_state.terminated:
            if (time.monotonic() - start_time) >= self.max_sample_time_sec:
                belief_state.terminated = True
                break

            step_idx += 1
            retry_count = 0
            committed = False

            while retry_count <= self.max_retries:
                if (time.monotonic() - start_time) >= self.max_sample_time_sec:
                    belief_state.terminated = True
                    break

                candidate_action: POMDPAction = self.planner.plan_next_action(
                    belief_state=belief_state,
                    pruned_tools=pruned_tools,
                    few_shot_examples=examples,
                    step_idx=step_idx,
                )

                if (
                    candidate_action.action_type == ActionType.TERMINATE
                    or not candidate_action.tool_name
                ):
                    belief_state.terminated = True
                    break

                val_outcome = self.validator.validate_action(
                    action=candidate_action,
                    belief_state=belief_state,
                    available_tools=tools_map,
                    env=env,
                )

                if val_outcome["ok"]:
                    # Validation succeeded -> Execute on the official environment
                    exec_res = env.call(candidate_action.tool_name, candidate_action.arguments)
                    is_ok = bool(exec_res.get("ok", True)) if isinstance(exec_res, dict) else True
                    err_msg = (
                        str(exec_res.get("error"))
                        if isinstance(exec_res, dict) and not is_ok
                        else None
                    )
                    obs = Observation(
                        step_idx=step_idx,
                        source="env_call",
                        ok=is_ok,
                        tool_name=candidate_action.tool_name,
                        arguments=dict(candidate_action.arguments),
                        result=exec_res if isinstance(exec_res, dict) else {"result": exec_res},
                        error=err_msg,
                    )
                    belief_state.register_observation(
                        TransitionStep(
                            step_idx=step_idx,
                            action=candidate_action,
                            observation=obs,
                            was_committed_to_env=True,
                        )
                    )
                    committed = True
                    break

                # Validation failed -> Check if validator suggests a prerequisite read tool
                suggested_read = val_outcome.get("suggested_read_tool")
                if suggested_read and suggested_read in tools_map:
                    read_action = POMDPAction(
                        action_type=ActionType.CALL_TOOL,
                        tool_name=suggested_read,
                        arguments={},
                        thought=f"Tự động gọi công cụ đọc '{suggested_read}' để lấy ID thực thể x-ref trước khi ghi.",
                        utility_score=1.0,
                    )
                    read_val = self.validator.validate_action(
                        action=read_action,
                        belief_state=belief_state,
                        available_tools=tools_map,
                        env=env,
                    )
                    if read_val["ok"]:
                        exec_res = env.call(read_action.tool_name, read_action.arguments)
                        is_ok = bool(exec_res.get("ok", True)) if isinstance(exec_res, dict) else True
                        obs = Observation(
                            step_idx=step_idx,
                            source="env_call",
                            ok=is_ok,
                            tool_name=read_action.tool_name,
                            arguments={},
                            result=exec_res if isinstance(exec_res, dict) else {"result": exec_res},
                        )
                        belief_state.register_observation(
                            TransitionStep(
                                step_idx=step_idx,
                                action=read_action,
                                observation=obs,
                                was_committed_to_env=True,
                            )
                        )
                        committed = True
                        break

                # If missing required parameters cannot be resolved via read tools, convert to sys.ask_missing_info
                missing_params = val_outcome.get("missing_params")
                if (
                    missing_params
                    and retry_count == self.max_retries
                    and "sys.ask_missing_info" in tools_map
                ):
                    ask_action = POMDPAction(
                        action_type=ActionType.ASK_MISSING,
                        tool_name="sys.ask_missing_info",
                        arguments={
                            "tool_name": candidate_action.tool_name,
                            "params": list(missing_params),
                        },
                        thought=f"Thiếu tham số bắt buộc {missing_params} cho '{candidate_action.tool_name}', chuyển sang hỏi người dùng.",
                        utility_score=0.9,
                    )
                    ask_val = self.validator.validate_action(
                        action=ask_action,
                        belief_state=belief_state,
                        available_tools=tools_map,
                        env=env,
                    )
                    if ask_val["ok"]:
                        exec_res = env.call(ask_action.tool_name, ask_action.arguments)
                        obs = Observation(
                            step_idx=step_idx,
                            source="env_call",
                            ok=True,
                            tool_name=ask_action.tool_name,
                            arguments=dict(ask_action.arguments),
                            result=exec_res if isinstance(exec_res, dict) else {},
                        )
                        belief_state.register_observation(
                            TransitionStep(
                                step_idx=step_idx,
                                action=ask_action,
                                observation=obs,
                                was_committed_to_env=True,
                            )
                        )
                        committed = True
                        break

                # Record validation failure in belief state so Planner self-corrects on next retry
                err_obs = Observation(
                    step_idx=step_idx,
                    source=str(val_outcome.get("source", "validate_guard")),
                    ok=False,
                    tool_name=candidate_action.tool_name,
                    arguments=dict(candidate_action.arguments),
                    error=str(val_outcome.get("error", "Schema validation error")),
                )
                belief_state.register_observation(
                    TransitionStep(
                        step_idx=step_idx,
                        action=candidate_action,
                        observation=err_obs,
                        was_committed_to_env=False,
                    )
                )
                retry_count += 1

            if not committed and not belief_state.terminated:
                # Exhausted retries for this step without a valid call -> terminate safely
                belief_state.terminated = True

        return {
            "tool_calls": env.get_tool_calls(),
            "state_hash": env.get_state_hash(),
            "belief_state": belief_state,
            "elapsed_sec": round(time.monotonic() - start_time, 4),
        }