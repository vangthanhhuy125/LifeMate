"""Validate-Before-Call Self-Correction Guard for Schema, X-One-Of, and X-Ref Integrity."""

import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from src.utils.pomdp_types import ActionType, BeliefState, POMDPAction, ToolSpec


class ValidatorGuard:
    """Intercepts candidate POMDP actions before `env.call()` to guarantee zero invalid tool calls."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        exec_cfg = self.config.get("execution", self.config)
        self.enable_validate_guard = bool(exec_cfg.get("enable_validate_guard", True))
        self.enable_xref_guard = bool(exec_cfg.get("enable_xref_guard", True))
        self.prevent_duplicate_calls = bool(exec_cfg.get("prevent_duplicate_calls", True))
        self.enforce_read_before_write_xref = bool(
            exec_cfg.get("enforce_read_before_write_xref", True)
        )

    @staticmethod
    def _normalize_arg_value(val: Any) -> Any:
        """Recursively apply Unicode NFC normalization to all string arguments."""
        if isinstance(val, str):
            return unicodedata.normalize("NFC", val.strip())
        if isinstance(val, list):
            return [ValidatorGuard._normalize_arg_value(x) for x in val]
        if isinstance(val, dict):
            return {
                str(k): ValidatorGuard._normalize_arg_value(v) for k, v in val.items()
            }
        return val

    def sanitize_arguments(
        self, arguments: Dict[str, Any], tool_spec: Optional[ToolSpec] = None
    ) -> Dict[str, Any]:
        """Clean None values, coerce basic scalar types, and normalize strings to NFC."""
        if not isinstance(arguments, dict):
            return {}
        cleaned: Dict[str, Any] = {}
        props = (tool_spec.parameters.get("properties") or {}) if tool_spec else {}

        for key, raw_val in arguments.items():
            if raw_val is None:
                continue
            val = self._normalize_arg_value(raw_val)
            prop_schema = props.get(key) if isinstance(props, dict) else None
            if isinstance(prop_schema, dict):
                expected_type = prop_schema.get("type")
                if expected_type == "integer" and isinstance(val, (str, float)):
                    try:
                        val = int(float(val))
                    except (ValueError, TypeError):
                        pass
                elif expected_type == "number" and isinstance(val, str):
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        pass
                elif expected_type == "boolean" and isinstance(val, str):
                    if val.lower() in ("true", "1", "yes", "on", "bật"):
                        val = True
                    elif val.lower() in ("false", "0", "no", "off", "tắt"):
                        val = False
                elif expected_type == "string" and not isinstance(val, str):
                    val = str(val)
            cleaned[str(key)] = val
        return cleaned

    def check_missing_required_and_one_of(
        self, tool_spec: ToolSpec, arguments: Dict[str, Any]
    ) -> List[str]:
        """Return list of missing `required` or `x-one-of` parameter names."""
        missing: List[str] = []
        for req_param in tool_spec.required_params:
            val = arguments.get(req_param)
            if val is None or (isinstance(val, str) and val.strip() == ""):
                missing.append(req_param)

        for group in tool_spec.one_of_groups:
            if not group:
                continue
            satisfied = any(
                arguments.get(p) is not None
                and not (isinstance(arguments.get(p), str) and str(arguments.get(p)).strip() == "")
                for p in group
            )
            if not satisfied:
                for p in group:
                    if p not in missing:
                        missing.append(p)
        return missing

    def check_xref_grounding(
        self,
        tool_spec: ToolSpec,
        arguments: Dict[str, Any],
        belief_state: BeliefState,
        available_tools: Dict[str, ToolSpec],
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Verify that any `x-ref` parameter value exists in `belief_state.known_entities`.

        Returns:
            Tuple of (is_valid, error_message, suggested_read_tool_name).
        """
        if not self.enable_xref_guard or not tool_spec.xref_params:
            return True, None, None

        for param_name, xref_target in tool_spec.xref_params.items():
            if param_name not in arguments:
                continue
            candidate_id = str(arguments[param_name])
            valid_ids = belief_state.get_valid_ids_for_xref(xref_target)

            # Find a corresponding read tool for this x-ref target
            ref_app = xref_target.split(".")[0]
            suggested_read_tool: Optional[str] = None
            for t_name, t_spec in available_tools.items():
                if t_spec.app == ref_app and t_spec.is_read and t_name not in belief_state.observed_read_tools:
                    suggested_read_tool = t_name
                    break

            if not valid_ids:
                if self.enforce_read_before_write_xref and suggested_read_tool:
                    return (
                        False,
                        f"Tham số '{param_name}' yêu cầu x-ref từ '{xref_target}' nhưng chưa gọi công cụ đọc '{suggested_read_tool}'.",
                        suggested_read_tool,
                    )
            elif candidate_id not in valid_ids:
                return (
                    False,
                    f"Giá trị '{candidate_id}' của tham số '{param_name}' không khớp với danh sách ID hợp lệ đã quan sát: {valid_ids}.",
                    suggested_read_tool,
                )

        return True, None, None

    def validate_action(
        self,
        action: POMDPAction,
        belief_state: BeliefState,
        available_tools: Dict[str, ToolSpec],
        env: Any,
    ) -> Dict[str, Any]:
        """Run full pre-call validation suite (installed app check, duplicate check, x-ref check, `env.validate`)."""
        if action.action_type == ActionType.TERMINATE or not action.tool_name:
            return {"ok": True, "action": action, "source": "terminate"}

        tool_name = action.tool_name
        tool_spec = available_tools.get(tool_name)

        # 1. Check if tool is allowed in current sample's installed_apps
        if tool_spec is None and tool_name != "sys.ask_missing_info":
            return {
                "ok": False,
                "source": "validate_guard",
                "error": f"Công cụ '{tool_name}' không tồn tại hoặc không thuộc installed_apps {belief_state.installed_apps}.",
                "suggested_read_tool": None,
            }

        sanitized_args = self.sanitize_arguments(action.arguments, tool_spec)
        action.arguments = sanitized_args

        # 2. Prevent duplicate identical tool calls in the same trajectory
        if self.prevent_duplicate_calls:
            for prev_step in belief_state.steps:
                if (
                    prev_step.was_committed_to_env
                    and prev_step.observation.ok
                    and prev_step.action.tool_name == tool_name
                    and prev_step.action.arguments == sanitized_args
                ):
                    return {
                        "ok": False,
                        "source": "validate_guard",
                        "error": f"Lệnh '{tool_name}' với cùng bộ tham số đã được thực thi thành công trước đó.",
                        "suggested_read_tool": None,
                    }

        # 3. Check required & x-one-of parameters locally
        if tool_spec is not None and tool_name != "sys.ask_missing_info":
            missing_params = self.check_missing_required_and_one_of(tool_spec, sanitized_args)
            if missing_params:
                return {
                    "ok": False,
                    "source": "validate_guard",
                    "error": f"Thiếu tham số bắt buộc cho '{tool_name}': {missing_params}",
                    "missing_params": missing_params,
                    "suggested_read_tool": None,
                }

            # 4. Check x-ref entity grounding
            xref_ok, xref_err, suggested_read = self.check_xref_grounding(
                tool_spec=tool_spec,
                arguments=sanitized_args,
                belief_state=belief_state,
                available_tools=available_tools,
            )
            if not xref_ok:
                return {
                    "ok": False,
                    "source": "xref_guard",
                    "error": xref_err,
                    "suggested_read_tool": suggested_read,
                }

        # 5. Official environment schema pre-check via `env.validate(tool_name, arguments)`
        if self.enable_validate_guard and env is not None and hasattr(env, "validate"):
            val_res = env.validate(tool_name, sanitized_args)
            if isinstance(val_res, dict) and not val_res.get("ok", True):
                return {
                    "ok": False,
                    "source": "validate_guard",
                    "error": str(val_res.get("error", "Lỗi kiểm tra schema từ env.validate")),
                    "suggested_read_tool": None,
                }

        return {
            "ok": True,
            "source": "validate_guard",
            "action": action,
        }