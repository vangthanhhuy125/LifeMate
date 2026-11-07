"""Formal POMDP data structures for state tracking, observations, actions, and utilities."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ActionType(str, Enum):
    """Discrete action classes in the LifeMate POMDP action space."""

    CALL_TOOL = "CALL_TOOL"
    ASK_MISSING = "ASK_MISSING"
    TERMINATE = "TERMINATE"


@dataclass
class ToolSpec:
    """Normalized representation of a single tool from `<split>_tools_catalog.json`."""

    name: str
    app: str
    kind: str  # "read" | "write" | "meta"
    description: str
    parameters: Dict[str, Any]
    returns: str = ""
    required_params: List[str] = field(default_factory=list)
    xref_params: Dict[str, str] = field(default_factory=dict)
    fuzzy_params: List[str] = field(default_factory=list)
    one_of_groups: List[List[str]] = field(default_factory=list)

    @property
    def is_read(self) -> bool:
        return self.kind == "read"

    @property
    def is_write(self) -> bool:
        return self.kind == "write"

    @property
    def is_meta(self) -> bool:
        return self.kind == "meta" or self.name == "sys.ask_missing_info"


@dataclass
class POMDPAction:
    """Action $a_k$ selected by the POMDP policy at decision step $k$."""

    action_type: ActionType
    tool_name: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    thought: str = ""
    utility_score: float = 0.0

    def to_tool_call_dict(self) -> Optional[Dict[str, Any]]:
        """Convert to official submission tool call format."""
        if self.action_type == ActionType.TERMINATE or not self.tool_name:
            return None
        return {
            "name": self.tool_name,
            "arguments": self.arguments,
        }


@dataclass
class Observation:
    """Observation $o_k$ returned by the environment or validator guard."""

    step_idx: int
    source: str  # "env_call" | "validate_guard" | "xref_guard" | "system"
    ok: bool
    tool_name: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class TransitionStep:
    """Single step $(b_{k-1}, a_k, o_k)$ in the agent's interaction trajectory."""

    step_idx: int
    action: POMDPAction
    observation: Observation
    was_committed_to_env: bool = False


@dataclass
class UtilityEstimate:
    """Expected utility decomposition for candidate POMDP actions."""

    candidate_action: POMDPAction
    goal_alignment: float = 0.0
    information_gain: float = 0.0
    schema_validity_prob: float = 1.0
    invalid_call_penalty: float = 0.0

    @property
    def total_utility(self) -> float:
        return (
            self.goal_alignment
            + self.information_gain
            - self.invalid_call_penalty
        ) * self.schema_validity_prob


@dataclass
class BeliefState:
    """Factored POMDP belief state $b_k$ maintained across interaction steps."""

    sample_id: str
    raw_utterance: str
    normalized_utterance: str
    current_time: str
    timezone: str
    current_location: Optional[str]
    user_profile: Dict[str, Any]
    installed_apps: List[str]
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    extracted_clues: Dict[str, Any] = field(default_factory=dict)
    candidate_apps: List[str] = field(default_factory=list)
    is_out_of_scope_likely: bool = False
    known_entities: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    observed_read_tools: List[str] = field(default_factory=list)
    executed_write_tools: List[str] = field(default_factory=list)
    steps: List[TransitionStep] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    asked_missing_info: bool = False
    terminated: bool = False

    def register_observation(self, step: TransitionStep) -> None:
        """Update belief state $b_k$ with new action-observation transition."""
        self.steps.append(step)
        obs = step.observation
        if not obs.ok:
            if obs.error:
                self.validation_errors.append(f"{obs.tool_name}: {obs.error}")
            return

        if step.action.action_type == ActionType.ASK_MISSING:
            self.asked_missing_info = True
            self.terminated = True
            return

        tool_name = step.action.tool_name or ""
        if step.was_committed_to_env and tool_name:
            app_prefix = tool_name.split(".")[0] if "." in tool_name else "unknown"
            self._ingest_entities_from_result(app_prefix, tool_name, obs.result)

    def _ingest_entities_from_result(
        self, app_prefix: str, tool_name: str, result: Dict[str, Any]
    ) -> None:
        """Extract entity lists and IDs returned by read tools into `known_entities`."""
        if not isinstance(result, dict):
            return

        if any(k in tool_name for k in ("get_", "list", "search", "status", "check", "find")):
            if tool_name not in self.observed_read_tools:
                self.observed_read_tools.append(tool_name)
        elif tool_name != "sys.ask_missing_info":
            if tool_name not in self.executed_write_tools:
                self.executed_write_tools.append(tool_name)

        for key, value in result.items():
            if key in ("ok", "error", "status_code"):
                continue
            if isinstance(value, list):
                dict_items = [item for item in value if isinstance(item, dict)]
                if dict_items:
                    ref_key = f"{app_prefix}.{key}"
                    self.known_entities[ref_key] = dict_items
                    self.known_entities[key] = dict_items
            elif isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    if isinstance(sub_val, list):
                        dict_items = [item for item in sub_val if isinstance(item, dict)]
                        if dict_items:
                            ref_key = f"{app_prefix}.{sub_key}"
                            self.known_entities[ref_key] = dict_items
                            self.known_entities[sub_key] = dict_items
                if "id" in value:
                    ref_key = f"{app_prefix}.items"
                    self.known_entities.setdefault(ref_key, []).append(value)

    def get_valid_ids_for_xref(self, xref_target: str) -> List[str]:
        """Return all observed valid entity IDs matching an `x-ref` target path."""
        candidates: List[Dict[str, Any]] = []
        if xref_target in self.known_entities:
            candidates.extend(self.known_entities[xref_target])
        short_key = xref_target.split(".")[-1]
        if short_key in self.known_entities:
            candidates.extend(self.known_entities[short_key])
        app_prefix = xref_target.split(".")[0]
        for k, items in self.known_entities.items():
            if k.startswith(f"{app_prefix}."):
                candidates.extend(items)

        valid_ids: List[str] = []
        for item in candidates:
            for id_field in ("id", "device_id", "account_id", "contact_id", "bill_id", "track_id", "playlist_id", "reminder_id", "alarm_id", "email_id", "order_id"):
                val = item.get(id_field)
                if isinstance(val, str) and val not in valid_ids:
                    valid_ids.append(val)
        return valid_ids


@dataclass
class SampleInput:
    """Parsed sample from `<split>.jsonl`."""

    sample_id: str
    audio_rel_path: str
    context: Dict[str, Any]
    history: List[Dict[str, Any]] = field(default_factory=list)
    audio_text: Optional[str] = None
    label: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SampleInput":
        return cls(
            sample_id=str(data["id"]),
            audio_rel_path=str(data.get("audio", "")),
            context=dict(data.get("context", {})),
            history=list(data.get("history", [])),
            audio_text=data.get("audio_text"),
            label=data.get("label"),
        )

    @property
    def installed_app_names(self) -> List[str]:
        installed = self.context.get("installed_apps", {})
        if isinstance(installed, dict):
            return list(installed.keys())
        if isinstance(installed, list):
            return [str(x) for x in installed]
        return []


@dataclass
class ToolCallRecord:
    """Single tool call entry `{name, arguments}`."""

    name: str
    arguments: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments}


@dataclass
class SubmissionRecord:
    """Single line of `submission.jsonl`."""

    sample_id: str
    tool_calls: List[Dict[str, Any]]
    state_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.sample_id,
            "tool_calls": self.tool_calls,
            "state_hash": self.state_hash,
        }