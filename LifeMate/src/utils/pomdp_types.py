from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

@dataclass
class UserContextC:
    """Biểu diễn ngữ cảnh C = (t, l, P, K) trong Problem Formulation."""
    current_time: str
    timezone: str
    current_location: Optional[str]
    user_profile: Dict[str, Any]
    installed_apps: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserContextC":
        return cls(
            current_time=data.get("current_time", ""),
            timezone=data.get("timezone", "Asia/Ho_Chi_Minh"),
            current_location=data.get("current_location"),
            user_profile=data.get("user_profile", {}),
            installed_apps=data.get("installed_apps", {})
        )

@dataclass
class Stage1OutputBundle:
    """Đầu ra chuẩn hóa của Stage 1 chuyển giao sang Stage 2."""
    sample_id: str
    utterance_A: str
    context_C: UserContextC
    dialogue_history: List[Dict[str, Any]]
    pruned_schema_SK: List[Dict[str, Any]]
    xref_tool_constraints: Dict[str, Dict[str, str]]
    few_shot_trajectories: List[Dict[str, Any]]

@dataclass
class POMDPAction:
    """
    Biểu diễn hành động a_k thuộc A = {Call, AskMissing, EmptySet (TERMINATE)}.
    """
    action_type: str  # "CALL" | "ASK_MISSING" | "TERMINATE"
    thought: str = ""
    tool_name: Optional[str] = None
    arguments: Dict[str, Any] = field(default_factory=dict)
    missing_params: List[str] = field(default_factory=list)

    def to_env_tool_call(self) -> Optional[Dict[str, Any]]:
        if self.action_type == "CALL" and self.tool_name:
            return {"name": self.tool_name, "arguments": self.arguments}
        elif self.action_type == "ASK_MISSING" and self.tool_name:
            return {
                "name": "sys.ask_missing_info",
                "arguments": {
                    "tool_name": self.tool_name,
                    "params": self.missing_params
                }
            }
        return None

@dataclass
class POMDPTransitionStep:
    """Lưu vết một bước chuyển trạng thái (a_k, o_k) trong lịch sử tương tác H_k."""
    step_k: int
    action: Dict[str, Any]
    observation: Dict[str, Any]
    validation_retries: int = 0
