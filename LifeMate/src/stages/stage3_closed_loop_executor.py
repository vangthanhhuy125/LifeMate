import time
from typing import Dict, Any, List, Tuple
from src.utils.pomdp_types import Stage1OutputBundle, POMDPAction, POMDPTransitionStep

class Stage3_ValidateBeforeCallExecutor:
    """
    ================================================================================
    STAGE 3: VALIDATE-BEFORE-CALL SELF-CORRECTION & CLOSED-LOOP EXECUTION
    - [Contribution 1]:
      Trong đề bài LifeMate, mọi lệnh gọi `env.call()` bị lỗi Schema đều bị ghi vào
      log nộp bài, làm tăng U (unmatched calls) khiến Precision P = M / (M + U) sụt giảm.
      Stage 3 giải quyết triệt để bằng màng lọc kép:
        (1) Kiểm tra ràng buộc `x-ref` Read-before-Write.
        (2) Gọi `env.validate(tool_name, arguments)` (hàm này KHÔNG ghi vào call log).
      Nếu phát hiện bất kỳ lỗi nào, Stage 3 trả thông báo lỗi về Stage 2 để LLM tự sửa
      (Self-Correction) trong vòng lặp nội bộ trước khi thực thi `env.call()` thật!
    ================================================================================
    """
    def __init__(self, stage3_cfg: Dict[str, Any]):
        self.max_steps = stage3_cfg.get("max_pomdp_steps", 6)
        self.validate_before_call = stage3_cfg.get("validate_before_call", True)
        self.max_retries = stage3_cfg.get("max_self_correction_retries", 2)
        self.auto_ask_missing = stage3_cfg.get("auto_ask_missing_on_schema_error", True)

    def run_pomdp_episode(
        self,
        env: Any,
        stage1_bundle: Stage1OutputBundle,
        stage2_planner: Any,
        episode_start_time: float,
        max_time_sec: float = 28.0
    ) -> Tuple[List[Dict[str, Any]], str, List[POMDPTransitionStep]]:
        history_H: List[POMDPTransitionStep] = []
        has_executed_read_tool = False

        for step_k in range(1, self.max_steps + 1):
            if (time.time() - episode_start_time) >= max_time_sec:
                break

            validation_feedback = None
            approved_call: Dict[str, Any] = None
            retries_used = 0

            for attempt in range(self.max_retries + 1):
                action_ak: POMDPAction = stage2_planner.sample_action(
                    bundle=stage1_bundle,
                    history_H=history_H,
                    validation_error_feedback=validation_feedback
                )

                # Nếu Agent chọn kết thúc (a_k = EmptySet)
                if action_ak.action_type == "TERMINATE":
                    approved_call = None
                    break

                candidate_env_call = action_ak.to_env_tool_call()
                if not candidate_env_call:
                    approved_call = None
                    break

                t_name = candidate_env_call["name"]
                t_args = candidate_env_call["arguments"]

                # Nếu không bật chế độ Validate-Before-Call (ví dụ khi chạy Ablation 1)
                if not self.validate_before_call:
                    approved_call = candidate_env_call
                    break

                # Kiểm tra 1: Ràng buộc x-ref (Phải gọi ít nhất 1 tool read trước khi truyền ID vào x-ref)
                if t_name in stage1_bundle.xref_tool_constraints and not has_executed_read_tool:
                    xref_fields = list(stage1_bundle.xref_tool_constraints[t_name].keys())
                    if any(f in t_args for f in xref_fields):
                        validation_feedback = (
                            f"Công cụ '{t_name}' chứa tham số x-ref {xref_fields}. "
                            f"Bạn BẮT BUỘC phải gọi công cụ đọc (read/get_status/list) trước để lấy ID thực tế!"
                        )
                        retries_used += 1
                        continue

                # Kiểm tra 2: Gọi env.validate() của zaic_tools (Không làm tăng U trong call log)
                if hasattr(env, "validate"):
                    val_res = env.validate(t_name, t_args)
                    if not val_res.get("ok", False):
                        err_msg = val_res.get("error", "Lỗi định dạng JSON Schema")
                        validation_feedback = f"env.validate('{t_name}') thất bại: {err_msg}"
                        retries_used += 1
                        continue

                # Đã vượt qua toàn bộ màng lọc tiền kiểm tra!
                approved_call = candidate_env_call
                break

            # Nếu không tìm được lệnh hợp lệ sau số lần tự sửa hoặc Agent chọn TERMINATE
            if approved_call is None:
                break

            # Thực thi chính thức trên môi trường ẩn E (zaic_tools.ToolEnv)
            obs_ok = env.call(approved_call["name"], approved_call["arguments"])

            # Đánh dấu nếu vừa chạy một công cụ đọc dữ liệu
            if self._is_read_tool(approved_call["name"], stage1_bundle.pruned_schema_SK):
                has_executed_read_tool = True

            history_H.append(
                POMDPTransitionStep(
                    step_k=step_k,
                    action=approved_call,
                    observation=obs_ok,
                    validation_retries=retries_used
                )
            )

            # Nếu đã gọi sys.ask_missing_info thì dừng chu trình để chờ người dùng trả lời ở lượt sau
            if approved_call["name"] == "sys.ask_missing_info":
                break

        return env.get_tool_calls(), env.get_state_hash(), history_H

    @staticmethod
    def _is_read_tool(tool_name: str, schema_list: List[Dict[str, Any]]) -> bool:
        for t in schema_list:
            if t.get("name") == tool_name:
                return t.get("kind") == "read"
        return any(k in tool_name for k in ["get_", "list", "search", "read", "status"])
