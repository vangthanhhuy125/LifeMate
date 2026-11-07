import time
from pathlib import Path
from typing import Dict, Any, Tuple, List
from src.stages.stage1_perception_and_grounding import Stage1_PerceptionAndGrounding
from src.stages.stage2_pomdp_planner import Stage2_POMDPPolicyPlanner
from src.stages.stage3_closed_loop_executor import Stage3_ValidateBeforeCallExecutor
from src.utils.pomdp_types import POMDPTransitionStep

class LifeMatePOMDPPipeline:
    """
    Bộ điều phối trung tâm kết nối liền mạch 3 Stage theo đúng định nghĩa POMDP:
      Stage 1 (Perception & Grounding) -> Stage 2 (Policy Planner) <-> Stage 3 (Closed-Loop Executor).
    """
    def __init__(self, full_config: Dict[str, Any]):
        self.config = full_config
        self.max_time_sec = full_config.get("experiment", {}).get("max_time_per_sample_sec", 28.0)

        self.stage1 = Stage1_PerceptionAndGrounding(
            full_config.get("stage1_perception_and_grounding", {})
        )
        self.stage2 = Stage2_POMDPPolicyPlanner(
            full_config.get("stage2_pomdp_planner", {})
        )
        self.stage3 = Stage3_ValidateBeforeCallExecutor(
            full_config.get("stage3_closed_loop_executor", {})
        )

    def infer_single_sample(
        self,
        raw_sample: Dict[str, Any],
        env: Any,
        split_dir: Path,
        full_catalog: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[POMDPTransitionStep]]:
        """
        Thực thi trọn vẹn 1 mẫu dữ liệu và trả về:
          - submission_record: {"id": ..., "tool_calls": [...], "state_hash": "..."}
          - pomdp_trace: Lịch sử chi tiết từng bước (a_k, o_k) phục vụ phân tích bài báo.
        """
        start_time = time.time()

        # --- CHẠY STAGE 1 ---
        stage1_bundle = self.stage1.process(
            raw_sample=raw_sample,
            split_dir=split_dir,
            full_catalog=full_catalog
        )

        # --- CHẠY VÒNG LẶP KÍN STAGE 2 <-> STAGE 3 ---
        tool_calls, state_hash, pomdp_trace = self.stage3.execute_episode(
            env=env,
            stage1_bundle=stage1_bundle,
            stage2_planner=self.stage2,
            episode_start_time=start_time
        ) if hasattr(self.stage3, "execute_episode") else self.stage3.run_pomdp_episode(
            env=env,
            stage1_bundle=stage1_bundle,
            stage2_planner=self.stage2,
            episode_start_time=start_time,
            max_time_sec=self.max_time_sec
        )

        submission_record = {
            "id": raw_sample.get("id"),
            "tool_calls": tool_calls,
            "state_hash": state_hash
        }
        return submission_record, pomdp_trace
