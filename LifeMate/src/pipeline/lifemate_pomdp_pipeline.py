"""Complete 3-Stage Resource-Constrained State-Grounded POMDP Pipeline for LifeMate."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from src.modules.validator_guard import ValidatorGuard
from src.pipeline.pipeline import BaseAgentPipeline
from src.stages.stage1_perception_and_grounding import Stage1PerceptionAndGrounding
from src.stages.stage2_pomdp_planner import Stage2POMDPPlanner
from src.stages.stage3_closed_loop_executor import Stage3ClosedLoopExecutor
from src.utils.pomdp_types import SampleInput, SubmissionRecord
from src.utils.reproducibility import enforce_reproducibility, load_yaml_config


class LifeMatePOMDPPipeline(BaseAgentPipeline):
    """Orchestrates Stage 1 (Perception/Grounding), Stage 2 (POMDP Planner), and Stage 3 (Closed-Loop Execution)."""

    def __init__(
        self,
        config: Dict[str, Any] | str | Path,
        catalog_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        if isinstance(config, (str, Path)):
            cfg_dict = load_yaml_config(config)
        else:
            cfg_dict = dict(config)

        super().__init__(cfg_dict)
        self.repro_meta = enforce_reproducibility(self.config)

        self.stage1 = Stage1PerceptionAndGrounding(self.config, catalog_data=catalog_data)
        self.stage2 = Stage2POMDPPlanner(self.config)
        self.validator = ValidatorGuard(self.config)
        self.stage3 = Stage3ClosedLoopExecutor(
            config=self.config,
            planner=self.stage2,
            validator=self.validator,
        )

    def load_split_catalog(self, data_root: str | Path, split_name: str) -> None:
        """Load `<split>/<split>_tools_catalog.json` into Stage 1's DynamicSchemaPruner."""
        catalog_path = Path(data_root) / split_name / f"{split_name}_tools_catalog.json"
        if catalog_path.exists():
            with open(catalog_path, "r", encoding="utf-8") as f:
                catalog_data = json.load(f)
            self.stage1.update_catalog(catalog_data)

    def run_sample(
        self,
        sample: SampleInput | Dict[str, Any],
        env: Any,
        data_root: str | Path = ".",
        split_name: str = "public_test",
    ) -> SubmissionRecord:
        """Execute the 3-stage POMDP pipeline on a single sample and return its `SubmissionRecord`."""
        parsed_sample = (
            sample if isinstance(sample, SampleInput) else SampleInput.from_dict(sample)
        )

        # Stage 1: Acoustic ASR, Vietnamese NLU Normalization, Dynamic Schema Pruning, and RAG
        s1_out = self.stage1.run(
            sample=parsed_sample,
            data_root=data_root,
            split_name=split_name,
        )

        # Stage 2 & Stage 3: Closed-Loop POMDP Planning and Validated Execution
        s3_out = self.stage3.execute_episode(
            env=env,
            belief_state=s1_out["belief_state"],
            pruned_tools=s1_out["pruned_tools"],
            few_shot_examples=s1_out["few_shot_examples"],
        )

        return SubmissionRecord(
            sample_id=parsed_sample.sample_id,
            tool_calls=s3_out["tool_calls"],
            state_hash=s3_out["state_hash"],
        )