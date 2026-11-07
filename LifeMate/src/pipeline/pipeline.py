"""Abstract base pipeline interface for Zalo AI Challenge 2026 voice agents."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

from src.utils.pomdp_types import SampleInput, SubmissionRecord


class BaseAgentPipeline(ABC):
    """Base interface enforcing one `ToolEnv` per sample and standard `SubmissionRecord` output."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def run_sample(
        self,
        sample: SampleInput | Dict[str, Any],
        env: Any,
        data_root: str | Path = ".",
        split_name: str = "public_test",
    ) -> SubmissionRecord:
        """Execute pipeline on a single sample with its dedicated `ToolEnv` instance."""
        raise NotImplementedError