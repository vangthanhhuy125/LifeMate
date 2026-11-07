"""Stage 1: Multimodal Perception, NLU Normalization, Schema Pruning, and RAG Grounding."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.modules.acoustic_frontend import AcousticFrontend
from src.modules.app_overlap_retriever import AppOverlapRetriever
from src.modules.context_memory import ContextMemoryManager
from src.modules.dynamic_schema_pruner import DynamicSchemaPruner
from src.modules.nlu_normalizers import NLUNormalizer
from src.utils.pomdp_types import BeliefState, SampleInput, ToolSpec


class Stage1PerceptionAndGrounding:
    """Executes Stage 1 of the LifeMate architecture: Audio -> Normalized Text -> Grounded BeliefState."""

    def __init__(
        self,
        config: Dict[str, Any],
        catalog_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.config = config
        self.acoustic = AcousticFrontend(config)
        self.normalizer = NLUNormalizer(config)
        self.memory = ContextMemoryManager(self.normalizer)
        self.pruner = DynamicSchemaPruner(config, catalog_data=catalog_data)
        self.retriever = AppOverlapRetriever(config)

    def update_catalog(self, catalog_data: Dict[str, Any]) -> None:
        """Load or refresh the split's tool catalog."""
        self.pruner.load_catalog_dict(catalog_data)

    def run(
        self,
        sample: SampleInput,
        data_root: str | Path = ".",
        split_name: str = "public_test",
    ) -> Dict[str, Any]:
        """Process a single sample through ASR, NLU, Schema Pruning, and Few-Shot Retrieval."""
        audio_full_path = Path(data_root) / split_name / sample.audio_rel_path

        # 1. Acoustic transcription
        asr_out = self.acoustic.transcribe(
            audio_path=audio_full_path,
            oracle_text=sample.audio_text,
        )
        raw_utterance = asr_out["raw_transcript"]
        clean_utterance = asr_out["transcript"]

        # 2. NLU normalization & clue extraction
        nlu_out = self.normalizer.normalize(clean_utterance, sample.context)
        normalized_utterance = nlu_out["normalized_text"]
        extracted_clues = nlu_out["clues"]

        # 3. Dynamic Schema Pruning & Out-of-Scope Detection
        prune_out = self.pruner.prune_for_sample(
            sample_context=sample.context,
            normalized_utterance=normalized_utterance,
            extracted_clues=extracted_clues,
        )
        pruned_tools: List[ToolSpec] = prune_out["tools"]
        candidate_apps: List[str] = prune_out["candidate_apps"]
        is_out_of_scope_likely: bool = prune_out["is_out_of_scope_likely"]

        # 4. Initialize POMDP Belief State b_0
        belief_state: BeliefState = self.memory.initialize_belief_state(
            sample=sample,
            raw_utterance=raw_utterance,
            normalized_utterance=normalized_utterance,
            extracted_clues=extracted_clues,
            candidate_apps=candidate_apps,
            is_out_of_scope_likely=is_out_of_scope_likely,
        )

        # 5. Few-Shot Trajectory Retrieval
        few_shot_examples = self.retriever.retrieve(
            utterance=normalized_utterance,
            installed_apps=sample.installed_app_names,
            candidate_apps=candidate_apps,
            exclude_sample_id=sample.sample_id,
        )

        return {
            "belief_state": belief_state,
            "pruned_tools": pruned_tools,
            "pruned_schemas_json": prune_out["schemas_json"],
            "few_shot_examples": few_shot_examples,
            "asr_meta": asr_out,
            "pruning_stats": prune_out["stats"],
        }