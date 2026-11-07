from pathlib import Path
from typing import Dict, Any
from src.utils.pomdp_types import UserContextC, Stage1OutputBundle
from src.modules.acoustic_frontend import AcousticFrontendASR
from src.modules.dynamic_schema_pruner import DynamicSchemaPruner
from src.modules.app_overlap_retriever import AppOverlapTrajectoryRetriever

class Stage1_PerceptionAndGrounding:
    """
    ================================================================================
    STAGE 1: MULTIMODAL PERCEPTION, DYNAMIC SCHEMA PRUNING & APP-OVERLAP RAG
    - Thực hiện:
      1. Nhận diện giọng nói tiếng Việt 16kHz + Chuẩn hóa từ lóng/tiếng Anh (A).
      2. Khởi tạo cấu trúc ngữ cảnh C = (t, l, P, K).
      3. Cắt tỉa động lược đồ công cụ S(K) dựa trên K = installed_apps (Contribution 2).
      4. Truy xuất Top-K quỹ đạo mẫu có giao thoa ứng dụng cao nhất (Contribution 3).
    ================================================================================
    """
    def __init__(self, stage1_cfg: Dict[str, Any]):
        self.cfg = stage1_cfg
        self.asr_cfg = stage1_cfg.get("asr", {})
        self.pruning_cfg = stage1_cfg.get("schema_pruning", {})
        self.rag_cfg = stage1_cfg.get("few_shot_rag", {})

        self.acoustic_asr = AcousticFrontendASR(self.asr_cfg)
        self.retriever = AppOverlapTrajectoryRetriever(
            top_k=self.rag_cfg.get("top_k", 3),
            app_overlap_boost=self.rag_cfg.get("app_overlap_boost", 1.35)
        )
        if self.rag_cfg.get("enabled", True) and self.rag_cfg.get("index_path"):
            self.retriever.load_index(self.rag_cfg["index_path"])

    def process(
        self,
        raw_sample: Dict[str, Any],
        split_dir: Path,
        full_catalog: Dict[str, Any]
    ) -> Stage1OutputBundle:
        sample_id = raw_sample.get("id", "unknown")

        # 1. Trích xuất câu lệnh A (Từ file .wav hoặc từ Oracle transcript khi chạy Ablation)
        if self.asr_cfg.get("use_oracle_transcript", False) and "audio_text" in raw_sample:
            a_txt = raw_sample["audio_text"]
            utterance_A = a_txt[-1] if isinstance(a_txt, list) and a_txt else str(a_txt)
            utterance_A = self.acoustic_asr.normalize_code_switching(utterance_A)
        else:
            wav_full_path = str(split_dir / raw_sample.get("audio", ""))
            utterance_A = self.acoustic_asr.transcribe_waveform(wav_full_path)

        # 2. Đóng gói ngữ cảnh C = (t, l, P, K)
        context_C = UserContextC.from_dict(raw_sample.get("context", {}))

        # 3. Dynamic Schema Pruning: Cắt tỉa S(K) và bóc tách ràng buộc x-ref (Contribution 2)
        pruned_SK, xref_constraints = DynamicSchemaPruner.prune_and_analyze(
            full_catalog=full_catalog,
            installed_apps=context_C.installed_apps,
            enable_pruning=self.pruning_cfg.get("enabled", True)
        )

        # 4. Few-Shot RAG nội bộ có trọng số App-Overlap (Contribution 3)
        few_shots = []
        if self.rag_cfg.get("enabled", True):
            active_apps = list(context_C.installed_apps.keys())
            few_shots = self.retriever.retrieve_top_k(utterance_A, active_apps)

        return Stage1OutputBundle(
            sample_id=sample_id,
            utterance_A=utterance_A,
            context_C=context_C,
            dialogue_history=raw_sample.get("history", []),
            pruned_schema_SK=pruned_SK,
            xref_tool_constraints=xref_constraints,
            few_shot_trajectories=few_shots
        )
