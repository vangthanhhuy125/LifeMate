import yaml
import argparse
from pathlib import Path
from src.utils.reproducibility import lock_deterministic_environment
from src.modules.app_overlap_retriever import AppOverlapTrajectoryRetriever
from src.training.pomdp_trajectory_builder import POMDPTrajectoryBuilder

def main():
    parser = argparse.ArgumentParser(description="Build App-Overlap RAG Index & POMDP SFT Dataset")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--data_root", type=str, required=True, help="Thư mục chứa tập train/")
    parser.add_argument("--sft_output", type=str, default="outputs/sft_pomdp_trajectories.jsonl")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    lock_deterministic_environment(config.get("experiment", {}).get("seed", 42))

    train_dir = Path(args.data_root) / "train"
    train_jsonl = str(train_dir / "train.jsonl")
    catalog_json = str(train_dir / "train_tools_catalog.json")

    rag_cfg = config["stage1_perception_and_grounding"]["few_shot_rag"]
    retriever = AppOverlapTrajectoryRetriever(
        top_k=rag_cfg.get("top_k", 3),
        app_overlap_boost=rag_cfg.get("app_overlap_boost", 1.35)
    )
    n_indexed = retriever.build_and_save_index(train_jsonl, rag_cfg["index_path"])
    print(f"[Stage 1 RAG] Đã lập chỉ mục {n_indexed} quỹ đạo mẫu vào {rag_cfg['index_path']}")

    n_steps = POMDPTrajectoryBuilder.build_sft_dataset(train_jsonl, catalog_json, args.sft_output)
    print(f"[Stage 2 SFT] Đã tạo {n_steps} mẫu huấn luyện trạng thái-hành động POMDP tại {args.sft_output}")

if __name__ == "__main__":
    main()
