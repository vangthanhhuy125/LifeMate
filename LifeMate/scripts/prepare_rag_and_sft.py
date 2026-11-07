"""Script to build the Few-Shot BM25 RAG Index and Multi-Step POMDP SFT Dataset from `train.jsonl`."""

import argparse
import json
from pathlib import Path

from src.modules.retriever_bm25 import RetrieverBM25
from src.training.pomdp_trajectory_builder import POMDPTrajectoryBuilder
from src.utils.reproducibility import enforce_reproducibility, load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare RAG index and POMDP SFT dataset.")
    parser.add_argument("--data-root", type=str, default=".", help="Path to data root folder.")
    parser.add_argument(
        "--config", type=str, default="configs/default_config.yaml", help="Path to YAML config."
    )
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    enforce_reproducibility(config)

    data_root = Path(args.data_root)
    train_jsonl = data_root / "train" / "train.jsonl"
    train_catalog = data_root / "train" / "train_tools_catalog.json"

    if not train_jsonl.exists():
        print(f"[INFO] {train_jsonl} not found. Skipping offline index build.")
        return

    val_ratio = float(config.get("training", {}).get("val_split_ratio", 0.1))
    builder = POMDPTrajectoryBuilder(config)
    rag_records, sft_records, val_samples = builder.process_train_split(
        train_jsonl_path=train_jsonl,
        catalog_path=train_catalog,
        val_ratio=val_ratio,
    )

    rag_index_path = Path(
        config.get("retrieval", {}).get("index_path", "artifacts/rag_trajectories.json")
    )
    bm25 = RetrieverBM25()
    bm25.fit(rag_records)
    bm25.save_index(rag_index_path)

    sft_out_path = Path(
        config.get("training", {}).get("sft_output_path", "artifacts/sft_pomdp_dataset.jsonl")
    )
    sft_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sft_out_path, "w", encoding="utf-8") as fout:
        for rec in sft_records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "status": "success",
                "rag_index_path": str(rag_index_path),
                "num_rag_trajectories": len(rag_records),
                "sft_output_path": str(sft_out_path),
                "num_sft_steps": len(sft_records),
                "num_heldout_val_samples": len(val_samples),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()