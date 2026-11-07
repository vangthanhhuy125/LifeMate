"""Evaluation and Ablation Study Runner using the Official Competition Metric."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from src.evaluation.official_scorer import OfficialScorer
from src.pipeline.lifemate_pomdp_pipeline import LifeMatePOMDPPipeline
from src.utils.reproducibility import load_yaml_config


def _get_tool_env_cls(data_root: Path, split_name: str) -> Any:
    split_path = str((data_root / split_name).resolve())
    if split_path not in sys.path:
        sys.path.insert(0, split_path)
    try:
        from zaic_tools import ToolEnv  # type: ignore

        return ToolEnv
    except ImportError:
        from tests.fakes.fake_env import FakeToolEnv

        return FakeToolEnv


def compute_gold_hash_via_replay(
    tool_env_cls: Any,
    split_name: str,
    sample_id: str,
    data_root: Path,
    gold_calls: List[Dict[str, Any]],
) -> str:
    """Replay gold tool calls in a clean environment instance to obtain the reference `gold_state_hash`."""
    gold_env = tool_env_cls(split_name, sample_id, root=str(data_root))
    for tc in gold_calls:
        if isinstance(tc, dict) and tc.get("name"):
            gold_env.call(tc["name"], tc.get("arguments") or {})
    return str(gold_env.get_state_hash())


def evaluate_config_on_samples(
    config_path: str | Path,
    samples: List[Dict[str, Any]],
    catalog_data: Dict[str, Any],
    data_root: Path,
    split_name: str = "train",
) -> Dict[str, Any]:
    """Run a single configuration across evaluation samples and compute official metrics."""
    cfg = load_yaml_config(config_path)
    pipeline = LifeMatePOMDPPipeline(cfg, catalog_data=catalog_data)
    scorer = OfficialScorer.from_catalog(catalog_data)
    tool_env_cls = _get_tool_env_cls(data_root, split_name)

    predictions: Dict[str, Dict[str, Any]] = {}
    ground_truths: Dict[str, Dict[str, Any]] = {}

    for item in samples:
        sid = str(item["id"])
        label = item.get("label") or {}
        gold_calls = label.get("tool_calls") or []

        gold_hash = label.get("state_hash")
        if not gold_hash:
            gold_hash = compute_gold_hash_via_replay(
                tool_env_cls, split_name, sid, data_root, gold_calls
            )

        ground_truths[sid] = {
            "tool_calls": gold_calls,
            "state_hash": gold_hash,
        }

        # Run agent on its single dedicated ToolEnv
        env = tool_env_cls(split_name, sid, root=str(data_root))
        sub_rec = pipeline.run_sample(item, env, data_root=data_root, split_name=split_name)
        predictions[sid] = {
            "tool_calls": sub_rec.tool_calls,
            "state_hash": sub_rec.state_hash,
        }

    metrics = scorer.score_dataset(predictions, ground_truths)
    return {
        "config_path": str(config_path),
        "experiment_name": cfg.get("experiment_name", "default"),
        "final_score": metrics["final_score"],
        "tie_breaker_f1": metrics["tie_breaker_f1"],
        "state_match_rate": metrics["state_match_rate"],
        "N": metrics["N"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LifeMate and run ablation experiments.")
    parser.add_argument("--data-root", type=str, default=".", help="Root directory containing train split.")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Main config.")
    parser.add_argument("--run-ablations", action="store_true", help="Run all ablation configs.")
    parser.add_argument("--max-samples", type=int, default=100, help="Max validation samples to score.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_jsonl = data_root / "train" / "train.jsonl"
    catalog_path = data_root / "train" / "train_tools_catalog.json"

    if not train_jsonl.exists() or not catalog_path.exists():
        print(f"[INFO] Train split files not found under {data_root / 'train'}.")
        return

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog_data = json.load(f)

    all_samples: List[Dict[str, Any]] = []
    with open(train_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_samples.append(json.loads(line))

    eval_samples = all_samples[-args.max_samples :] if len(all_samples) > args.max_samples else all_samples

    configs_to_run = [Path(args.config)]
    if args.run_ablations:
        ablation_dir = Path("configs/ablations")
        if ablation_dir.exists():
            configs_to_run.extend(sorted(ablation_dir.glob("*.yaml")))

    results = []
    for cfg_path in configs_to_run:
        res = evaluate_config_on_samples(
            config_path=cfg_path,
            samples=eval_samples,
            catalog_data=catalog_data,
            data_root=data_root,
            split_name="train",
        )
        results.append(res)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()