"""Official Inference Entrypoint adhering strictly to the Zalo AI Challenge 2026 template."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Tuple

from src.pipeline.lifemate_pomdp_pipeline import LifeMatePOMDPPipeline
from src.utils.reproducibility import load_yaml_config


def load_tool_env_class(data_root: Path, split_name: str) -> Any:
    """Import official `ToolEnv` from `<data_root>/<split>/zaic_tools` or fallback to test fake in dev."""
    split_path = str((data_root / split_name).resolve())
    if split_path not in sys.path:
        sys.path.insert(0, split_path)
    try:
        from zaic_tools import ToolEnv  # type: ignore

        return ToolEnv
    except ImportError:
        from tests.fakes.fake_env import FakeToolEnv

        return FakeToolEnv


def infer_single_sample(
    item: dict,
    pipeline: LifeMatePOMDPPipeline,
    tool_env_cls: Any,
    data_root: Path,
    split_name: str,
) -> Tuple[list, str]:
    """Create exactly ONE `ToolEnv` instance for the sample, run pipeline, and return calls + hash."""
    env = tool_env_cls(split_name, item["id"], root=str(data_root))
    record = pipeline.run_sample(
        sample=item,
        env=env,
        data_root=data_root,
        split_name=split_name,
    )
    return record.tool_calls, record.state_hash


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic LifeMate inference on a split.")
    parser.add_argument("--data-root", type=str, default=".", help="Folder holding split directories.")
    parser.add_argument("--split", type=str, default="public_test", help="Split name (e.g. public_test).")
    parser.add_argument(
        "--config", type=str, default="configs/default_config.yaml", help="Path to YAML config."
    )
    parser.add_argument(
        "--output", type=str, default="submission.jsonl", help="Output submission JSONL path."
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    split_name = args.split
    config = load_yaml_config(args.config)

    pipeline = LifeMatePOMDPPipeline(config)
    pipeline.load_split_catalog(data_root, split_name)
    tool_env_cls = load_tool_env_class(data_root, split_name)

    input_jsonl = data_root / split_name / f"{split_name}.jsonl"
    output_jsonl = Path(args.output)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    if not input_jsonl.exists():
        raise FileNotFoundError(f"Split input file not found: {input_jsonl}")

    with open(input_jsonl, "r", encoding="utf-8") as fin, open(
        output_jsonl, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            tool_calls, state_hash = infer_single_sample(
                item=item,
                pipeline=pipeline,
                tool_env_cls=tool_env_cls,
                data_root=data_root,
                split_name=split_name,
            )
            fout.write(
                json.dumps(
                    {
                        "id": item["id"],
                        "tool_calls": tool_calls,
                        "state_hash": state_hash,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()