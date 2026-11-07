"""Dataset and Tool Catalog Inspection Script for LifeMate."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def inspect_split(data_root: Path, split_name: str) -> Dict[str, Any]:
    """Inspect a dataset split (`train`, `public_test`, or `private_test`)."""
    split_dir = data_root / split_name
    jsonl_path = split_dir / f"{split_name}.jsonl"
    catalog_path = split_dir / f"{split_name}_tools_catalog.json"

    summary: Dict[str, Any] = {
        "split": split_name,
        "exists": split_dir.exists(),
        "num_samples": 0,
        "num_catalog_apps": 0,
        "num_catalog_tools": 0,
        "num_oos_samples": 0,
        "num_ask_missing_samples": 0,
    }

    if catalog_path.exists():
        with open(catalog_path, "r", encoding="utf-8") as f:
            cat = json.load(f)
        apps = cat.get("apps", [])
        summary["num_catalog_apps"] = len(apps)
        summary["num_catalog_tools"] = sum(len(a.get("tools", [])) for a in apps if isinstance(a, dict))

    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                summary["num_samples"] += 1
                item = json.loads(line)
                label = item.get("label")
                if isinstance(label, dict):
                    t_calls = label.get("tool_calls") or []
                    if len(t_calls) == 0:
                        summary["num_oos_samples"] += 1
                    elif any(tc.get("name") == "sys.ask_missing_info" for tc in t_calls if isinstance(tc, dict)):
                        summary["num_ask_missing_samples"] += 1

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Zalo AI Challenge 2026 LifeMate data splits.")
    parser.add_argument("--data-root", type=str, default=".", help="Root directory containing split folders.")
    args = parser.parse_args()

    root = Path(args.data_root)
    report = {}
    for split in ("train", "public_test", "private_test"):
        report[split] = inspect_split(root, split)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()