"""Configuration loader with inheritance (`extends`) and offline reproducibility checks."""

import copy
import os
from pathlib import Path
from typing import Any, Dict

import yaml

from src.utils.seed import set_deterministic_seeds


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dictionary into base dictionary."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML configuration, resolving any `extends` base configuration path."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    parent_path_str = raw_cfg.get("extends")
    if parent_path_str:
        parent_path = Path(parent_path_str)
        if not parent_path.is_absolute() and not parent_path.exists():
            candidate = path.parent.parent / parent_path.name
            if candidate.exists():
                parent_path = candidate
        base_cfg = load_yaml_config(parent_path)
        return _deep_merge(base_cfg, raw_cfg)

    return raw_cfg


def enforce_reproducibility(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply seed locking, offline flags, and parameter budget checks from config."""
    proj_cfg = config.get("project", {})
    seed = int(proj_cfg.get("seed", 42))
    deterministic = bool(proj_cfg.get("deterministic", True))
    set_deterministic_seeds(seed=seed, deterministic_torch=deterministic)

    hw_cfg = config.get("hardware_constraints", {})
    if hw_cfg.get("offline_mode", True):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    asr_params = float(config.get("asr", {}).get("estimated_params_b", 0.24))
    llm_params = float(config.get("pomdp_planner", {}).get("estimated_params_b", 3.09))
    total_params = asr_params + llm_params

    max_llm = float(hw_cfg.get("max_llm_params_b", 5.0))
    max_total = float(hw_cfg.get("max_total_params_b", 6.0))

    if llm_params >= max_llm:
        raise ValueError(
            f"LLM parameter budget violated: {llm_params:.2f}B >= {max_llm:.2f}B limit."
        )
    if total_params >= max_total:
        raise ValueError(
            f"Total model parameter budget violated: {total_params:.2f}B >= {max_total:.2f}B limit."
        )

    return {
        "seed": seed,
        "deterministic": deterministic,
        "offline_mode": bool(hw_cfg.get("offline_mode", True)),
        "asr_params_b": asr_params,
        "llm_params_b": llm_params,
        "total_params_b": round(total_params, 4),
    }