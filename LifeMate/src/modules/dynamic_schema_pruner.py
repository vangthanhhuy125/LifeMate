"""Dynamic Schema Pruning Module tailored for <5B LLMs and 16K Context Window."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from src.modules.schema_pruner import SchemaPruner
from src.utils.pomdp_types import ToolSpec


class DynamicSchemaPruner(SchemaPruner):
    """Prunes the full split catalog down to `installed_apps` and detects Out-of-Scope requests."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        catalog_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(catalog_data=catalog_data)
        self.config = config or {}
        prune_cfg = self.config.get("schema_pruning", self.config)
        self.enable_dynamic_pruning = bool(prune_cfg.get("enable_dynamic_pruning", True))
        self.retain_meta_tools = bool(prune_cfg.get("retain_meta_tools", True))
        self.retain_xref_provider_read_tools = bool(
            prune_cfg.get("retain_xref_provider_read_tools", True)
        )
        self.compact_descriptions = bool(prune_cfg.get("compact_descriptions", True))

        self.app_keywords: Dict[str, List[str]] = {}
        self._load_app_keywords()

    def _load_app_keywords(self) -> None:
        """Load domain keyword mappings from lexicon config."""
        asr_cfg = self.config.get("asr", {})
        lex_path = Path(asr_cfg.get("lexicon_path", "configs/lexicon/code_switching_vi_en.yaml"))
        if lex_path.exists():
            with open(lex_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            raw_kw = data.get("app_keywords") or {}
            for app_name, kw_list in raw_kw.items():
                if isinstance(kw_list, list):
                    self.app_keywords[str(app_name)] = [str(k).lower() for k in kw_list]

    def _extract_context_installed_tools(
        self, sample_context: Dict[str, Any]
    ) -> Dict[str, List[ToolSpec]]:
        """Extract tool schemas embedded directly inside `context.installed_apps` if present."""
        installed = sample_context.get("installed_apps") or {}
        extracted: Dict[str, List[ToolSpec]] = {}

        if isinstance(installed, dict):
            for app_name, app_val in installed.items():
                app_id = str(app_name)
                extracted.setdefault(app_id, [])
                if isinstance(app_val, dict) and isinstance(app_val.get("tools"), list):
                    for t_item in app_val["tools"]:
                        if isinstance(t_item, dict):
                            spec = self.parse_tool_dict(t_item, default_app=app_id)
                            self.tools_by_name[spec.name] = spec
                            extracted[app_id].append(spec)
                        elif isinstance(t_item, str) and t_item in self.tools_by_name:
                            extracted[app_id].append(self.tools_by_name[t_item])
                if not extracted[app_id] and app_id in self.tools_by_app:
                    extracted[app_id] = list(self.tools_by_app[app_id])
        elif isinstance(installed, list):
            for item in installed:
                app_id = str(item)
                extracted[app_id] = list(self.tools_by_app.get(app_id, []))

        return extracted

    def detect_intent_apps(
        self, normalized_utterance: str, extracted_clues: Dict[str, Any]
    ) -> List[str]:
        """Identify which app domains are requested by the user's utterance."""
        lower = normalized_utterance.lower()
        matched_apps: List[str] = []

        for app_id, kw_list in self.app_keywords.items():
            for kw in kw_list:
                if kw in lower:
                    if app_id not in matched_apps:
                        matched_apps.append(app_id)
                    break

        # Also check tool descriptions and names from catalog
        for app_id, specs in self.tools_by_app.items():
            if app_id == "sys" or app_id in matched_apps:
                continue
            for spec in specs:
                tool_short = spec.name.split(".")[-1].replace("_", " ").lower()
                if tool_short and len(tool_short) > 3 and tool_short in lower:
                    matched_apps.append(app_id)
                    break

        if "temperature" in extracted_clues or "device_types" in extracted_clues or "room" in extracted_clues:
            if "smart_home" not in matched_apps:
                matched_apps.append("smart_home")
        if "destination_address" in extracted_clues and any(w in lower for w in ("đặt xe", "gọi xe", "đón tôi")):
            if "ride" not in matched_apps:
                matched_apps.append("ride")

        return matched_apps

    def prune_for_sample(
        self,
        sample_context: Dict[str, Any],
        normalized_utterance: str,
        extracted_clues: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Filter tool schemas for a sample and compute out-of-scope likelihood."""
        clues = extracted_clues or {}
        installed_map = self._extract_context_installed_tools(sample_context)
        installed_app_names = list(installed_map.keys())
        installed_set: Set[str] = set(installed_app_names)

        intent_apps = self.detect_intent_apps(normalized_utterance, clues)
        candidate_apps = [app for app in intent_apps if app in installed_set]

        # Out-of-scope detection:
        # 1) No apps are installed at all, OR
        # 2) Utterance clearly targets specific domain apps (e.g. ride/food/banking),
        #    and NONE of the requested domain apps are installed on the device.
        is_out_of_scope_likely = False
        if not installed_app_names:
            is_out_of_scope_likely = True
        elif intent_apps and not candidate_apps:
            # Check if contacts is supporting banking/email; if not, it's out of scope
            is_out_of_scope_likely = True

        total_catalog_count = max(1, len(self.tools_by_name))

        if not self.enable_dynamic_pruning:
            all_tools = self.get_all_tools()
            return {
                "tools": all_tools,
                "candidate_apps": candidate_apps or installed_app_names,
                "is_out_of_scope_likely": is_out_of_scope_likely,
                "schemas_json": [
                    self.to_compact_json(t, self.compact_descriptions) for t in all_tools
                ],
                "stats": {
                    "total_catalog_tools": total_catalog_count,
                    "pruned_tools_count": len(all_tools),
                    "reduction_ratio": 0.0,
                },
            }

        selected_tools: List[ToolSpec] = []
        seen_names: Set[str] = set()

        # Keep all tools belonging to `context.installed_apps`
        for app_id in installed_app_names:
            for spec in installed_map.get(app_id, []):
                if spec.name not in seen_names:
                    selected_tools.append(spec)
                    seen_names.add(spec.name)

        # Ensure any read tools providing x-ref entities within installed apps are preserved
        if self.retain_xref_provider_read_tools:
            referenced_apps: Set[str] = set()
            for spec in list(selected_tools):
                for xref_target in spec.xref_params.values():
                    ref_app = xref_target.split(".")[0]
                    if ref_app in installed_set:
                        referenced_apps.add(ref_app)
            for ref_app in referenced_apps:
                for spec in installed_map.get(ref_app, []):
                    if spec.is_read and spec.name not in seen_names:
                        selected_tools.append(spec)
                        seen_names.add(spec.name)

        # Always retain `sys.ask_missing_info` if meta tools are enabled
        if self.retain_meta_tools:
            meta_spec = self.tools_by_name.get("sys.ask_missing_info", self.META_ASK_MISSING_SPEC)
            if meta_spec.name not in seen_names:
                selected_tools.append(meta_spec)
                seen_names.add(meta_spec.name)

        reduction = 1.0 - (len(selected_tools) / float(total_catalog_count))
        return {
            "tools": selected_tools,
            "candidate_apps": candidate_apps if candidate_apps else installed_app_names,
            "is_out_of_scope_likely": is_out_of_scope_likely,
            "schemas_json": [
                self.to_compact_json(t, self.compact_descriptions) for t in selected_tools
            ],
            "stats": {
                "total_catalog_tools": total_catalog_count,
                "pruned_tools_count": len(selected_tools),
                "reduction_ratio": round(max(0.0, reduction), 4),
            },
        }