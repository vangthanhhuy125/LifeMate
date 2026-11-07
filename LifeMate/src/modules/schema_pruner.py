"""Base Tool Catalog Parser and Static Schema Pruner for LifeMate."""

import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.pomdp_types import ToolSpec


class SchemaPruner:
    """Parses `<split>_tools_catalog.json` into structured `ToolSpec` objects and filters schemas."""

    META_ASK_MISSING_SPEC = ToolSpec(
        name="sys.ask_missing_info",
        app="sys",
        kind="meta",
        description="Yêu cầu người dùng cung cấp thêm các tham số bắt buộc còn thiếu để gọi công cụ.",
        parameters={
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Tên công cụ dự định gọi nhưng bị thiếu thông tin",
                },
                "params": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Danh sách tên các tham số bắt buộc đang bị thiếu",
                },
            },
            "required": ["tool_name", "params"],
        },
        returns="",
        required_params=["tool_name", "params"],
    )

    def __init__(self, catalog_data: Optional[Dict[str, Any]] = None) -> None:
        self.raw_catalog: Dict[str, Any] = {}
        self.tools_by_name: Dict[str, ToolSpec] = {}
        self.tools_by_app: Dict[str, List[ToolSpec]] = {}
        if catalog_data:
            self.load_catalog_dict(catalog_data)

    @staticmethod
    def _normalize(text: str) -> str:
        return unicodedata.normalize("NFC", str(text).strip())

    def load_catalog_file(self, catalog_path: str | Path) -> None:
        """Load tool catalog from a JSON file path."""
        path = Path(catalog_path)
        if not path.exists():
            raise FileNotFoundError(f"Tool catalog file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.load_catalog_dict(data)

    def load_catalog_dict(self, catalog_data: Dict[str, Any]) -> None:
        """Parse raw catalog dictionary into indexed `ToolSpec` collections."""
        self.raw_catalog = catalog_data or {}
        self.tools_by_name = {}
        self.tools_by_app = {}

        apps_list = self.raw_catalog.get("apps", [])
        if isinstance(apps_list, list):
            for app_entry in apps_list:
                if not isinstance(app_entry, dict):
                    continue
                app_id = str(app_entry.get("id") or app_entry.get("app") or "").strip()
                for raw_tool in app_entry.get("tools", []):
                    if not isinstance(raw_tool, dict):
                        continue
                    spec = self.parse_tool_dict(raw_tool, default_app=app_id)
                    self.tools_by_name[spec.name] = spec
                    self.tools_by_app.setdefault(spec.app, []).append(spec)

        # Ensure meta tool sys.ask_missing_info is always registered
        if "sys.ask_missing_info" not in self.tools_by_name:
            self.tools_by_name["sys.ask_missing_info"] = self.META_ASK_MISSING_SPEC
            self.tools_by_app.setdefault("sys", []).append(self.META_ASK_MISSING_SPEC)

    @classmethod
    def parse_tool_dict(cls, raw_tool: Dict[str, Any], default_app: str = "") -> ToolSpec:
        """Convert a raw tool JSON schema definition into a structured `ToolSpec`."""
        name = cls._normalize(raw_tool.get("name", ""))
        app = cls._normalize(raw_tool.get("app") or default_app or (name.split(".")[0] if "." in name else "unknown"))
        kind = cls._normalize(raw_tool.get("kind", "")).lower()
        if not kind:
            if name == "sys.ask_missing_info":
                kind = "meta"
            elif any(k in name for k in ("get_", "list", "search", "status", "find", "check", "read")):
                kind = "read"
            else:
                kind = "write"

        description = cls._normalize(raw_tool.get("description", ""))
        returns = cls._normalize(raw_tool.get("returns", ""))
        params_schema = dict(raw_tool.get("parameters") or {"type": "object", "properties": {}})
        props = params_schema.get("properties") or {}
        required_params = [str(p) for p in (params_schema.get("required") or [])]

        xref_params: Dict[str, str] = {}
        fuzzy_params: List[str] = []
        for prop_name, prop_def in props.items():
            if not isinstance(prop_def, dict):
                continue
            if "x-ref" in prop_def and prop_def["x-ref"]:
                xref_params[str(prop_name)] = str(prop_def["x-ref"])
            if prop_def.get("x-match") == "fuzzy":
                fuzzy_params.append(str(prop_name))

        one_of_groups: List[List[str]] = []
        raw_one_of = params_schema.get("x-one-of") or raw_tool.get("x-one-of") or []
        if isinstance(raw_one_of, list):
            for group in raw_one_of:
                if isinstance(group, list):
                    one_of_groups.append([str(x) for x in group])
                elif isinstance(group, str):
                    one_of_groups.append([group])
                elif isinstance(group, dict) and "required" in group:
                    one_of_groups.append([str(x) for x in group["required"]])

        return ToolSpec(
            name=name,
            app=app,
            kind=kind,
            description=description,
            parameters=params_schema,
            returns=returns,
            required_params=required_params,
            xref_params=xref_params,
            fuzzy_params=fuzzy_params,
            one_of_groups=one_of_groups,
        )

    @staticmethod
    def to_compact_json(spec: ToolSpec, compact_descriptions: bool = True) -> Dict[str, Any]:
        """Serialize a `ToolSpec` into a compact JSON dictionary for LLM prompting."""
        desc = spec.description
        if compact_descriptions and len(desc) > 120:
            desc = desc[:117] + "..."
        out: Dict[str, Any] = {
            "name": spec.name,
            "kind": spec.kind,
            "description": desc,
            "parameters": spec.parameters,
        }
        if spec.xref_params:
            out["x_ref_fields"] = spec.xref_params
        if spec.one_of_groups:
            out["x_one_of"] = spec.one_of_groups
        return out

    def get_all_tools(self) -> List[ToolSpec]:
        """Return all registered tools in the catalog."""
        return list(self.tools_by_name.values())