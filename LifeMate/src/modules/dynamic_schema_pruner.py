from typing import Dict, List, Any, Tuple

class DynamicSchemaPruner:
    """
    [Contribution 2 - Dynamic Schema Pruning S(K)]:
    Cắt tỉa danh mục công cụ tổng thể dựa trên tập ứng dụng khả dụng K (context.installed_apps).
    Giảm >70% lượng token đầu vào, đồng thời bóc tách rõ các ràng buộc x-ref, x-one-of, x-match: fuzzy.
    """
    @classmethod
    def prune_and_analyze(
        cls,
        full_catalog: Dict[str, Any],
        installed_apps: Dict[str, Any],
        enable_pruning: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, str]]]:
        """
        Trả về:
          - pruned_schema_SK: Danh sách JSON Schema của các tool thuộc S(K).
          - xref_constraints: Bản đồ {tool_name: {param_name: xref_source}} để kiểm tra Read-before-Write.
        """
        installed_ids = set(installed_apps.keys())
        pruned_tools: List[Dict[str, Any]] = []
        xref_constraints: Dict[str, Dict[str, str]] = {}

        for app in full_catalog.get("apps", []):
            app_id = app.get("id", "")
            keep_app = (not enable_pruning) or (app_id in installed_ids) or (app_id == "sys")

            if keep_app:
                allowed_subset = installed_apps.get(app_id, {}).get("tools", None) if app_id in installed_ids else None
                for tool in app.get("tools", []):
                    t_name = tool.get("name", "")
                    if allowed_subset is not None and t_name not in allowed_subset:
                        continue
                    compact_t, tool_xrefs = cls._compact_single_tool(tool)
                    pruned_tools.append(compact_t)
                    if tool_xrefs:
                        xref_constraints[t_name] = tool_xrefs

        # Luôn bảo đảm meta-tool sys.ask_missing_info có mặt trong không gian hành động
        if not any(t.get("name") == "sys.ask_missing_info" for t in pruned_tools):
            pruned_tools.append({
                "name": "sys.ask_missing_info",
                "kind": "meta",
                "description": "Yêu cầu người dùng cung cấp thêm thông tin khi câu lệnh thiếu tham số bắt buộc.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_name": {"type": "string", "description": "Tên công cụ dự định gọi"},
                        "params": {"type": "array", "items": {"type": "string"}, "description": "Danh sách tham số bị thiếu"}
                    },
                    "required": ["tool_name", "params"]
                }
            })

        return pruned_tools, xref_constraints

    @staticmethod
    def _compact_single_tool(tool: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        params = tool.get("parameters", {})
        props = params.get("properties", {})
        tool_xrefs = {}
        for p_name, p_spec in props.items():
            if isinstance(p_spec, dict) and "x-ref" in p_spec:
                tool_xrefs[p_name] = p_spec["x-ref"]

        compact = {
            "name": tool.get("name"),
            "kind": tool.get("kind", "write"),
            "description": tool.get("description", ""),
            "parameters": params,
            "returns": tool.get("returns", "")
        }
        return compact, tool_xrefs

    @staticmethod
    def extract_fuzzy_argument_map(full_catalog: Dict[str, Any]) -> Dict[str, List[str]]:
        """Trích xuất danh sách các tham số có x-match: 'fuzzy' cho từng công cụ để chấm Token-F1."""
        fuzzy_map: Dict[str, List[str]] = {}
        for app in full_catalog.get("apps", []):
            for tool in app.get("tools", []):
                t_name = tool.get("name", "")
                props = tool.get("parameters", {}).get("properties", {})
                f_keys = [k for k, v in props.items() if isinstance(v, dict) and v.get("x-match") == "fuzzy"]
                if f_keys:
                    fuzzy_map[t_name] = f_keys
        return fuzzy_map
