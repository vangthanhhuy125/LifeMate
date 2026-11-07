import json
class DynamicSchemaPruner:
    @staticmethod
    def prune(catalog, installed_apps):
        pruned = []
        installed = set(installed_apps.keys())
        original_len = len(json.dumps(catalog))
        
        for app in catalog.get("apps", []):
            if app["id"] in installed or app["id"] == "sys":
                allowed = installed_apps.get(app["id"], {}).get("tools", None)
                for t in app.get("tools", []):
                    if not allowed or t["name"] in allowed:
                        pruned.append(t)
        
        # Ensure sys.ask_missing_info
        if not any(t["name"] == "sys.ask_missing_info" for t in pruned):
            pruned.append({"name": "sys.ask_missing_info", "kind": "meta", "parameters": {"properties": {"tool_name": {}, "params": {}}, "required": ["tool_name", "params"]}})
            
        pruned_len = len(json.dumps(pruned))
        reduction = 100 * (1 - pruned_len/max(1, original_len))
        return pruned, reduction