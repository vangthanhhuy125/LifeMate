class FakeToolEnv:
    def __init__(self, split, id, root):
        self.calls = []
        self._hash = "dummy_hash"
    def validate(self, name, args):
        if "fail" in str(args): return {"ok": False, "error": "schema error"}
        return {"ok": True}
    def call(self, name, args):
        self.calls.append({"name": name, "arguments": args})
        if name == "smart_home.get_status": return {"ok": True, "devices": [{"id": "ac_123", "name": "điều hòa"}]}
        return {"ok": True}
    def get_tool_calls(self): return self.calls
    def get_state_hash(self): return self._hash