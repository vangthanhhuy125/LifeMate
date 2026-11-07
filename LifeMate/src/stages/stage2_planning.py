import json
from src.utils.pomdp_types import Action

class Stage2:
    def __init__(self, config):
        self.config = config
        
    def plan(self, stage1_out, history, context):
        # Mock LLM Planner with Constrained Decoding
        # Trả về action dựa trên history
        if not history:
            return Action("CALL", "smart_home.get_status", {})
        elif len(history) == 1:
            return Action("CALL", "smart_home.light_set", {"device_id": "mock_id", "power": "on"})
        return Action("TERMINATE")