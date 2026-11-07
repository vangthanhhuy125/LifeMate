from src.modules.validator_guard import ValidatorGuard

class Stage3:
    def __init__(self, config):
        self.max_retries = config.get("max_retries", 2)
        
    def execute(self, env, planner, stage1_out, context):
        history = []
        for step in range(6):
            action = planner.plan(stage1_out, history, context)
            if action.action_type in ["TERMINATE", "OUT_OF_SCOPE"]:
                break
                
            if action.action_type == "CALL":
                # Contribution 1: Validate-Before-Call
                is_valid_xref, err = ValidatorGuard.check_xref(action.tool_name, action.arguments, history)
                if not is_valid_xref:
                    # Self-correction logic here (mocked)
                    action.tool_name = "sys.ask_missing_info"
                    action.missing_params = ["device_id"]
                    break
                    
                val_res = env.validate(action.tool_name, action.arguments)
                if not val_res.get("ok", False):
                    break
                    
                obs = env.call(action.tool_name, action.arguments)
                history.append({"action": action.tool_name, "observation": obs})
                
            elif action.action_type == "ASK_MISSING":
                break
                
        return env.get_tool_calls(), env.get_state_hash()