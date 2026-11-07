class ValidatorGuard:
    @staticmethod
    def check_xref(tool_name, args, history):
        # Contribution 1: X-Ref Integrity
        # Returns True if valid, False and error message if invalid
        for k, v in args.items():
            if "id" in k.lower():
                # Mock check
                found = False
                for h in history:
                    obs = h.get("observation", {})
                    if str(v) in str(obs):
                        found = True
                if not found:
                    return False, f"x-ref ID {v} not found in prior read observations."
        return True, ""