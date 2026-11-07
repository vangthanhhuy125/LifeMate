def enforce_token_budget(prompt: str, max_tokens=16384) -> str:
    # A simple mock budget enforcer
    if len(prompt) > max_tokens * 3:
        return prompt[:max_tokens * 3] + "..."
    return prompt