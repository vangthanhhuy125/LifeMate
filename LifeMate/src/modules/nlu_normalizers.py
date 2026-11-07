import re
class TimeNormalizer:
    @staticmethod
    def normalize(text: str, current_time: str) -> str:
        # Mock ISO-8601 normalization
        if "mai" in text: return current_time + "+1day"
        return text

class EntityResolver:
    @staticmethod
    def resolve(mention: str, observations: list) -> str:
        for obs in observations:
            if isinstance(obs, dict) and "devices" in obs:
                for d in obs["devices"]:
                    if mention.lower() in d.get("name", "").lower():
                        return d["id"]
        return mention