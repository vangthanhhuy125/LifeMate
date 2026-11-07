import unittest
from pathlib import Path
from src.utils.pomdp_types import POMDPAction
from src.modules.dynamic_schema_pruner import DynamicSchemaPruner
from src.evaluation.official_scorer import OfficialLifeMateScorer
from src.stages.stage3_closed_loop_executor import Stage3_ValidateBeforeCallExecutor
from src.stages.stage1_perception_and_grounding import Stage1_PerceptionAndGrounding

class MockZaicToolEnv:
    """Môi trường giả lập mô phỏng chính xác cơ chế của zaic_tools.ToolEnv."""
    def __init__(self):
        self.logged_calls = []
        self.ac_temp = 28

    def validate(self, tool_name: str, arguments: dict) -> dict:
        if tool_name == "smart_home.climate_set":
            if "device_id" not in arguments:
                return {"ok": False, "error": "thiếu tham số bắt buộc: device_id"}
            if not isinstance(arguments.get("temperature", 26), int):
                return {"ok": False, "error": "temperature phải là kiểu integer"}
        return {"ok": True}

    def call(self, tool_name: str, arguments: dict) -> dict:
        # Mọi lệnh gọi qua env.call() đều bị ghi vào logged_calls
        self.logged_calls.append({"name": tool_name, "arguments": arguments})
        val = self.validate(tool_name, arguments)
        if not val["ok"]:
            return val
        if tool_name == "smart_home.get_status":
            return {"ok": True, "devices": [{"id": "ac_living_room", "type": "ac"}]}
        if tool_name == "smart_home.climate_set":
            self.ac_temp = arguments.get("temperature", 26)
            return {"ok": True}
        return {"ok": True}

    def get_tool_calls(self):
        return self.logged_calls

    def get_state_hash(self):
        return f"hash_temp_{self.ac_temp}"

class DummySelfCorrectingPlanner:
    """Giả lập Stage 2 Planner: Cố tình mắc lỗi ở lần thử đầu, sau đó tự sửa khi nhận phản hồi từ Stage 3."""
    def __init__(self):
        self.call_count = 0

    def sample_action(self, bundle, history_H, validation_error_feedback=None):
        self.call_count += 1
        # Bước 1: Chưa gọi read mà đòi gọi write có x-ref -> Bị màng lọc chặn
        if len(history_H) == 0 and validation_error_feedback is None:
            return POMDPAction(
                action_type="CALL",
                tool_name="smart_home.climate_set",
                arguments={"device_id": "guessed_id", "temperature": 26}
            )
        # Sau khi nhận cảnh báo x-ref từ Stage 3 -> Tự sửa sang gọi smart_home.get_status
        if len(history_H) == 0 and validation_error_feedback is not None:
            return POMDPAction(
                action_type="CALL",
                tool_name="smart_home.get_status",
                arguments={}
            )
        # Bước 2: Đã có kết quả đọc, cố tình truyền sai kiểu dữ liệu temperature="26" -> Bị env.validate chặn
        if len(history_H) == 1 and validation_error_feedback is None:
            return POMDPAction(
                action_type="CALL",
                tool_name="smart_home.climate_set",
                arguments={"device_id": "ac_living_room", "temperature": "26_string_error"}
            )
        # Tự sửa lại thành integer 26 sau khi nhận lỗi từ env.validate
        if len(history_H) == 1 and validation_error_feedback is not None:
            return POMDPAction(
                action_type="CALL",
                tool_name="smart_home.climate_set",
                arguments={"device_id": "ac_living_room", "temperature": 26}
            )
        return POMDPAction(action_type="TERMINATE")

class TestPOMDPResearchFramework(unittest.TestCase):
    def setUp(self):
        self.sample_catalog = {
            "apps": [
                {
                    "id": "smart_home",
                    "tools": [
                        {"name": "smart_home.get_status", "kind": "read", "parameters": {"properties": {}}},
                        {
                            "name": "smart_home.climate_set",
                            "kind": "write",
                            "parameters": {
                                "properties": {
                                    "device_id": {"type": "string", "x-ref": "smart_home.devices"},
                                    "temperature": {"type": "integer"}
                                },
                                "required": ["device_id"]
                            }
                        }
                    ]
                },
                {
                    "id": "banking",
                    "tools": [
                        {"name": "banking.transfer", "kind": "write", "parameters": {"properties": {"note": {"x-match": "fuzzy"}}}}
                    ]
                }
            ]
        }

    def test_contribution1_validate_before_call_protects_precision(self):
        """Chứng minh Contribution 1 chặn toàn bộ lệnh lỗi, giữ U = 0 và P = 1.0."""
        stage1 = Stage1_PerceptionAndGrounding({
            "asr": {"use_oracle_transcript": True, "code_switching_normalization": True, "lexicon_map": {"ây xi": "AC"}},
            "schema_pruning": {"enabled": True},
            "few_shot_rag": {"enabled": False}
        })
        raw_sample = {
            "id": "s_01",
            "audio_text": ["Bật ây xi phòng khách 26 độ"],
            "context": {"installed_apps": {"smart_home": {}}}
        }
        bundle = stage1.process(raw_sample, Path("."), self.sample_catalog)
        self.assertIn("AC", bundle.utterance_A)

        env = MockZaicToolEnv()
        planner = DummySelfCorrectingPlanner()
        executor = Stage3_ValidateBeforeCallExecutor({
            "max_pomdp_steps": 5,
            "validate_before_call": True,
            "max_self_correction_retries": 2
        })

        calls, state_hash, _ = executor.run_pomdp_episode(env, bundle, planner, episode_start_time=0.0, max_time_sec=9999999999.0)
        # Chỉ có đúng 2 lệnh hợp lệ được ghi vào env.get_tool_calls(), 2 lệnh lỗi đã bị chặn và tự sửa!
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["name"], "smart_home.get_status")
        self.assertEqual(calls[1]["name"], "smart_home.climate_set")
        self.assertEqual(state_hash, "hash_temp_26")

        gold_calls = [
            {"name": "smart_home.get_status", "arguments": {}},
            {"name": "smart_home.climate_set", "arguments": {"device_id": "ac_living_room", "temperature": 26}}
        ]
        score_res = OfficialLifeMateScorer.score_instance(calls, gold_calls, state_hash, "hash_temp_26")
        self.assertEqual(score_res["U"], 0)
        self.assertAlmostEqual(score_res["score"], 1.0)

    def test_contribution2_dynamic_schema_pruning(self):
        """Chứng minh Contribution 2 loại bỏ hoàn toàn ứng dụng không cài đặt (banking) và trích xuất đúng x-ref."""
        pruned, xrefs = DynamicSchemaPruner.prune_and_analyze(
            self.sample_catalog, {"smart_home": {}}, enable_pruning=True
        )
        names = [t["name"] for t in pruned]
        self.assertIn("smart_home.climate_set", names)
        self.assertIn("sys.ask_missing_info", names)
        self.assertNotIn("banking.transfer", names)
        self.assertIn("smart_home.climate_set", xrefs)
        self.assertEqual(xrefs["smart_home.climate_set"]["device_id"], "smart_home.devices")

if __name__ == "__main__":
    unittest.main()
