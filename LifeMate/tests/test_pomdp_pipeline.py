"""End-to-End Integration Tests for the 3-Stage LifeMate POMDP Pipeline."""

from src.evaluation.official_scorer import OfficialScorer
from src.pipeline.lifemate_pomdp_pipeline import LifeMatePOMDPPipeline
from src.utils.reproducibility import load_yaml_config
from tests.fakes.fake_env import FakeToolEnv, build_sample_catalog


def test_pipeline_multi_step_smart_home_and_ride_booking() -> None:
    """Verify the flagship competition example: Read status -> Set AC 26C -> Book ride home."""
    cfg = load_yaml_config("configs/default_config.yaml")
    catalog = build_sample_catalog()
    pipeline = LifeMatePOMDPPipeline(cfg, catalog_data=catalog)

    sample = {
        "id": "s_000123",
        "audio": "audio/s_000123/s_000123.wav",
        "audio_text": "Tôi sắp về nhà, bật điều hòa phòng khách 26 độ và đặt xe đón tôi.",
        "context": {
            "current_time": "2026-11-16T17:00:00+07:00",
            "timezone": "Asia/Ho_Chi_Minh",
            "location": "227 Nguyễn Văn Cừ, Quận 5, TP.HCM",
            "user_profile": {"home_address": "123 Trần Hưng Đạo, Quận 5, TP.HCM"},
            "installed_apps": {"smart_home": {}, "ride": {}},
        },
        "history": [],
    }

    env = FakeToolEnv("public_test", sample["id"], catalog=catalog)
    submission = pipeline.run_sample(sample, env)

    expected_gold_calls = [
        {"name": "smart_home.get_status", "arguments": {}},
        {
            "name": "smart_home.climate_set",
            "arguments": {
                "device_id": "ac_living_room",
                "mode": "cool",
                "temperature": 26,
            },
        },
        {
            "name": "ride.book",
            "arguments": {
                "from": "227 Nguyễn Văn Cừ, Quận 5, TP.HCM",
                "to": "123 Trần Hưng Đạo, Quận 5, TP.HCM",
            },
        },
    ]

    gold_env = FakeToolEnv("public_test", sample["id"], catalog=catalog)
    for tc in expected_gold_calls:
        gold_env.call(tc["name"], tc["arguments"])

    scorer = OfficialScorer.from_catalog(catalog)
    score_res = scorer.score_sample(
        submitted_calls=submission.tool_calls,
        submitted_hash=submission.state_hash,
        gold_calls=expected_gold_calls,
        gold_hash=gold_env.get_state_hash(),
    )
    assert score_res["state_match"] is True
    assert score_res["score"] == 1.0


def test_pipeline_ask_missing_info_when_parameters_absent() -> None:
    """Verify agent calls `sys.ask_missing_info` when required parameters are not provided."""
    cfg = load_yaml_config("configs/default_config.yaml")
    catalog = build_sample_catalog()
    pipeline = LifeMatePOMDPPipeline(cfg, catalog_data=catalog)

    sample = {
        "id": "public_0001",
        "audio": "audio/public_0001/public_0001.wav",
        "audio_text": "Thêm bài hát này vào playlist giúp tôi",
        "context": {
            "current_time": "2026-11-16T17:00:00+07:00",
            "timezone": "Asia/Ho_Chi_Minh",
            "user_profile": {},
            "installed_apps": {"music": {}},
        },
        "history": [],
    }

    env = FakeToolEnv("public_test", sample["id"], catalog=catalog)
    submission = pipeline.run_sample(sample, env)

    assert len(submission.tool_calls) == 1
    assert submission.tool_calls[0]["name"] == "sys.ask_missing_info"
    assert submission.tool_calls[0]["arguments"]["tool_name"] == "music.add_to_playlist"
    assert sorted(submission.tool_calls[0]["arguments"]["params"]) == ["playlist_id", "track_id"]


def test_pipeline_out_of_scope_returns_empty_list() -> None:
    """Verify agent returns `[]` when no installed app can handle the request."""
    cfg = load_yaml_config("configs/default_config.yaml")
    catalog = build_sample_catalog()
    pipeline = LifeMatePOMDPPipeline(cfg, catalog_data=catalog)

    sample = {
        "id": "public_0002",
        "audio": "audio/public_0002/public_0002.wav",
        "audio_text": "Đặt cho tôi 2 suất cơm tấm sườn bì chả giao tới công ty",
        "context": {
            "current_time": "2026-11-16T12:00:00+07:00",
            "timezone": "Asia/Ho_Chi_Minh",
            "user_profile": {},
            "installed_apps": {"smart_home": {}, "alarm": {}},
        },
        "history": [],
    }

    env = FakeToolEnv("public_test", sample["id"], catalog=catalog)
    submission = pipeline.run_sample(sample, env)
    assert submission.tool_calls == []