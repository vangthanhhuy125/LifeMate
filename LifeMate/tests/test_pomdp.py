"""Realistic simulation of the organisers' `zaic_tools.ToolEnv` for self-contained testing."""

import copy
import hashlib
import json
import unicodedata
from typing import Any, Dict, List, Optional


def build_sample_catalog() -> Dict[str, Any]:
    """Build a representative `<split>_tools_catalog.json` dictionary for testing."""
    return {
        "schema_version": "2.7.0",
        "apps": [
            {
                "id": "smart_home",
                "name_vi": "Nhà thông minh",
                "group": "home",
                "description": "Điều khiển thiết bị thông minh trong nhà",
                "tools": [
                    {
                        "name": "smart_home.get_status",
                        "kind": "read",
                        "description": "Lấy danh sách và trạng thái các thiết bị trong nhà",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                        "returns": "devices list",
                        "app": "smart_home",
                    },
                    {
                        "name": "smart_home.climate_set",
                        "kind": "write",
                        "description": "Bật và chỉnh nhiệt độ điều hòa",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "device_id": {
                                    "type": "string",
                                    "description": "ID điều hòa lấy từ smart_home.get_status",
                                    "x-ref": "smart_home.devices",
                                },
                                "mode": {"type": "string", "enum": ["cool", "dry", "fan", "off"]},
                                "temperature": {"type": "integer"},
                            },
                            "required": ["device_id", "mode", "temperature"],
                        },
                        "returns": "",
                        "app": "smart_home",
                    },
                    {
                        "name": "smart_home.light_set",
                        "kind": "write",
                        "description": "Bật/tắt/chỉnh sáng đèn",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "device_id": {
                                    "type": "string",
                                    "description": "ID thiết bị lấy từ smart_home.get_status",
                                    "x-ref": "smart_home.devices",
                                },
                                "power": {"type": "string", "enum": ["on", "off"]},
                                "brightness": {"type": "integer"},
                            },
                            "required": ["device_id", "power"],
                        },
                        "returns": "",
                        "app": "smart_home",
                    },
                ],
            },
            {
                "id": "ride",
                "name_vi": "Gọi xe",
                "group": "transport",
                "description": "Đặt xe di chuyển",
                "tools": [
                    {
                        "name": "ride.book",
                        "kind": "write",
                        "description": "Đặt chuyến xe từ điểm đón đến điểm đến",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "from": {"type": "string", "x-match": "fuzzy"},
                                "to": {"type": "string", "x-match": "fuzzy"},
                            },
                            "required": ["from", "to"],
                        },
                        "returns": "ride booking details",
                        "app": "ride",
                    }
                ],
            },
            {
                "id": "alarm",
                "name_vi": "Báo thức",
                "group": "utility",
                "description": "Quản lý báo thức",
                "tools": [
                    {
                        "name": "alarm.list",
                        "kind": "read",
                        "description": "Xem danh sách báo thức hiện tại",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                        "returns": "alarms",
                        "app": "alarm",
                    },
                    {
                        "name": "alarm.create",
                        "kind": "write",
                        "description": "Tạo báo thức mới",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "time": {"type": "string", "description": "Thời gian ISO 8601"},
                                "label": {"type": "string", "x-match": "fuzzy"},
                            },
                            "required": ["time"],
                        },
                        "returns": "",
                        "app": "alarm",
                    },
                ],
            },
            {
                "id": "music",
                "name_vi": "Âm nhạc",
                "group": "media",
                "description": "Nghe nhạc và quản lý playlist",
                "tools": [
                    {
                        "name": "music.add_to_playlist",
                        "kind": "write",
                        "description": "Thêm bài hát vào danh sách phát",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "playlist_id": {"type": "string"},
                                "track_id": {"type": "string"},
                            },
                            "required": ["playlist_id", "track_id"],
                        },
                        "returns": "",
                        "app": "music",
                    }
                ],
            },
        ],
    }


class FakeToolEnv:
    """Drop-in replacement for `zaic_tools.ToolEnv` during unit & pipeline testing."""

    def __init__(
        self,
        split: str,
        sample_id: str,
        root: str = ".",
        initial_state: Optional[Dict[str, Any]] = None,
        catalog: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.split = split
        self.sample_id = sample_id
        self.root = root
        self.catalog = catalog or build_sample_catalog()
        self._tool_schemas: Dict[str, Dict[str, Any]] = {}
        self._fuzzy_fields: Dict[str, List[str]] = {}
        self._index_catalog()

        self._state: Dict[str, Any] = copy.deepcopy(
            initial_state
            if initial_state is not None
            else {
                "smart_home": {
                    "devices": [
                        {
                            "id": "ac_living_room",
                            "name": "điều hòa phòng khách",
                            "type": "ac",
                            "room": "phòng khách",
                            "floor": 1,
                            "state": {"mode": "off", "temperature": 28},
                        },
                        {
                            "id": "light_living_1",
                            "name": "đèn phòng khách",
                            "type": "light",
                            "room": "phòng khách",
                            "floor": 1,
                            "state": {"power": "on", "brightness": 80},
                        },
                    ]
                },
                "ride": {"bookings": []},
                "alarm": {"alarms": []},
                "asked_missing": [],
            }
        )
        self._call_log: List[Dict[str, Any]] = []

    def _index_catalog(self) -> None:
        for app in self.catalog.get("apps", []):
            for tool in app.get("tools", []):
                t_name = tool["name"]
                self._tool_schemas[t_name] = tool
                props = (tool.get("parameters") or {}).get("properties") or {}
                for p_name, p_def in props.items():
                    if isinstance(p_def, dict) and p_def.get("x-match") == "fuzzy":
                        self._fuzzy_fields.setdefault(t_name, []).append(p_name)

        self._tool_schemas["sys.ask_missing_info"] = {
            "name": "sys.ask_missing_info",
            "kind": "meta",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string"},
                    "params": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["tool_name", "params"],
            },
        }

    def validate(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool existence and argument schema without mutating state or logging."""
        if tool_name not in self._tool_schemas:
            return {"ok": False, "error": f"công cụ không tồn tại: {tool_name}"}
        if not isinstance(arguments, dict):
            return {"ok": False, "error": "arguments phải là object"}

        schema = self._tool_schemas[tool_name].get("parameters") or {}
        required = schema.get("required") or []
        props = schema.get("properties") or {}

        for req in required:
            if req not in arguments or arguments[req] is None or arguments[req] == "":
                return {"ok": False, "error": f"thiếu tham số bắt buộc: {req}"}

        for k, v in arguments.items():
            if k not in props:
                return {"ok": False, "error": f"tham số không hợp lệ: {k}"}
            p_def = props[k]
            expected_type = p_def.get("type")
            if expected_type == "integer" and not isinstance(v, int):
                return {"ok": False, "error": f"sai kiểu dữ liệu cho {k}: cần integer"}
            if expected_type == "string" and not isinstance(v, str):
                return {"ok": False, "error": f"sai kiểu dữ liệu cho {k}: cần string"}
            if "enum" in p_def and v not in p_def["enum"]:
                return {"ok": False, "error": f"giá trị {v} không thuộc enum {p_def['enum']}"}

        return {"ok": True}

    def call(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call, log it unconditionally, update internal state, and return result."""
        norm_args = copy.deepcopy(arguments) if isinstance(arguments, dict) else {}
        self._call_log.append({"name": tool_name, "arguments": norm_args})

        val = self.validate(tool_name, norm_args)
        if not val["ok"]:
            return val

        if tool_name == "smart_home.get_status":
            return {"ok": True, "devices": copy.deepcopy(self._state["smart_home"]["devices"])}

        if tool_name == "smart_home.climate_set":
            dev_id = norm_args["device_id"]
            for dev in self._state["smart_home"]["devices"]:
                if dev["id"] == dev_id:
                    dev["state"]["mode"] = norm_args["mode"]
                    dev["state"]["temperature"] = norm_args["temperature"]
                    return {"ok": True, "device": copy.deepcopy(dev)}
            return {"ok": False, "error": f"không tìm thấy thiết bị: {dev_id}"}

        if tool_name == "smart_home.light_set":
            dev_id = norm_args["device_id"]
            for dev in self._state["smart_home"]["devices"]:
                if dev["id"] == dev_id:
                    dev["state"]["power"] = norm_args["power"]
                    if "brightness" in norm_args:
                        dev["state"]["brightness"] = norm_args["brightness"]
                    return {"ok": True, "device": copy.deepcopy(dev)}
            return {"ok": False, "error": f"không tìm thấy thiết bị: {dev_id}"}

        if tool_name == "ride.book":
            booking = {"status": "booked", "has_route": bool(norm_args.get("from") and norm_args.get("to"))}
            self._state["ride"]["bookings"].append(booking)
            return {"ok": True, "booking_id": "ride_01"}

        if tool_name == "alarm.list":
            return {"ok": True, "alarms": copy.deepcopy(self._state["alarm"]["alarms"])}

        if tool_name == "alarm.create":
            alarm_obj = {"time": norm_args["time"]}
            self._state["alarm"]["alarms"].append(alarm_obj)
            return {"ok": True, "alarm": alarm_obj}

        if tool_name == "sys.ask_missing_info":
            self._state["asked_missing"].append(
                {
                    "tool_name": norm_args["tool_name"],
                    "params": sorted(norm_args["params"]),
                }
            )
            return {"ok": True}

        return {"ok": True}

    def get_tool_calls(self) -> List[Dict[str, Any]]:
        """Return ordered list of all tool calls made on this environment."""
        return copy.deepcopy(self._call_log)

    def get_state_hash(self) -> str:
        """Compute deterministic 16-char hex digest of the final app state (excluding fuzzy text)."""
        canonical = unicodedata.normalize(
            "NFC", json.dumps(self._state, sort_keys=True, ensure_ascii=False)
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]