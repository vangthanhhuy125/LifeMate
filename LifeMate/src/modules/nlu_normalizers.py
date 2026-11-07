"""Vietnamese NLU Normalizer for Unicode NFC, relative ISO-8601 time, currency, and clues."""

import re
import unicodedata
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple


class NLUNormalizer:
    """Normalizes spoken Vietnamese transcripts and extracts structured grounding clues."""

    VI_DIGITS = {
        "không": 0,
        "một": 1,
        "mốt": 1,
        "hai": 2,
        "ba": 3,
        "bốn": 4,
        "tư": 4,
        "năm": 5,
        "lăm": 5,
        "sáu": 6,
        "bảy": 7,
        "bẩy": 7,
        "tám": 8,
        "chín": 9,
        "mười": 10,
    }

    ROOM_CANONICAL_MAP = {
        "phòng khách": "phòng khách",
        "phòng ngủ": "phòng ngủ",
        "phòng bếp": "phòng bếp",
        "nhà bếp": "phòng bếp",
        "bếp": "phòng bếp",
        "phòng tắm": "phòng tắm",
        "nhà vệ sinh": "phòng tắm",
        "phòng làm việc": "phòng làm việc",
        "ban công": "ban công",
        "sân thượng": "sân thượng",
    }

    DEVICE_TYPE_MAP = {
        "điều hòa": "ac",
        "máy lạnh": "ac",
        "ac": "ac",
        "đèn": "light",
        "bóng đèn": "light",
        "quạt": "fan",
        "rèm": "curtain",
        "rèm cửa": "curtain",
        "tivi": "tv",
        "tv": "tv",
        "cửa": "lock",
        "khóa cửa": "lock",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        nlu_cfg = self.config.get("nlu", self.config)
        self.enable_nfc = bool(nlu_cfg.get("enable_unicode_nfc", True))
        self.enable_time = bool(nlu_cfg.get("enable_time_normalizer", True))
        self.enable_number = bool(nlu_cfg.get("enable_number_normalizer", True))
        self.enable_profile = bool(nlu_cfg.get("enable_profile_grounding", True))
        self.default_tz = str(nlu_cfg.get("default_timezone", "Asia/Ho_Chi_Minh"))

    @staticmethod
    def normalize_nfc(text: str) -> str:
        """Standardize Vietnamese text into Unicode NFC canonical form."""
        if not text:
            return ""
        text = unicodedata.normalize("NFC", text)
        return re.sub(r"\s+", " ", text).strip()

    def convert_spoken_numbers(self, text: str) -> str:
        """Convert common Vietnamese spoken numbers and currency into digits."""
        if not self.enable_number or not text:
            return text

        out = text

        # Convert patterns like "500 nghìn", "500 ngàn", "2 triệu", "1 triệu rưỡi"
        def _replace_currency(match: re.Match) -> str:
            num_str = match.group(1).replace(",", ".")
            unit = match.group(2).lower()
            half = match.group(3)
            try:
                val = float(num_str)
            except ValueError:
                return match.group(0)
            if unit in ("nghìn", "ngàn", "k"):
                multiplier = 1_000
            elif unit in ("triệu", "củ", "tr"):
                multiplier = 1_000_000
            elif unit in ("chục",):
                multiplier = 10
            elif unit in ("trăm",):
                multiplier = 100
            else:
                multiplier = 1
            total = int(val * multiplier)
            if half and ("rưỡi" in half.lower()):
                total += multiplier // 2
            return str(total)

        out = re.sub(
            r"\b(\d+(?:[.,]\d+)?)\s*(nghìn|ngàn|triệu|củ|trăm|chục|k|tr)(?:\s+(rưỡi))?\b",
            _replace_currency,
            out,
            flags=re.IGNORECASE,
        )

        # Spoken hour patterns: "ba giờ" -> "3 giờ", "mười lăm giờ" -> "15 giờ"
        for word, digit in self.VI_DIGITS.items():
            out = re.sub(
                rf"\b{word}\s+(giờ|độ|phút|tầng|lầu)\b",
                rf"{digit} \1",
                out,
                flags=re.IGNORECASE,
            )

        return out

    @staticmethod
    def _parse_context_datetime(current_time_str: str) -> Tuple[datetime, str]:
        """Parse ISO-8601 `current_time` from context and preserve timezone offset string."""
        if not current_time_str:
            return datetime(2026, 11, 16, 12, 0, 0), "+07:00"
        tz_suffix = "+07:00"
        match_tz = re.search(r"([+-]\d{2}:\d{2}|Z)$", current_time_str)
        if match_tz:
            tz_suffix = "+00:00" if match_tz.group(1) == "Z" else match_tz.group(1)
        try:
            dt = datetime.fromisoformat(current_time_str.replace("Z", "+00:00"))
            return dt, tz_suffix
        except ValueError:
            return datetime(2026, 11, 16, 12, 0, 0), tz_suffix

    def extract_and_resolve_datetime(
        self, text: str, current_time_str: str
    ) -> Optional[str]:
        """Resolve Vietnamese relative day and time expressions to an ISO-8601 string."""
        if not self.enable_time or not text:
            return None

        base_dt, tz_suffix = self._parse_context_datetime(current_time_str)
        lower = text.lower()

        # Check relative offset like "sau 30 phút", "15 phút nữa", "2 tiếng nữa"
        rel_min = re.search(r"(\d+)\s*phút\s*(?:nữa|tới|sau)", lower)
        if rel_min:
            target = base_dt + timedelta(minutes=int(rel_min.group(1)))
            return target.strftime(f"%Y-%m-%dT%H:%M:%S{tz_suffix}")

        rel_hour = re.search(r"(\d+)\s*(?:tiếng|giờ)\s*(?:nữa|tới|sau)", lower)
        if rel_hour:
            target = base_dt + timedelta(hours=int(rel_hour.group(1)))
            return target.strftime(f"%Y-%m-%dT%H:%M:%S{tz_suffix}")

        # Determine day offset
        day_offset = 0
        has_time_reference = False
        if "ngày kia" in lower:
            day_offset = 2
            has_time_reference = True
        elif "ngày mốt" in lower or "bữa mốt" in lower:
            day_offset = 2
            has_time_reference = True
        elif any(kw in lower for kw in ("ngày mai", "chiều mai", "sáng mai", "tối mai", "trưa mai", "đêm mai", "hôm sau")):
            day_offset = 1
            has_time_reference = True
        elif any(kw in lower for kw in ("hôm nay", "chiều nay", "tối nay", "sáng nay", "trưa nay", "đêm nay")):
            day_offset = 0
            has_time_reference = True

        # Extract hour and minute: e.g., "3 giờ 30 chiều", "5 giờ rưỡi sáng", "14 giờ", "8h30"
        time_match = re.search(
            r"\b(\d{1,2})\s*(?:giờ|h|:)\s*(?:(\d{1,2}|rưỡi)\s*(?:phút)?)?\s*(sáng|trưa|chiều|tối|đêm)?",
            lower,
        )

        if not time_match and not has_time_reference:
            return None

        hour = base_dt.hour
        minute = 0

        if time_match:
            has_time_reference = True
            hour = int(time_match.group(1))
            min_raw = time_match.group(2)
            period = time_match.group(3) or ""

            if not period:
                if any(p in lower for p in ("buổi chiều", "chiều mai", "chiều nay")):
                    period = "chiều"
                elif any(p in lower for p in ("buổi tối", "tối mai", "tối nay", "đêm")):
                    period = "tối"
                elif any(p in lower for p in ("buổi sáng", "sáng mai", "sáng nay")):
                    period = "sáng"
                elif any(p in lower for p in ("buổi trưa", "trưa mai", "trưa nay")):
                    period = "trưa"

            if min_raw:
                minute = 30 if min_raw == "rưỡi" else int(min_raw)

            if period in ("chiều", "tối", "đêm") and hour < 12:
                hour += 12
            elif period == "trưa" and hour < 11:
                hour += 12
            elif period == "sáng" and hour == 12:
                hour = 0

        target_date = (base_dt + timedelta(days=day_offset)).date()
        resolved = datetime(
            target_date.year,
            target_date.month,
            target_date.day,
            min(max(hour, 0), 23),
            min(max(minute, 0), 59),
            0,
        )

        # If no explicit day was mentioned and the time already passed today, roll to tomorrow for alarms/reminders
        if (
            day_offset == 0
            and not any(k in lower for k in ("hôm nay", "chiều nay", "tối nay", "sáng nay", "trưa nay"))
            and resolved < base_dt.replace(tzinfo=None)
            and any(k in lower for k in ("báo thức", "nhắc", "đánh thức", "dậy"))
        ):
            resolved += timedelta(days=1)

        return resolved.strftime(f"%Y-%m-%dT%H:%M:%S{tz_suffix}")

    def extract_structured_clues(
        self, normalized_text: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract grounded parameters (temperature, brightness, room, addresses, amounts, time)."""
        text = normalized_text
        lower = text.lower()
        clues: Dict[str, Any] = {}

        # 1. Resolved ISO timestamp
        current_time = str(context.get("current_time", "2026-11-16T17:00:00+07:00"))
        iso_time = self.extract_and_resolve_datetime(text, current_time)
        if iso_time:
            clues["resolved_time_iso"] = iso_time

        # 2. Temperature (e.g., "26 độ", "24 độ C")
        temp_match = re.search(r"\b(1[6-9]|2\d|3[0-2])\s*độ\b", lower)
        if temp_match:
            clues["temperature"] = int(temp_match.group(1))

        # 3. Brightness / percentage (e.g., "độ sáng 80%", "50 phần trăm")
        bright_match = re.search(r"\b(\d{1,3})\s*(?:%|phần trăm)\b", lower)
        if bright_match:
            val = int(bright_match.group(1))
            if 0 <= val <= 100:
                clues["percentage_value"] = val

        # 4. Floor number (e.g., "tầng 1", "lầu 2")
        floor_match = re.search(r"\b(?:tầng|lầu)\s*(\d+)\b", lower)
        if floor_match:
            clues["floor"] = int(floor_match.group(1))

        # 5. Room & device type
        for phrase, canonical_room in self.ROOM_CANONICAL_MAP.items():
            if phrase in lower:
                clues["room"] = canonical_room
                break

        detected_devices: List[str] = []
        for phrase, dev_type in self.DEVICE_TYPE_MAP.items():
            if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", lower):
                if dev_type not in detected_devices:
                    detected_devices.append(dev_type)
        if detected_devices:
            clues["device_types"] = detected_devices

        # 6. Power / mode intentions
        if any(w in lower for w in ("bật", "mở", "khởi động")):
            clues["power"] = "on"
        elif any(w in lower for w in ("tắt", "đóng", "ngắt")):
            clues["power"] = "off"

        if "làm lạnh" in lower or "mát" in lower or "bật điều hòa" in lower or "chỉnh điều hòa" in lower:
            clues["ac_mode"] = "cool"

        # 7. Currency amount (for banking transfers/payments)
        amount_matches = [
            int(m.group(1))
            for m in re.finditer(r"\b(\d{4,10})\b", lower)
            if not re.search(rf"0\d{{8,10}}", m.group(1))
        ]
        if amount_matches:
            clues["amount"] = amount_matches[0]

        # 8. User profile & location grounding (for ride booking / delivery)
        if self.enable_profile:
            profile = context.get("user_profile") or {}
            home_addr = (
                profile.get("home")
                or profile.get("home_address")
                or profile.get("address")
            )
            work_addr = (
                profile.get("work")
                or profile.get("work_address")
                or profile.get("office")
            )
            curr_loc = (
                context.get("location")
                or context.get("current_location")
                or profile.get("current_location")
            )
            if isinstance(curr_loc, dict):
                curr_loc = curr_loc.get("address") or curr_loc.get("name")

            if any(w in lower for w in ("về nhà", "tới nhà", "đến nhà")):
                if home_addr:
                    clues["destination_address"] = self.normalize_nfc(str(home_addr))
                if curr_loc:
                    clues["pickup_address"] = self.normalize_nfc(str(curr_loc))
                elif work_addr:
                    clues["pickup_address"] = self.normalize_nfc(str(work_addr))
            elif any(w in lower for w in ("đến công ty", "đến cơ quan", "đi làm", "tới văn phòng")):
                if work_addr:
                    clues["destination_address"] = self.normalize_nfc(str(work_addr))
                if curr_loc:
                    clues["pickup_address"] = self.normalize_nfc(str(curr_loc))
                elif home_addr:
                    clues["pickup_address"] = self.normalize_nfc(str(home_addr))

            if "đón tôi" in lower and "pickup_address" not in clues:
                if curr_loc:
                    clues["pickup_address"] = self.normalize_nfc(str(curr_loc))
                elif work_addr and "về nhà" in lower:
                    clues["pickup_address"] = self.normalize_nfc(str(work_addr))
                elif home_addr:
                    clues["pickup_address"] = self.normalize_nfc(str(home_addr))

        # 9. Relation / Contact recipient clues
        for rel in ("mẹ", "bố", "ba", "má", "anh", "chị", "em", "vợ", "chồng", "sếp", "bạn"):
            if re.search(rf"\b(?:cho|tới|gửi)\s+{rel}\b", lower):
                clues["recipient_relation_or_name"] = rel
                break

        name_match = re.search(
            r"\b(?:cho|tới|gửi)\s+(?:bạn\s+|anh\s+|chị\s+|em\s+)?([A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]+)",
            text,
        )
        if name_match:
            clues["recipient_name"] = self.normalize_nfc(name_match.group(1))

        # 10. Reminder / note content
        remind_match = re.search(
            r"\bnhắc\s+(?:tôi\s+|mình\s+)?(.+?)(?:\s+lúc\s+|\s+vào\s+|$)",
            text,
            flags=re.IGNORECASE,
        )
        if remind_match:
            clues["reminder_title"] = self.normalize_nfc(remind_match.group(1).strip())

        return clues

    def normalize(
        self, transcript: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run full NLU normalization pipeline on transcript and context."""
        ctx = context or {}
        step1 = self.normalize_nfc(transcript) if self.enable_nfc else transcript.strip()
        step2 = self.convert_spoken_numbers(step1)
        step3 = self.normalize_nfc(step2) if self.enable_nfc else step2
        clues = self.extract_structured_clues(step3, ctx)
        return {
            "normalized_text": step3,
            "clues": clues,
        }