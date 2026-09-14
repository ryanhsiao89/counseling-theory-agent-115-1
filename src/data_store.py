"""Google Sheets 研究資料層。

API Key 永遠不會傳入此模組。原始逐輪內容只新增、不覆寫。
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import time
import uuid
from datetime import datetime
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import gspread
from gspread.exceptions import APIError

from .config import DEFAULT_SETTINGS


SCHEMAS: dict[str, list[str]] = {
    "IdentityMap": ["participant_id", "email", "created_at", "last_login_at", "role"],
    "Sessions": [
        "session_id", "conversation_thread_id", "participant_id", "agent_type", "mode",
        "continuation_role", "started_at", "ended_at", "duration_seconds", "case_id",
        "school_id", "selected_techniques", "selected_technique_names", "model_name",
        "prompt_version", "temperature", "completion_status", "theme", "difficulty",
    ],
    "ChatLogs": [
        "turn_id", "session_id", "conversation_thread_id", "participant_id", "turn_index",
        "speaker_role", "speaker_id", "content_raw", "nonverbal_cues", "timestamp",
        "stage_at_turn", "skill_labels", "selected_skill_match", "latency_ms", "error_flag",
    ],
    "Threads": [
        "conversation_thread_id", "participant_id", "mode", "continuation_role", "school_id",
        "school_name", "selected_techniques", "selected_technique_names", "case_id", "case_data", "latest_snapshot", "last_session_id",
        "recent_turns", "updated_at", "status",
    ],
    "Assessments": [
        "assessment_id", "session_id", "participant_id", "mode", "school_id", "rubric_version",
        "total_score", "dimension_scores", "skill_events", "strengths", "improvement_points",
        "quoted_examples", "next_practice_focus", "encouragement", "raw_model_output",
        "parsed_json", "created_at",
    ],
    "SkillEvents": [
        "assessment_id", "session_id", "participant_id", "technique_id", "status",
        "turn_index", "quality", "evidence_quote", "effect",
    ],
    "TeacherGrades": [
        "grade_id", "session_id", "participant_id", "teacher_email", "teacher_score",
        "teacher_comment", "created_at",
    ],
    "Settings": ["key", "value", "updated_at", "updated_by"],
    "RiskEvents": [
        "risk_event_id", "session_id", "participant_id", "timestamp", "event_type",
        "action_taken", "content_redacted",
    ],
}


class ServiceAccountFieldsMissingError(ValueError):
    """Required service-account fields are absent."""


class PrivateKeyIncompleteError(ValueError):
    """The PEM body is empty or shorter than its DER length header declares."""


class PrivateKeyEncodingError(ValueError):
    """The PEM body contains characters that are not valid Base64."""


class PrivateKeyParseError(ValueError):
    """The decoded value is not a usable PKCS#8 private key."""


def json_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def parse_json_cell(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if value in (None, ""):
        return default
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return default


def normalize_private_key(value: Any) -> str:
    """Accept either a complete PEM key or the body-only legacy format."""
    key = str(value or "").strip().replace("\\n", "\n")
    if not key:
        return key

    if "-----BEGIN PRIVATE KEY-----" in key and "-----END PRIVATE KEY-----" in key:
        return key

    # The earlier group Agent stored only the Base64 body. Reconstruct the
    # standard PKCS#8 PEM wrapper required by google-auth/cryptography.
    body = "".join(key.split())
    return (
        "-----BEGIN PRIVATE KEY-----\n"
        f"{body}\n"
        "-----END PRIVATE KEY-----\n"
    )


def validate_private_key_structure(pem: str) -> None:
    """Validate PEM/Base64/DER structure without logging any credential text."""
    begin = "-----BEGIN PRIVATE KEY-----"
    end = "-----END PRIVATE KEY-----"
    if begin not in pem or end not in pem:
        raise PrivateKeyIncompleteError

    body = pem.split(begin, 1)[1].split(end, 1)[0]
    compact = "".join(body.split())
    if not compact:
        raise PrivateKeyIncompleteError

    try:
        der = base64.b64decode(compact, validate=True)
    except (binascii.Error, ValueError):
        raise PrivateKeyEncodingError from None

    # DER begins with a SEQUENCE whose encoded length describes the whole key.
    if len(der) < 4 or der[0] != 0x30:
        raise PrivateKeyParseError
    first_length = der[1]
    if first_length & 0x80:
        length_octets = first_length & 0x7F
        if length_octets == 0 or len(der) < 2 + length_octets:
            raise PrivateKeyIncompleteError
        payload_length = int.from_bytes(der[2:2 + length_octets], "big")
        expected_length = 2 + length_octets + payload_length
    else:
        expected_length = 2 + first_length
    if expected_length != len(der):
        raise PrivateKeyIncompleteError


class GoogleSheetsStore:
    def __init__(self, spreadsheet_id: str, service_account: Mapping[str, Any], timezone: str):
        credentials = dict(service_account)
        required = {"client_email", "token_uri", "private_key"}
        if any(not str(credentials.get(field, "")).strip() for field in required):
            raise ServiceAccountFieldsMissingError
        if "private_key" in credentials:
            credentials["private_key"] = normalize_private_key(credentials["private_key"])
            validate_private_key_structure(credentials["private_key"])
        try:
            client = gspread.service_account_from_dict(credentials)
        except ValueError:
            raise PrivateKeyParseError from None
        self.book = client.open_by_key(spreadsheet_id)
        self.timezone = timezone
        self.worksheets: dict[str, gspread.Worksheet] = {}
        # Cache the exact row returned by append_row. Session completion can
        # then update that row without reading the entire worksheet first.
        self._row_cache: dict[tuple[str, str, str], int] = {}
        self.ensure_schema()

    @classmethod
    def from_secrets(cls, secrets: Mapping[str, Any], timezone: str) -> "GoogleSheetsStore":
        # Preferred compatibility path: same syntax as the existing group /
        # helping-skills Agents.
        spreadsheet_id = str(secrets.get("SPREADSHEET_ID", "")).strip()
        service_json = secrets.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
        service_account: Mapping[str, Any] | dict[str, Any] = {}
        if service_json:
            if isinstance(service_json, Mapping):
                service_account = dict(service_json)
            else:
                try:
                    parsed = json.loads(str(service_json))
                except json.JSONDecodeError:
                    raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON 必須是完整 JSON。") from None
                if not isinstance(parsed, dict):
                    raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON 必須是 JSON 物件。")
                service_account = parsed

        # Current nested syntax remains supported for existing deployments.
        block = secrets.get("google_sheets", {})
        if not isinstance(block, Mapping):
            block = {}
        if not spreadsheet_id:
            spreadsheet_id = str(block.get("spreadsheet_id", "")).strip()
        if not service_account:
            nested_service = block.get("service_account", {})
            if isinstance(nested_service, Mapping):
                service_account = nested_service

        # Older deployments used a top-level [gcp_service_account] section.
        if not service_account:
            legacy_service = secrets.get("gcp_service_account", {})
            if isinstance(legacy_service, Mapping):
                service_account = legacy_service
        if not spreadsheet_id or not service_account:
            raise RuntimeError(
                "尚未設定 SPREADSHEET_ID／GOOGLE_SERVICE_ACCOUNT_JSON，"
                "或 google_sheets／gcp_service_account 相容欄位。"
            )
        return cls(spreadsheet_id, service_account, timezone)

    def now(self) -> str:
        return datetime.now(ZoneInfo(self.timezone)).isoformat(timespec="seconds")

    def ensure_schema(self) -> None:
        existing = {ws.title: ws for ws in self.book.worksheets()}
        for name, headers in SCHEMAS.items():
            ws = existing.get(name)
            if ws is None:
                ws = self.book.add_worksheet(title=name, rows=1000, cols=max(20, len(headers) + 2))
                ws.append_row(headers, value_input_option="RAW")
                ws.freeze(rows=1)
            else:
                current = ws.row_values(1)
                if not current:
                    ws.append_row(headers, value_input_option="RAW")
                    ws.freeze(rows=1)
                elif current != headers:
                    missing = [h for h in headers if h not in current]
                    if missing:
                        ws.update(values=[current + missing], range_name="A1")
            self.worksheets[name] = ws
        settings = self.all_records("Settings")
        if not settings:
            for key, value in DEFAULT_SETTINGS.items():
                self.append("Settings", {"key": key, "value": value, "updated_at": self.now(), "updated_by": "system"})

    @staticmethod
    def _run_with_retry(operation, attempts: int = 3):
        """Retry transient Google Sheets quota/server errors briefly."""
        transient_statuses = {429, 500, 502, 503, 504}
        for attempt in range(attempts):
            try:
                return operation()
            except APIError as exc:
                response = getattr(exc, "response", None)
                status = getattr(response, "status_code", None)
                if attempt == attempts - 1 or status not in transient_statuses:
                    raise
                time.sleep(0.35 * (2 ** attempt))

    @staticmethod
    def _row_from_append_response(response: Any) -> int | None:
        if not isinstance(response, Mapping):
            return None
        updated_range = str(response.get("updates", {}).get("updatedRange", ""))
        match = re.search(r"![A-Z]+(\d+)(?::|$)", updated_range, re.IGNORECASE)
        return int(match.group(1)) if match else None

    def append(self, sheet: str, record: Mapping[str, Any]) -> Any:
        headers = SCHEMAS[sheet]
        values = [json_cell(record.get(key, "")) for key in headers]
        return self._run_with_retry(
            lambda: self.worksheets[sheet].append_row(values, value_input_option="RAW")
        )

    def all_records(self, sheet: str) -> list[dict[str, Any]]:
        return self._run_with_retry(
            lambda: self.worksheets[sheet].get_all_records(default_blank="")
        )

    def _upsert_by_key(self, sheet: str, key: str, value: str, record: Mapping[str, Any]) -> None:
        ws = self.worksheets[sheet]
        headers = SCHEMAS[sheet]
        cache_key = (sheet, key, str(value))
        row_index = self._row_cache.get(cache_key)
        if row_index is None:
            key_column = headers.index(key) + 1
            key_values = self._run_with_retry(lambda: ws.col_values(key_column))
            row_index = next(
                (i + 1 for i, cell in enumerate(key_values) if str(cell) == str(value)),
                None,
            )
        values = [json_cell(record.get(header, "")) for header in headers]
        if row_index is None:
            response = self._run_with_retry(
                lambda: ws.append_row(values, value_input_option="RAW")
            )
            row_index = self._row_from_append_response(response)
        else:
            self._run_with_retry(lambda: ws.update(values=[values], range_name=f"A{row_index}"))
        if row_index is not None:
            self._row_cache[cache_key] = row_index

    def get_or_create_participant(self, email: str, role: str, participant_salt: str) -> str:
        normalized = email.strip().lower()
        records = self.all_records("IdentityMap")
        existing = next((r for r in records if str(r.get("email", "")).lower() == normalized), None)
        if existing:
            participant_id = str(existing["participant_id"])
            self._upsert_by_key("IdentityMap", "participant_id", participant_id, {
                **existing, "last_login_at": self.now(), "role": role,
            })
            return participant_id
        digest = hashlib.sha256(f"{participant_salt}:{normalized}".encode("utf-8")).hexdigest()[:12]
        participant_id = f"P-{digest.upper()}"
        self.append("IdentityMap", {
            "participant_id": participant_id,
            "email": normalized,
            "created_at": self.now(),
            "last_login_at": self.now(),
            "role": role,
        })
        return participant_id

    def start_session(self, session: Mapping[str, Any]) -> None:
        response = self.append("Sessions", session)
        row_index = self._row_from_append_response(response)
        if row_index is not None:
            cache_key = ("Sessions", "session_id", str(session["session_id"]))
            self._row_cache[cache_key] = row_index

    def finish_session(self, session: Mapping[str, Any]) -> None:
        self._upsert_by_key("Sessions", "session_id", str(session["session_id"]), session)

    def append_turn(self, turn: Mapping[str, Any]) -> None:
        self.append("ChatLogs", turn)

    def save_thread(self, thread: Mapping[str, Any]) -> None:
        self._upsert_by_key(
            "Threads", "conversation_thread_id", str(thread["conversation_thread_id"]), thread
        )

    def list_threads(self, participant_id: str) -> list[dict[str, Any]]:
        rows = [
            r for r in self.all_records("Threads")
            if str(r.get("participant_id")) == participant_id and str(r.get("status", "active")) == "active"
        ]
        rows.sort(key=lambda r: str(r.get("updated_at", "")), reverse=True)
        return rows

    def save_assessment(self, record: Mapping[str, Any]) -> None:
        self.append("Assessments", record)
        for event in parse_json_cell(record.get("skill_events"), []):
            self.append("SkillEvents", {
                "assessment_id": record.get("assessment_id", ""),
                "session_id": record.get("session_id", ""),
                "participant_id": record.get("participant_id", ""),
                **event,
            })

    def get_settings(self) -> dict[str, str]:
        return {str(r.get("key")): str(r.get("value")) for r in self.all_records("Settings")}

    def save_setting(self, key: str, value: Any, updated_by: str) -> None:
        self._upsert_by_key("Settings", "key", key, {
            "key": key, "value": json_cell(value), "updated_at": self.now(), "updated_by": updated_by,
        })

    def count_sessions(self, participant_id: str) -> int:
        return sum(1 for r in self.all_records("Sessions") if str(r.get("participant_id")) == participant_id)

    def total_usage_seconds(self, participant_id: str) -> int:
        """Sum completed usage once per unique Session."""
        durations: dict[str, int] = {}
        for row in self.all_records("Sessions"):
            if str(row.get("participant_id", "")) != participant_id:
                continue
            if str(row.get("completion_status", "")) not in {"completed", "safety_stopped"}:
                continue
            try:
                seconds = max(0, int(float(row.get("duration_seconds", 0) or 0)))
            except (TypeError, ValueError):
                seconds = 0
            session_id = str(row.get("session_id", ""))
            durations[session_id or f"row-{len(durations)}"] = max(
                seconds, durations.get(session_id, 0)
            )
        return sum(durations.values())

    def session_turns(self, session_id: str) -> list[dict[str, Any]]:
        rows = [r for r in self.all_records("ChatLogs") if str(r.get("session_id")) == session_id]
        return sorted(rows, key=lambda r: int(r.get("turn_index", 0) or 0))

    def get_assessment(self, session_id: str) -> dict[str, Any] | None:
        rows = [r for r in self.all_records("Assessments") if str(r.get("session_id")) == session_id]
        return rows[-1] if rows else None

    def add_teacher_grade(self, session_id: str, participant_id: str, teacher_email: str, score: int | None, comment: str) -> None:
        self.append("TeacherGrades", {
            "grade_id": str(uuid.uuid4()),
            "session_id": session_id,
            "participant_id": participant_id,
            "teacher_email": teacher_email,
            "teacher_score": "" if score is None else score,
            "teacher_comment": comment,
            "created_at": self.now(),
        })


class MemoryStore:
    """僅供本機畫面測試；重新整理或換使用者後資料不保留。"""

    def __init__(self, timezone: str):
        self.timezone = timezone
        self.rows = {name: [] for name in SCHEMAS}
        for key, value in DEFAULT_SETTINGS.items():
            self.rows["Settings"].append({"key": key, "value": value, "updated_at": self.now(), "updated_by": "system"})

    def now(self) -> str:
        return datetime.now(ZoneInfo(self.timezone)).isoformat(timespec="seconds")

    def append(self, sheet: str, record: Mapping[str, Any]) -> None:
        self.rows[sheet].append(dict(record))

    def all_records(self, sheet: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self.rows[sheet]]

    def _upsert_by_key(self, sheet: str, key: str, value: str, record: Mapping[str, Any]) -> None:
        for i, row in enumerate(self.rows[sheet]):
            if str(row.get(key)) == str(value):
                self.rows[sheet][i] = dict(record)
                return
        self.append(sheet, record)

    get_or_create_participant = GoogleSheetsStore.get_or_create_participant
    start_session = GoogleSheetsStore.start_session
    finish_session = GoogleSheetsStore.finish_session
    append_turn = GoogleSheetsStore.append_turn
    save_thread = GoogleSheetsStore.save_thread
    list_threads = GoogleSheetsStore.list_threads
    save_assessment = GoogleSheetsStore.save_assessment
    get_settings = GoogleSheetsStore.get_settings
    save_setting = GoogleSheetsStore.save_setting
    count_sessions = GoogleSheetsStore.count_sessions
    total_usage_seconds = GoogleSheetsStore.total_usage_seconds
    session_turns = GoogleSheetsStore.session_turns
    get_assessment = GoogleSheetsStore.get_assessment
    add_teacher_grade = GoogleSheetsStore.add_teacher_grade
