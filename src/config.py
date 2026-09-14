"""應用程式設定讀取與預設值。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


def _section(source: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = source.get(name, {})
    return value if isinstance(value, Mapping) else {}


def _string_tuple(value: Any, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return (value.strip().lower(),) if value.strip() else default
    return tuple(str(item).strip().lower() for item in value if str(item).strip())


def _unique_strings(*groups: tuple[str, ...]) -> tuple[str, ...]:
    """Merge email/domain lists while preserving their configured order."""
    return tuple(dict.fromkeys(item for group in groups for item in group))


@dataclass(frozen=True)
class AppConfig:
    app_title: str
    timezone: str
    model_name: str
    prompt_version: str
    rubric_version: str
    allowed_domains: tuple[str, ...]
    login_allowlist: tuple[str, ...]
    teacher_emails: tuple[str, ...]
    otp_ttl_seconds: int
    max_input_chars: int
    recent_context_turns: int
    semester_target_minutes: int

    @classmethod
    def from_secrets(cls, secrets: Mapping[str, Any]) -> "AppConfig":
        app = _section(secrets, "app")
        auth = _section(secrets, "auth")
        legacy_test_emails = _string_tuple(app.get("teacher_test_emails"))
        student_login_emails = _string_tuple(app.get("student_login_emails"))
        allowed_domains = _string_tuple(
            auth.get("allowed_domains", app.get("allowed_domains", app.get("allowed_domain"))),
            ("hcu.edu.tw",),
        )
        # teacher_test_emails is retained for compatibility with the existing
        # Agents. student_login_emails grants login only; it never grants access
        # to the teacher dashboard.
        login_allowlist = _unique_strings(
            legacy_test_emails,
            student_login_emails,
            _string_tuple(auth.get("login_allowlist")),
        )
        teacher_emails = _string_tuple(auth.get("teacher_emails"), legacy_test_emails)
        return cls(
            app_title=str(app.get("title", "諮商理論技巧訓練 Agent")),
            timezone=str(app.get("timezone", "Asia/Taipei")),
            model_name=str(app.get("model_name", "gemini-3.8-flash")),
            prompt_version=str(app.get("prompt_version", "theory-dialogue-v1.0")),
            rubric_version=str(app.get("rubric_version", "theory-rubric-v1.0")),
            allowed_domains=allowed_domains,
            login_allowlist=login_allowlist,
            teacher_emails=teacher_emails,
            otp_ttl_seconds=int(auth.get("otp_ttl_seconds", 600)),
            max_input_chars=int(app.get("max_input_chars", 800)),
            recent_context_turns=int(app.get("recent_context_turns", 14)),
            semester_target_minutes=max(1, int(app.get("semester_target_minutes", 120))),
        )


DEFAULT_SETTINGS = {
    "system_enabled": "true",
    "open_start": "",
    "open_end": "",
    "max_sessions_per_student": "0",
    "duration_experience_min": "8",
    "duration_practice_min": "15",
    "allowed_modes": "experience,practice",
    "student_feedback_visible": "true",
    "student_score_visible": "true",
}


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
