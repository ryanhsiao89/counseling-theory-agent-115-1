from __future__ import annotations

import io
import json
import logging
import time
import uuid
import zipfile
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from src.auth import create_otp, is_email_allowed, is_teacher, normalize_email, send_otp_email, verify_otp
from src.config import AppConfig, DEFAULT_SETTINGS, as_bool
from src.data_store import GoogleSheetsStore, MemoryStore, SCHEMAS, json_cell, parse_json_cell
from src.gemini_client import GeminiQuotaError, GeminiService, parse_json_response
from src.prompts import (
    build_case_prompt,
    build_dialogue_prompt,
    build_experience_analysis_prompt,
    build_practice_evaluator_prompt,
    build_snapshot_prompt,
)
from src.safety import detect_immediate_risk, detect_pii, redact_for_preview, safety_message
from src.session_service import finish_session, new_session, new_turn
from src.theory_library import PRACTICE_THEMES, SCHOOLS, get_school, get_techniques, validate_selected_techniques
from src.transcript import make_transcript_txt, safe_filename


st.set_page_config(page_title="諮商理論技巧訓練 Agent", page_icon="🧭", layout="wide")
LOGGER = logging.getLogger(__name__)


def secrets_dict() -> dict[str, Any]:
    try:
        return st.secrets.to_dict()
    except Exception:
        return {}


SECRETS = secrets_dict()
CONFIG = AppConfig.from_secrets(SECRETS)


def initialize_state() -> None:
    defaults = {
        "authenticated": False,
        "email": "",
        "participant_id": "",
        "view": "student",
        "api_key": "",
        "api_validated": False,
        "active_session": None,
        "turns": [],
        "case_data": None,
        "continuation_snapshot": None,
        "prior_turns_context": [],
        "assessment": None,
        "raw_assessment": "",
        "otp_hash": "",
        "otp_expires": 0.0,
        "otp_email": "",
        "otp_last_sent": 0.0,
        "settings_cache": None,
        "settings_cache_at": 0.0,
        "usage_seconds_cache": None,
        "usage_participant_id": "",
        "usage_error": "",
        "persistence_warnings": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_state()


@st.cache_resource(show_spinner=False)
def build_shared_store(serialized_secrets: str, timezone: str):
    values = json.loads(serialized_secrets)
    return GoogleSheetsStore.from_secrets(values, timezone)


def get_store():
    if "data_store" in st.session_state:
        return st.session_state.data_store
    try:
        store = build_shared_store(json.dumps(SECRETS, sort_keys=True), CONFIG.timezone)
        st.session_state.store_mode = "google_sheets"
    except Exception as exc:
        store = MemoryStore(CONFIG.timezone)
        st.session_state.store_mode = "memory"
        # Never expose credential fragments returned by google-auth/cryptography.
        st.session_state.store_error = (
            "Google Sheets 驗證失敗"
            f"（安全錯誤類型：{type(exc).__name__}）。"
            "請檢查服務帳戶欄位、private_key 格式與試算表共用權限。"
        )
    st.session_state.data_store = store
    return store


STORE = get_store()


def participant_salt() -> str:
    auth = SECRETS.get("auth", {})
    app = SECRETS.get("app", {})
    value = str(auth.get("participant_salt", app.get("participant_salt", ""))).strip()
    return value or "local-demo-change-this-salt"


def smtp_secrets() -> dict[str, Any]:
    """Support both the new [smtp] and existing Agent [email] syntax."""
    value = SECRETS.get("smtp") or SECRETS.get("email") or {}
    return dict(value) if isinstance(value, dict) else value


def local_demo_enabled() -> bool:
    return as_bool(SECRETS.get("app", {}).get("local_demo_mode", False))


def logout() -> None:
    for key in list(st.session_state.keys()):
        if key not in {"data_store", "store_mode", "store_error"}:
            del st.session_state[key]
    st.rerun()


def render_header() -> None:
    st.title(CONFIG.app_title)
    st.caption("11 學派 × 體驗與實作 × 跨次續談 × 形成性回饋")
    st.warning(
        "本系統僅供教學演練，不提供心理治療、診斷、臨床決策或緊急危機服務。"
        "請勿輸入真實個案姓名、電話、地址、學校或機構等可識別資訊。",
        icon="⚠️",
    )


def login_page() -> None:
    render_header()
    if st.session_state.store_mode == "memory" and not local_demo_enabled():
        st.error(
            "Google Sheets 尚未連線，為避免學生練習紀錄遺失，正式模式已停止登入。"
            "請先完成 Streamlit Secrets 與試算表共用權限設定。"
        )
        st.caption(st.session_state.get("store_error", ""))
        return
    st.subheader("登入")
    st.write(
        "學生原則上請用學校 `@hcu.edu.tw` 信箱接收驗證碼；"
        "經教師核准的學分班帳號可使用指定 Gmail。"
    )
    email = normalize_email(st.text_input("登入 Email", value=st.session_state.otp_email))
    col1, col2 = st.columns(2)
    with col1:
        if st.button("寄送驗證碼", use_container_width=True):
            if not is_email_allowed(email, CONFIG.allowed_domains, CONFIG.login_allowlist):
                st.error("此信箱不在允許的學校網域或測試白名單中。")
            elif time.time() - float(st.session_state.otp_last_sent or 0) < 60:
                st.error("請等待 60 秒後再重新寄送驗證碼。")
            else:
                code, digest, expires = create_otp(CONFIG.otp_ttl_seconds)
                try:
                    if local_demo_enabled():
                        st.info(f"本機測試驗證碼：{code}")
                    else:
                        send_otp_email(email, code, smtp_secrets())
                    st.session_state.otp_hash = digest
                    st.session_state.otp_expires = expires
                    st.session_state.otp_email = email
                    st.session_state.otp_last_sent = time.time()
                    st.success("驗證碼已寄出，請查看收件匣與垃圾郵件匣。")
                except Exception as exc:
                    st.error(f"驗證碼寄送失敗：{exc}")
    with col2:
        otp = st.text_input("六位數驗證碼", max_chars=6)
        if st.button("驗證並登入", use_container_width=True):
            if email != st.session_state.otp_email:
                st.error("目前輸入的 Email 與接收驗證碼的 Email 不同。")
            elif not verify_otp(otp, st.session_state.otp_hash, st.session_state.otp_expires):
                st.error("驗證碼錯誤或已逾時。")
            else:
                role = "teacher" if is_teacher(email, CONFIG.teacher_emails) else "student"
                participant_id = STORE.get_or_create_participant(email, role, participant_salt())
                st.session_state.authenticated = True
                st.session_state.email = email
                st.session_state.participant_id = participant_id
                st.session_state.view = "teacher" if role == "teacher" else "student"
                st.rerun()

    if st.session_state.store_mode == "memory":
        st.info("目前為本機暫存模式；部署正式版前必須完成 Google Sheets Secrets 設定。")


def sidebar() -> None:
    with st.sidebar:
        st.markdown(f"**已登入：** {st.session_state.email}")
        st.caption(f"教學代碼：{st.session_state.participant_id}")
        if is_teacher(st.session_state.email, CONFIG.teacher_emails):
            st.session_state.view = st.radio(
                "使用介面", ["student", "teacher"],
                format_func=lambda x: "學生模擬端" if x == "student" else "教師後台",
                index=0 if st.session_state.view == "student" else 1,
            )
        st.divider()
        st.markdown("**兩個帳號的用途**")
        st.caption("學校信箱：登入本系統。\n\n個人 Gmail：到 Google AI Studio 申請自己的 Gemini API Key。")
        if st.button("登出", use_container_width=True):
            logout()


def settings_with_defaults(force: bool = False) -> dict[str, str]:
    values = dict(DEFAULT_SETTINGS)
    cached = st.session_state.get("settings_cache")
    cache_age = time.time() - float(st.session_state.get("settings_cache_at", 0.0) or 0.0)
    if not force and isinstance(cached, dict) and cache_age < 60:
        values.update(cached)
        return values
    try:
        remote = STORE.get_settings()
        st.session_state.settings_cache = remote
        st.session_state.settings_cache_at = time.time()
        values.update(remote)
    except Exception:
        if isinstance(cached, dict):
            values.update(cached)
    return values


def safe_store_action(label: str, operation, *args, **kwargs) -> bool:
    """Keep the student UI usable when a transient Sheets write fails."""
    try:
        operation(*args, **kwargs)
        return True
    except Exception:
        LOGGER.exception("Google Sheets write failed: %s", label)
        warnings = list(st.session_state.get("persistence_warnings", []))
        if label not in warnings:
            warnings.append(label)
        st.session_state.persistence_warnings = warnings
        return False


def usage_seconds() -> int | None:
    participant_id = st.session_state.participant_id
    if (
        st.session_state.get("usage_participant_id") == participant_id
        and st.session_state.get("usage_seconds_cache") is not None
    ):
        return int(st.session_state.usage_seconds_cache)
    try:
        seconds = STORE.total_usage_seconds(participant_id)
    except Exception:
        LOGGER.exception("Unable to load accumulated usage")
        st.session_state.usage_error = "累積時間暫時無法從 Google Sheets 讀取，請稍後重新整理。"
        return None
    st.session_state.usage_seconds_cache = seconds
    st.session_state.usage_participant_id = participant_id
    st.session_state.usage_error = ""
    return seconds


def add_completed_usage(session: dict[str, Any]) -> None:
    if st.session_state.get("usage_participant_id") != session.get("participant_id"):
        st.session_state.usage_seconds_cache = None
        return
    cached = st.session_state.get("usage_seconds_cache")
    if cached is not None:
        st.session_state.usage_seconds_cache = int(cached) + max(
            0, int(session.get("duration_seconds", 0) or 0)
        )


def render_usage_progress() -> None:
    seconds = usage_seconds()
    if seconds is None:
        st.info(st.session_state.get("usage_error", "累積時間暫時無法讀取。"))
        return
    target_minutes = CONFIG.semester_target_minutes
    completed_minutes = seconds / 60
    remaining_minutes = max(0.0, target_minutes - completed_minutes)
    st.markdown("### 本學期上機累積")
    c1, c2 = st.columns([1, 2])
    c1.metric("已累積使用時間", f"{completed_minutes:.1f} / {target_minutes} 分鐘")
    with c2:
        st.progress(min(completed_minutes / target_minutes, 1.0))
        if remaining_minutes > 0:
            st.caption(f"距離本學期目標尚需 {remaining_minutes:.1f} 分鐘。以已結束並保存的 Session 計算。")
        else:
            st.success(f"已達成本學期 {target_minutes} 分鐘上機目標！")


def parse_local_datetime(value: str) -> datetime | None:
    value = (value or "").strip()
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo(CONFIG.timezone))
    return parsed


def student_access_error(settings: dict[str, str]) -> str | None:
    if not as_bool(settings.get("system_enabled"), True):
        return "目前系統未開放。"
    now = datetime.now(ZoneInfo(CONFIG.timezone))
    try:
        start = parse_local_datetime(settings.get("open_start", ""))
        end = parse_local_datetime(settings.get("open_end", ""))
    except ValueError:
        return "教師端開放時間格式錯誤，請通知授課教師。"
    if start and now < start:
        return f"系統將於 {start.isoformat(timespec='minutes')} 開放。"
    if end and now > end:
        return "本次練習開放時間已結束。"
    max_sessions = int(settings.get("max_sessions_per_student", "0") or 0)
    if max_sessions > 0 and STORE.count_sessions(st.session_state.participant_id) >= max_sessions:
        return f"你已達教師設定的 {max_sessions} 次使用上限。"
    return None


def api_key_gate() -> GeminiService | None:
    st.subheader("連接你自己的 Gemini API Key")
    st.write(
        "請用個人的 `@gmail.com` 帳號到 Google AI Studio 申請 API Key，再貼到下方。"
        "Key 僅保留於目前瀏覽器工作階段，不會寫入 Google Sheets、逐字稿或研究資料。"
    )
    st.caption("「測試 API Key」只檢查連線與模型權限，不會消耗一次對話生成額度；實際可用額度仍以 Google AI Studio 為準。")
    key = st.text_input("Gemini API Key", type="password", value=st.session_state.api_key)
    if st.button("測試 API Key"):
        try:
            with st.spinner("正在測試連線…"):
                service = GeminiService(key, CONFIG.model_name)
                service.validate_key()
            st.session_state.api_key = key.strip()
            st.session_state.api_validated = True
            st.success("API Key 已驗證，可以開始練習。")
            st.rerun()
        except Exception as exc:
            st.session_state.api_validated = False
            st.error(str(exc))
    return None


def gemini() -> GeminiService:
    return GeminiService(st.session_state.api_key, CONFIG.model_name)


def store_turn(turn: dict[str, Any]) -> None:
    st.session_state.turns.append(turn)
    safe_store_action("本輪逐字稿", STORE.append_turn, turn)


def generate_ai_turn(is_opening: bool, latest_student_message: str = "") -> None:
    session = st.session_state.active_session
    system, prompt = build_dialogue_prompt(
        mode=session["mode"],
        school_id=session["school_id"],
        selected_ids=session["selected_techniques"],
        turns=st.session_state.prior_turns_context + st.session_state.turns,
        latest_student_message=latest_student_message,
        case_data=st.session_state.case_data,
        continuation_snapshot=st.session_state.continuation_snapshot,
        is_opening=is_opening,
    )
    started = time.perf_counter()
    response = gemini().generate_text(
        prompt,
        system_instruction=system,
        temperature=0.55,
        max_output_tokens=2048,
    )
    latency = int((time.perf_counter() - started) * 1000)
    role = "ai_client" if session["mode"] == "practice" else "ai_counselor"
    store_turn(new_turn(
        session=session,
        turn_index=len(st.session_state.turns) + 1,
        speaker_role=role,
        content=response,
        timezone=CONFIG.timezone,
        latency_ms=latency,
    ))


def start_new_session(mode: str, school_id: str, selected_ids: list[str], theme: str, difficulty: str) -> None:
    validate_selected_techniques(school_id, selected_ids)
    case_data = None
    case_id = "student_topic"
    if mode == "practice":
        raw_case = gemini().generate_text(
            build_case_prompt(school_id, selected_ids, theme, difficulty),
            system_instruction="你是標準化諮商教學案例設計器，只輸出符合 schema 的 JSON。",
            temperature=0.65,
            max_output_tokens=1500,
            response_json=True,
        )
        case_data = parse_json_response(raw_case)
        case_id = str(case_data.get("case_id", f"case-{uuid.uuid4().hex[:8]}"))
    session = new_session(
        participant_id=st.session_state.participant_id,
        mode=mode,
        school_id=school_id,
        selected_ids=selected_ids,
        model_name=CONFIG.model_name,
        prompt_version=CONFIG.prompt_version,
        timezone=CONFIG.timezone,
        theme=theme,
        difficulty=difficulty,
        case_id=case_id,
    )
    st.session_state.active_session = session
    st.session_state.turns = []
    st.session_state.case_data = case_data
    st.session_state.continuation_snapshot = None
    st.session_state.prior_turns_context = []
    st.session_state.assessment = None
    st.session_state.persistence_warnings = []
    STORE.start_session(session)
    generate_ai_turn(is_opening=True)


def start_continuation(thread: dict[str, Any], selected_ids: list[str]) -> None:
    mode = str(thread["mode"])
    school_id = str(thread["school_id"])
    validate_selected_techniques(school_id, selected_ids)
    session = new_session(
        participant_id=st.session_state.participant_id,
        mode=mode,
        school_id=school_id,
        selected_ids=selected_ids,
        model_name=CONFIG.model_name,
        prompt_version=CONFIG.prompt_version,
        timezone=CONFIG.timezone,
        theme="續談上次議題",
        difficulty="延續前次",
        thread_id=str(thread["conversation_thread_id"]),
        case_id=str(thread.get("case_id", "student_topic")),
    )
    st.session_state.active_session = session
    st.session_state.turns = []
    st.session_state.case_data = parse_json_cell(thread.get("case_data"), None)
    st.session_state.continuation_snapshot = parse_json_cell(thread.get("latest_snapshot"), {})
    st.session_state.prior_turns_context = parse_json_cell(thread.get("recent_turns"), [])
    st.session_state.assessment = None
    st.session_state.persistence_warnings = []
    STORE.start_session(session)
    generate_ai_turn(is_opening=True)


def new_practice_panel(settings: dict[str, str]) -> None:
    allowed_modes = {x.strip() for x in settings.get("allowed_modes", "experience,practice").split(",")}
    mode_options = [m for m in ("experience", "practice") if m in allowed_modes]
    if not mode_options:
        st.error("教師目前未開放任何模式。")
        return
    mode = st.radio(
        "選擇模式",
        mode_options,
        format_func=lambda x: "學派體驗：我當個案，AI 當諮商師" if x == "experience" else "學派實作：我當諮商師，AI 當個案",
    )
    school_ids = list(SCHOOLS)
    school_id = st.selectbox("選擇學派", school_ids, format_func=lambda x: SCHOOLS[x]["name"])
    school = get_school(school_id)
    st.caption(school["core"])
    options = [t["id"] for t in school["techniques"]]
    label = {t["id"]: f"{t['name']}｜{t['short']}" for t in school["techniques"]}
    if mode == "experience":
        selected = list(school["experience_default"])
        st.markdown("**本次由 AI 示範的三項技巧**")
        for item in get_techniques(school_id, selected):
            st.write(f"- {item['name']}：{item['short']}")
    else:
        selected = st.multiselect(
            "從五項技巧中選擇恰好三項",
            options,
            format_func=lambda x: label[x],
            max_selections=3,
        )
        st.caption(f"已選 {len(selected)}／3 項。系統會依這三項技巧建立具有練習機會的案例。")
    theme_id = st.selectbox("練習主題", list(PRACTICE_THEMES), format_func=lambda x: PRACTICE_THEMES[x])
    difficulty = st.select_slider("案例難度", ["初階", "中階", "進階"], value="中階")
    if st.button("開始新的模擬", type="primary", use_container_width=True):
        if len(selected) != 3:
            st.error("開始前必須選擇恰好三項技巧。")
            return
        try:
            with st.spinner("正在建立一致的模擬角色與開場…"):
                start_new_session(mode, school_id, list(selected), PRACTICE_THEMES[theme_id], difficulty)
            st.rerun()
        except Exception as exc:
            st.error(f"無法開始模擬：{exc}")


def continuation_panel() -> None:
    try:
        threads = STORE.list_threads(st.session_state.participant_id)
    except Exception as exc:
        st.error(f"讀取續談資料失敗：{exc}")
        return
    if not threads:
        st.info("目前沒有可續談的晤談歷程。請先完成一次模擬。")
        return
    choices = {str(t["conversation_thread_id"]): t for t in threads}
    selected_thread_id = st.selectbox(
        "選擇要續談的歷程",
        list(choices),
        format_func=lambda x: (
            f"{choices[x].get('school_name')}｜"
            f"{'AI 個案' if choices[x].get('mode') == 'practice' else 'AI 諮商師'}｜"
            f"更新 {choices[x].get('updated_at')}"
        ),
    )
    thread = choices[selected_thread_id]
    school_id = str(thread["school_id"])
    school = get_school(school_id)
    default_ids = parse_json_cell(thread.get("selected_techniques"), school["experience_default"])
    if thread.get("mode") == "experience":
        selected = list(school["experience_default"])
        st.write("續談會載入同一位 AI 諮商師、同一學派與前次工作焦點。")
    else:
        option_ids = [t["id"] for t in school["techniques"]]
        name_map = {t["id"]: f"{t['name']}｜{t['short']}" for t in school["techniques"]}
        valid_defaults = [x for x in default_ids if x in option_ids][:3]
        selected = st.multiselect(
            "本次續談要練習的三項技巧",
            option_ids,
            default=valid_defaults,
            format_func=lambda x: name_map[x],
            max_selections=3,
        )
        st.write("續談會載入同一位 AI 個案、已揭露內容、關係狀態與未完成議題。")
    snapshot = parse_json_cell(thread.get("latest_snapshot"), {})
    if snapshot.get("unfinished_issues"):
        st.caption("前次尚未完成：" + "；".join(snapshot["unfinished_issues"]))
    if st.button("開始續談", type="primary", use_container_width=True):
        if len(selected) != 3:
            st.error("開始前必須選擇恰好三項技巧。")
            return
        try:
            with st.spinner("正在接續上次晤談關係…"):
                start_continuation(thread, list(selected))
            st.rerun()
        except Exception as exc:
            st.error(f"無法開始續談：{exc}")


def render_chat() -> None:
    session = st.session_state.active_session
    st.subheader(f"{session['school_name']}｜{'學派體驗' if session['mode'] == 'experience' else '學生實作'}")
    st.caption("本次技巧：" + "／".join(session["selected_technique_names"]))
    settings = settings_with_defaults()
    target_key = "duration_experience_min" if session["mode"] == "experience" else "duration_practice_min"
    target_minutes = int(settings.get(target_key, "8" if session["mode"] == "experience" else "15") or 0)
    elapsed = datetime.now(ZoneInfo(CONFIG.timezone)) - datetime.fromisoformat(session["started_at"])
    st.caption(f"目前約 {max(0, int(elapsed.total_seconds() // 60))} 分鐘｜建議練習 {target_minutes} 分鐘；由你自行決定何時結束，不強制跳轉。")
    role_labels = {
        "student_client": "你（個案）",
        "student_counselor": "你（諮商師）",
        "ai_client": "AI 模擬個案",
        "ai_counselor": "AI 示範諮商師",
        "system": "系統",
    }
    for turn in st.session_state.turns:
        role = str(turn["speaker_role"])
        with st.chat_message("user" if role.startswith("student") else "assistant"):
            st.caption(role_labels.get(role, role))
            st.write(turn["content_raw"])

    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("可在括弧中輸入非語言訊息，例如（語氣放緩）、（停頓數秒）。")
    with col2:
        if st.button("結束本次晤談", type="primary", use_container_width=True):
            finalize_session()
            st.rerun()

    prompt = st.chat_input("輸入你的回應…", max_chars=CONFIG.max_input_chars)
    if not prompt:
        return
    pii = detect_pii(prompt)
    if pii:
        st.error("內容疑似包含 Email、電話或身分證格式。請刪除可識別資訊後再送出。")
        return
    student_role = "student_counselor" if session["mode"] == "practice" else "student_client"
    student_turn = new_turn(
        session=session,
        turn_index=len(st.session_state.turns) + 1,
        speaker_role=student_role,
        content=prompt,
        timezone=CONFIG.timezone,
    )
    store_turn(student_turn)
    if detect_immediate_risk(prompt):
        message = safety_message()
        store_turn(new_turn(
            session=session,
            turn_index=len(st.session_state.turns) + 1,
            speaker_role="system",
            content=message,
            timezone=CONFIG.timezone,
            error_flag="immediate_risk_stop",
        ))
        safe_store_action("安全事件", STORE.append, "RiskEvents", {
            "risk_event_id": str(uuid.uuid4()),
            "session_id": session["session_id"],
            "participant_id": session["participant_id"],
            "timestamp": STORE.now(),
            "event_type": "immediate_risk_language",
            "action_taken": "simulation_stopped_and_human_help_displayed",
            "content_redacted": redact_for_preview(prompt),
        })
        st.session_state.active_session = finish_session(session, CONFIG.timezone, "safety_stopped")
        if safe_store_action("Session 結束時間", STORE.finish_session, st.session_state.active_session):
            add_completed_usage(st.session_state.active_session)
        st.rerun()
    try:
        with st.spinner("AI 正在回應…"):
            generate_ai_turn(is_opening=False, latest_student_message=prompt)
    except GeminiQuotaError as exc:
        store_turn(new_turn(
            session=session,
            turn_index=len(st.session_state.turns) + 1,
            speaker_role="system",
            content=str(exc),
            timezone=CONFIG.timezone,
            error_flag="gemini_quota_exhausted",
        ))
    except Exception as exc:
        LOGGER.exception("Gemini dialogue request failed")
        store_turn(new_turn(
            session=session,
            turn_index=len(st.session_state.turns) + 1,
            speaker_role="system",
            content="本輪模型暫時無法回應，請稍後再試或結束本次晤談。",
            timezone=CONFIG.timezone,
            error_flag=str(exc)[:300],
        ))
    st.rerun()


def finalize_session() -> None:
    session = finish_session(st.session_state.active_session, CONFIG.timezone, "completed")
    st.session_state.active_session = session
    if safe_store_action("Session 結束時間", STORE.finish_session, session):
        add_completed_usage(session)
    raw = ""
    parsed: dict[str, Any]
    try:
        if session["mode"] == "practice":
            prompt = build_practice_evaluator_prompt(
                session["school_id"], session["selected_techniques"], st.session_state.turns
            )
        else:
            prompt = build_experience_analysis_prompt(
                session["school_id"], session["selected_techniques"], st.session_state.turns
            )
        raw = gemini().generate_text(
            prompt,
            system_instruction="你是形成性教學回饋評量器。只能根據逐字稿證據輸出 JSON。",
            temperature=0.1,
            max_output_tokens=3200,
            response_json=True,
        )
        parsed = parse_json_response(raw)
    except Exception as exc:
        parsed = {
            "total_score": None,
            "strengths": [],
            "improvement_points": [],
            "encouragement": "本次晤談與逐字稿已完整保存；評量服務暫時無法完成，可請教師稍後重新檢視。",
            "limitations": str(exc),
        }

    assessment_id = str(uuid.uuid4())
    record = {
        "assessment_id": assessment_id,
        "session_id": session["session_id"],
        "participant_id": session["participant_id"],
        "mode": session["mode"],
        "school_id": session["school_id"],
        "rubric_version": CONFIG.rubric_version,
        "total_score": parsed.get("total_score", parsed.get("score", "")),
        "dimension_scores": parsed.get("dimensions", {}),
        "skill_events": parsed.get("skill_events", parsed.get("technique_explanations", [])),
        "strengths": parsed.get("strengths", []),
        "improvement_points": parsed.get("improvement_points", []),
        "quoted_examples": parsed.get("alternative_responses", []),
        "next_practice_focus": parsed.get("next_practice_focus", parsed.get("reflection_questions", [])),
        "encouragement": parsed.get("encouragement", ""),
        "raw_model_output": raw,
        "parsed_json": parsed,
        "created_at": STORE.now(),
    }
    st.session_state.assessment = parsed
    st.session_state.raw_assessment = raw
    safe_store_action("形成性回饋", STORE.save_assessment, record)

    try:
        snapshot_raw = gemini().generate_text(
            build_snapshot_prompt(
                session["mode"], session["school_id"], session["selected_techniques"],
                st.session_state.turns, st.session_state.continuation_snapshot,
            ),
            system_instruction="你是續談狀態摘要器，只輸出不含可識別資訊的 JSON。",
            temperature=0.1,
            max_output_tokens=1600,
            response_json=True,
        )
        snapshot = parse_json_response(snapshot_raw)
    except Exception:
        snapshot = {
            "continuation_role": session["continuation_role"],
            "relationship_summary": "本次逐字稿已保存，續談時可由最近對話接續。",
            "disclosed_topics": [],
            "unfinished_issues": [],
            "next_session_focus": [],
        }
    safe_store_action("續談摘要", STORE.save_thread, {
        "conversation_thread_id": session["conversation_thread_id"],
        "participant_id": session["participant_id"],
        "mode": session["mode"],
        "continuation_role": session["continuation_role"],
        "school_id": session["school_id"],
        "school_name": session["school_name"],
        "selected_techniques": session["selected_techniques"],
        "selected_technique_names": session["selected_technique_names"],
        "case_id": session["case_id"],
        "case_data": st.session_state.case_data or {},
        "latest_snapshot": snapshot,
        "last_session_id": session["session_id"],
        "recent_turns": st.session_state.turns[-6:],
        "updated_at": STORE.now(),
        "status": "active",
    })


def render_feedback(settings: dict[str, str]) -> None:
    session = st.session_state.active_session
    assessment = st.session_state.assessment or {}
    st.subheader("本次晤談已完成")
    persistence_warnings = st.session_state.get("persistence_warnings", [])
    if persistence_warnings:
        st.warning(
            "本次晤談已結束，畫面中的逐字稿與回饋仍可下載；"
            "但 Google Sheets 暫時未完成部分同步（"
            + "、".join(persistence_warnings)
            + "）。請保留逐字稿並通知授課教師。"
        )
    else:
        st.success("完整逐字稿、練習時間、學派、技巧與形成性回饋已保存。你之後可選擇續談同一位 AI 對話角色。")
    if as_bool(settings.get("student_feedback_visible"), True):
        if session["mode"] == "practice":
            if as_bool(settings.get("student_score_visible"), True) and assessment.get("total_score") is not None:
                st.metric("AI 形成性分數", f"{assessment.get('total_score')} / 100")
                st.caption("此分數供練習參考，不是標準化測驗結果，也不會覆寫教師人工成績。")
            if assessment.get("encouragement"):
                st.markdown("### 鼓勵與整體回饋")
                st.write(assessment["encouragement"])
            if assessment.get("strengths"):
                st.markdown("### 具體做得好的地方")
                for item in assessment["strengths"]:
                    st.write(f"- {item.get('point', '')}「{item.get('evidence_quote', '')}」{item.get('effect', '')}")
            if assessment.get("improvement_points"):
                st.markdown("### 最值得優先調整")
                for item in assessment["improvement_points"]:
                    st.write(f"- {item.get('point', '')}：{item.get('reason', '')}")
            if assessment.get("alternative_responses"):
                st.markdown("### 可嘗試的替代回應")
                for item in assessment["alternative_responses"]:
                    st.write(f"原句：「{item.get('original_quote', '')}」")
                    st.write(f"可改為：「{item.get('better_response', '')}」— {item.get('why', '')}")
            if assessment.get("next_practice_focus"):
                st.markdown("### 下次練習焦點")
                for item in assessment["next_practice_focus"]:
                    st.write(f"- {item}")
        else:
            st.info("體驗模式不評分學生的自我揭露或『個案表現』。以下只解析 AI 諮商師的示範。")
            if assessment.get("technique_explanations"):
                for item in assessment["technique_explanations"]:
                    name = next((t["name"] for t in get_school(session["school_id"])["techniques"] if t["id"] == item.get("technique_id")), item.get("technique_id", "技巧"))
                    with st.expander(name):
                        st.write(f"AI 原句：「{item.get('ai_quote', '')}」")
                        st.write(f"使用理由：{item.get('why_used', '')}")
                        st.write(f"可能效果：{item.get('possible_effect', '')}")
            if assessment.get("overall_learning"):
                st.write(assessment["overall_learning"])
            if assessment.get("encouragement"):
                st.write(assessment["encouragement"])
    else:
        st.info("教師目前設定為不向學生顯示 AI 回饋；本次資料仍已保存供教師檢視。")

    transcript = make_transcript_txt(session, st.session_state.turns)
    st.download_button(
        "下載本次晤談逐字稿 TXT（選配）",
        transcript,
        file_name=safe_filename(session["session_id"]),
        mime="text/plain",
        use_container_width=True,
    )
    if st.button("回到練習首頁", use_container_width=True):
        st.session_state.active_session = None
        st.session_state.turns = []
        st.session_state.case_data = None
        st.session_state.continuation_snapshot = None
        st.session_state.prior_turns_context = []
        st.session_state.assessment = None
        st.rerun()


def student_page() -> None:
    render_header()
    settings = settings_with_defaults()
    error = student_access_error(settings)
    if error and not is_teacher(st.session_state.email, CONFIG.teacher_emails):
        st.error(error)
        return
    render_usage_progress()
    if not st.session_state.api_validated or not st.session_state.api_key:
        api_key_gate()
        return
    if st.session_state.active_session:
        if st.session_state.active_session.get("completion_status") == "in_progress":
            render_chat()
        else:
            render_feedback(settings)
        return
    tab1, tab2 = st.tabs(["開始新模擬", "續談上次歷程"])
    with tab1:
        new_practice_panel(settings)
    with tab2:
        continuation_panel()


def export_research_zip() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for sheet in SCHEMAS:
            rows = STORE.all_records(sheet)
            frame = pd.DataFrame(rows, columns=SCHEMAS[sheet])
            zf.writestr(f"{sheet}.csv", frame.to_csv(index=False).encode("utf-8-sig"))
    return output.getvalue()


def teacher_settings_panel(settings: dict[str, str]) -> None:
    st.subheader("教學開放設定")
    with st.form("teacher_settings"):
        enabled = st.checkbox("開放學生使用", value=as_bool(settings.get("system_enabled"), True))
        open_start = st.text_input("開放時間（ISO，可留白）", settings.get("open_start", ""), placeholder="2026-09-15T08:00:00+08:00")
        open_end = st.text_input("關閉時間（ISO，可留白）", settings.get("open_end", ""), placeholder="2027-01-15T23:59:00+08:00")
        max_sessions = st.number_input("每位學生最多 Session 數（0 表示不限）", min_value=0, value=int(settings.get("max_sessions_per_student", "0") or 0))
        duration_experience = st.number_input("體驗模式建議分鐘數", min_value=1, max_value=60, value=int(settings.get("duration_experience_min", "8") or 8))
        duration_practice = st.number_input("實作模式建議分鐘數", min_value=1, max_value=60, value=int(settings.get("duration_practice_min", "15") or 15))
        modes = st.multiselect(
            "開放模式", ["experience", "practice"],
            default=[x.strip() for x in settings.get("allowed_modes", "experience,practice").split(",") if x.strip()],
            format_func=lambda x: "學派體驗" if x == "experience" else "學生實作",
        )
        feedback = st.checkbox("學生可看形成性回饋", value=as_bool(settings.get("student_feedback_visible"), True))
        score = st.checkbox("學生可看 AI 形成性分數", value=as_bool(settings.get("student_score_visible"), True))
        if st.form_submit_button("儲存設定"):
            updates = {
                "system_enabled": str(enabled).lower(),
                "open_start": open_start.strip(),
                "open_end": open_end.strip(),
                "max_sessions_per_student": max_sessions,
                "duration_experience_min": duration_experience,
                "duration_practice_min": duration_practice,
                "allowed_modes": ",".join(modes),
                "student_feedback_visible": str(feedback).lower(),
                "student_score_visible": str(score).lower(),
            }
            try:
                if open_start:
                    parse_local_datetime(open_start)
                if open_end:
                    parse_local_datetime(open_end)
                for key, value in updates.items():
                    STORE.save_setting(key, value, st.session_state.email)
                st.session_state.settings_cache = None
                st.session_state.settings_cache_at = 0.0
                st.success("設定已儲存。")
            except Exception as exc:
                st.error(f"設定未儲存：{exc}")


def teacher_dashboard() -> None:
    render_header()
    if not is_teacher(st.session_state.email, CONFIG.teacher_emails):
        st.error("此帳號沒有教師後台權限。")
        return
    if st.session_state.store_mode == "memory":
        st.warning("目前是本機暫存模式，無法查看其他學生資料；請完成 Google Sheets 設定。")
    settings = settings_with_defaults()
    tab1, tab2, tab3 = st.tabs(["學生進度與逐字稿", "開放設定", "研究資料匯出"])
    with tab1:
        sessions = STORE.all_records("Sessions")
        identities = STORE.all_records("IdentityMap")
        if not sessions:
            st.info("目前尚無 Session 資料。")
        else:
            sdf = pd.DataFrame(sessions)
            idf = pd.DataFrame(identities)[["participant_id", "email"]] if identities else pd.DataFrame(columns=["participant_id", "email"])
            merged = sdf.merge(idf, on="participant_id", how="left")
            completed = merged[merged["completion_status"].isin(["completed", "safety_stopped"])]
            c1, c2, c3 = st.columns(3)
            c1.metric("學生人數", int(merged["participant_id"].nunique()))
            c2.metric("Session 數", len(merged))
            durations = pd.to_numeric(merged.get("duration_seconds", pd.Series(dtype=float)), errors="coerce").fillna(0)
            c3.metric("累計練習分鐘", f"{durations.sum()/60:.1f}")
            email_options = ["全部"] + sorted(str(x) for x in merged["email"].dropna().unique())
            selected_email = st.selectbox("依學校 Email 篩選", email_options)
            shown = completed if selected_email == "全部" else completed[completed["email"] == selected_email]
            columns = [c for c in ["email", "started_at", "mode", "school_id", "selected_technique_names", "duration_seconds", "completion_status", "session_id"] if c in shown.columns]
            st.dataframe(shown[columns], use_container_width=True, hide_index=True)
            if not shown.empty:
                session_ids = list(shown["session_id"].astype(str))
                chosen = st.selectbox("查看單次 Session", session_ids, format_func=lambda x: f"{x[:8]}…")
                row = shown[shown["session_id"].astype(str) == chosen].iloc[0].to_dict()
                st.markdown(f"**學生：** {row.get('email', '')}　 **學派：** {row.get('school_id', '')}　 **模式：** {row.get('mode', '')}")
                turns = STORE.session_turns(chosen)
                for turn in turns:
                    st.write(f"**[{turn.get('turn_index')}] {turn.get('speaker_role')}：** {turn.get('content_raw')}")
                assessment = STORE.get_assessment(chosen)
                if assessment:
                    st.markdown("### AI 原始形成性回饋")
                    parsed = parse_json_cell(assessment.get("parsed_json"), {})
                    st.json(parsed, expanded=False)
                st.markdown("### 教師人工成績與評語")
                with st.form(f"grade-{chosen}"):
                    use_score = st.checkbox("本次填寫教師分數")
                    teacher_score = st.number_input("教師分數", min_value=0, max_value=100, value=80, disabled=not use_score)
                    teacher_comment = st.text_area("教師評語")
                    if st.form_submit_button("另存教師評量"):
                        STORE.add_teacher_grade(
                            chosen, str(row.get("participant_id")), st.session_state.email,
                            int(teacher_score) if use_score else None, teacher_comment,
                        )
                        st.success("教師評量已另存，不會覆寫 AI 原始結果。")
    with tab2:
        teacher_settings_panel(settings)
    with tab3:
        st.write("匯出包含 Sessions、ChatLogs、Threads、Assessments、SkillEvents、TeacherGrades、Settings 與 RiskEvents。IdentityMap 含 Email，研究去識別化時應單獨保管或移除。")
        try:
            payload = export_research_zip()
            st.download_button(
                "下載完整後台 CSV 壓縮檔",
                payload,
                file_name=f"theory_agent_research_export_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                mime="application/zip",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"資料匯出失敗：{exc}")


if not st.session_state.authenticated:
    login_page()
else:
    sidebar()
    if st.session_state.view == "teacher":
        teacher_dashboard()
    else:
        student_page()
