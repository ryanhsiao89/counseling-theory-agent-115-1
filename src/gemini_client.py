"""Gemini 呼叫封裝；學生 API Key 只留在記憶體，不寫入任何資料表。"""

from __future__ import annotations

import json
import math
import re
import time
from typing import Any

from google import genai
from google.genai import types


class GeminiQuotaError(RuntimeError):
    """The student's Google AI project has reached a request/token quota."""


class GeminiAuthenticationError(RuntimeError):
    """The supplied API key is invalid or not authorized."""


def _retry_after_seconds(message: str) -> int | None:
    match = re.search(r"retry(?:\s+in|delay['\":\s]+)\s*([0-9]+(?:\.[0-9]+)?)s", message, re.I)
    return math.ceil(float(match.group(1))) if match else None


def _friendly_api_exception(exc: Exception, model_name: str) -> RuntimeError | None:
    message = str(exc)
    lowered = message.lower()
    if any(token in lowered for token in ("429", "resource_exhausted", "quota exceeded")):
        retry_after = _retry_after_seconds(message)
        advice = f"Google 建議約 {retry_after} 秒後再試。" if retry_after else "請稍後再試。"
        if "perday" in lowered or "free_tier_requests" in lowered:
            advice += "若仍無法使用，表示當日免費額度尚未重置。"
        return GeminiQuotaError(
            f"API Key 有效，但其 Google AI 專案的 {model_name} 免費額度已達上限。"
            f"{advice}額度依 Google AI 專案計算，不是重新產生同一專案的 Key 就會歸零。"
        )
    if any(token in lowered for token in ("api_key_invalid", "api key not valid", "invalid api key")):
        return GeminiAuthenticationError(
            "Gemini API Key 無效。請回到 Google AI Studio 複製完整 Key，確認前後沒有空格後再貼上。"
        )
    return None


def _finish_reason(response: Any) -> str:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""
    reason = getattr(candidates[0], "finish_reason", None)
    value = getattr(reason, "value", reason)
    return str(value or "").upper()


def _looks_incomplete_text(text: str) -> bool:
    """Catch obvious cut-offs even if an SDK omits finish_reason."""
    value = (text or "").strip()
    if not value:
        return True
    if value.count("（") > value.count("）") or value.count("「") > value.count("」"):
        return True
    suspicious_suffixes = (
        "，", "、", "：", "；", "卻", "但", "但是", "可是", "而且", "因為",
        "所以", "如果", "雖然", "只是", "以及", "或者", "或是", "例如",
    )
    return value.endswith(suspicious_suffixes)


class GeminiService:
    def __init__(self, api_key: str, model_name: str):
        if not api_key or not api_key.strip():
            raise ValueError("Gemini API Key 不可空白。")
        self.client = genai.Client(api_key=api_key.strip())
        self.model_name = model_name

    def generate_text(
        self,
        prompt: str,
        *,
        system_instruction: str | None = None,
        temperature: float = 0.4,
        max_output_tokens: int = 1200,
        response_json: bool = False,
        thinking_level: str = "low",
        attempts: int = 3,
    ) -> str:
        last_error: Exception | None = None
        output_limit = max_output_tokens
        for attempt in range(attempts):
            config_values: dict[str, Any] = dict(
                system_instruction=system_instruction,
                temperature=temperature,
                max_output_tokens=output_limit,
                response_mime_type="application/json" if response_json else "text/plain",
            )
            # Gemini 3 uses dynamic thinking. Explicit low thinking leaves more
            # of the output allowance available for the visible dialogue.
            if self.model_name.startswith("gemini-3"):
                config_values["thinking_config"] = {"thinking_level": thinking_level}
            config = types.GenerateContentConfig(**config_values)
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                text = (response.text or "").strip()
            except Exception as exc:  # SDK 的錯誤型別會隨版本調整，統一在此重試
                friendly = _friendly_api_exception(exc, self.model_name)
                if friendly is not None:
                    raise friendly from exc
                last_error = exc
                message = str(exc).lower()
                retryable = any(token in message for token in ("timeout", "503"))
                if not retryable or attempt == attempts - 1:
                    break
                time.sleep(1.5 * (2**attempt))
                continue

            reason = _finish_reason(response)
            truncated = (
                reason == "MAX_TOKENS"
                or not text
                or (not response_json and _looks_incomplete_text(text))
            )
            if truncated:
                last_error = RuntimeError("Gemini 回覆因輸出額度而未完整結束。")
                if attempt < attempts - 1:
                    # Retry the whole response with a larger allowance instead
                    # of joining a potentially duplicated continuation.
                    output_limit = min(max(output_limit * 2, 2048), 8192)
                    continue
                break
            if reason not in {"", "STOP", "FINISH_REASON_UNSPECIFIED"}:
                last_error = RuntimeError(f"Gemini 未正常完成回覆（{reason}）。")
                break
            return text
        raise RuntimeError(f"Gemini 呼叫失敗：{last_error}") from last_error

    def validate_key(self) -> None:
        # Listing models validates authentication without consuming one of the
        # student's generate_content requests.
        try:
            available = {
                str(getattr(model, "name", "")).removeprefix("models/")
                for model in self.client.models.list()
            }
        except Exception as exc:
            friendly = _friendly_api_exception(exc, self.model_name)
            if friendly is not None:
                raise friendly from exc
            raise RuntimeError("目前無法驗證 Gemini API Key，請稍後再試。") from exc
        if not any(
            name == self.model_name or name.startswith(f"{self.model_name}-")
            for name in available
        ):
            raise RuntimeError(
                f"API Key 可以連線，但目前無法使用課程指定模型 {self.model_name}。請通知授課教師。"
            )


def parse_json_response(raw: str) -> dict[str, Any]:
    value = (raw or "").strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
    value = re.sub(r"\s*```$", "", value)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("模型輸出不是 JSON 物件。")
    return parsed
