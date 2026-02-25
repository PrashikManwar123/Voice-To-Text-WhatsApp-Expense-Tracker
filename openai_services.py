import json
import os
from datetime import date
from io import BytesIO
from typing import Any

from openai import AsyncOpenAI


class ExpenseExtractionError(Exception):
    """Raised when transcription or structured extraction fails."""


def _build_llm_client() -> AsyncOpenAI:
    """
    Build an OpenAI-compatible client.

    Priority:
    1) GROQ_API_KEY (recommended for this project)
    2) OPENAI_API_KEY (fallback)
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if groq_api_key:
        return AsyncOpenAI(
            api_key=groq_api_key,
            base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
        )

    if openai_api_key:
        return AsyncOpenAI(api_key=openai_api_key)

    raise ExpenseExtractionError(
        "No LLM API key configured. Set GROQ_API_KEY (recommended) or OPENAI_API_KEY."
    )


async def transcribe_audio(audio_bytes: bytes, filename: str, mime_type: str) -> str:
    """Transcribe raw audio bytes using an OpenAI-compatible transcription endpoint."""
    if not audio_bytes:
        raise ExpenseExtractionError("Received empty audio file.")

    client = _build_llm_client()

    default_model = "whisper-large-v3" if os.getenv("GROQ_API_KEY") else "whisper-1"
    transcription_model = os.getenv("TRANSCRIBE_MODEL", default_model)

    audio_file = BytesIO(audio_bytes)
    audio_file.name = filename

    try:
        transcript = await client.audio.transcriptions.create(
            model=transcription_model,
            file=(filename, audio_file.read(), mime_type),
        )
    except Exception as exc:
        raise ExpenseExtractionError(f"Audio transcription failed: {exc}") from exc

    text = (getattr(transcript, "text", "") or "").strip()
    if not text:
        raise ExpenseExtractionError("Could not transcribe voice note into text.")

    return text


def _validate_expense_payload(payload: Any) -> dict[str, Any]:
    """Validate and normalize required keys from model output."""
    if not isinstance(payload, dict):
        raise ExpenseExtractionError("Model response is not a JSON object.")

    required_keys = ["amount", "category", "date", "description"]
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ExpenseExtractionError(f"Missing fields in model output: {', '.join(missing)}")

    try:
        amount = float(payload["amount"])
    except (TypeError, ValueError) as exc:
        raise ExpenseExtractionError("Amount must be a valid number.") from exc

    normalized = {
        "amount": int(amount) if amount.is_integer() else round(amount, 2),
        "category": str(payload["category"]).strip() or "Misc",
        "date": str(payload["date"]).strip(),
        "description": str(payload["description"]).strip() or "Expense entry",
    }

    # Basic format check. The prompt also enforces this and model usually complies.
    if len(normalized["date"]) != 10:
        raise ExpenseExtractionError("Date must be in YYYY-MM-DD format.")

    return normalized


async def extract_expense_json(raw_text: str) -> dict[str, Any]:
    """Extract a strict expense JSON object from free-form input text."""
    if not raw_text.strip():
        raise ExpenseExtractionError("Cannot extract expense from empty text.")

    client = _build_llm_client()

    default_extraction_model = (
        "llama-3.3-70b-versatile" if os.getenv("GROQ_API_KEY") else "gpt-4o-mini"
    )
    extraction_model = os.getenv("EXTRACTION_MODEL", default_extraction_model)

    today = date.today().isoformat()
    system_prompt = (
        "You extract structured expense entries from user messages. "
        "Return ONLY a valid JSON object with keys: amount, category, date, description. "
        "Rules: amount must be numeric. category should be concise (Food, Travel, Utilities, Misc, etc). "
        f"date must always be YYYY-MM-DD. Interpret relative dates using today={today}. "
        "If no date is provided, use today's date. description should be short and clear. "
        "Do not include markdown, commentary, or extra keys."
    )

    try:
        completion = await client.chat.completions.create(
            model=extraction_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_text},
            ],
        )
        content = (completion.choices[0].message.content or "").strip()
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ExpenseExtractionError("Model returned invalid JSON.") from exc
    except Exception as exc:
        raise ExpenseExtractionError(f"Failed to extract expense details: {exc}") from exc

    return _validate_expense_payload(payload)
