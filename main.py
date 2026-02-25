import logging
import os
from typing import Optional
from xml.sax.saxutils import escape

import httpx
from fastapi import FastAPI, Form, Request
from fastapi.responses import PlainTextResponse

from openai_services import ExpenseExtractionError, extract_expense_json, transcribe_audio
from sheets_service import SheetsServiceError, append_expense_row

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("expense-tracker")

app = FastAPI(title="WhatsApp Voice-to-Sheet Expense Tracker")


class WebhookProcessingError(Exception):
    """Raised when webhook processing cannot continue."""


async def download_twilio_media(media_url: str) -> tuple[bytes, str]:
    """Download media content from Twilio using account credentials."""
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    if not account_sid or not auth_token:
        raise WebhookProcessingError(
            "Twilio credentials are required to download media files."
        )

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(media_url, auth=(account_sid, auth_token))

    if response.status_code >= 400:
        raise WebhookProcessingError(
            f"Failed to download media from Twilio (status={response.status_code})."
        )

    return response.content, response.headers.get("Content-Type", "audio/ogg")


def build_twiml(message: str) -> str:
    """Build a safe TwiML XML response."""
    escaped_message = escape(message)
    return f"<?xml version=\"1.0\" encoding=\"UTF-8\"?><Response><Message>{escaped_message}</Message></Response>"


@app.get("/health", response_class=PlainTextResponse)
async def health() -> str:
    return "ok"


@app.post("/whatsapp", response_class=PlainTextResponse)
async def whatsapp_webhook(
    request: Request,
    body: str = Form(default=""),
    num_media: int = Form(default=0, alias="NumMedia"),
    media_url_0: Optional[str] = Form(default=None, alias="MediaUrl0"),
    media_content_type_0: Optional[str] = Form(default=None, alias="MediaContentType0"),
) -> PlainTextResponse:
    """
    Receive Twilio WhatsApp webhooks and log expenses to Google Sheets.

    Supports both text messages and voice notes.
    """
    try:
        incoming_text = (body or "").strip()

        if num_media > 0 and media_url_0:
            if not (media_content_type_0 or "").startswith("audio/"):
                raise WebhookProcessingError(
                    "Received media is not an audio file. Please send a voice note or text expense."
                )

            logger.info("Audio message received. Downloading and transcribing...")
            audio_bytes, mime_type = await download_twilio_media(media_url_0)
            incoming_text = await transcribe_audio(
                audio_bytes=audio_bytes,
                filename="voice-note.ogg",
                mime_type=mime_type,
            )

        if not incoming_text:
            raise WebhookProcessingError(
                "Message is empty. Please send expense details as text or voice note."
            )

        logger.info("Extracting expense fields from text: %s", incoming_text)
        expense = await extract_expense_json(incoming_text)

        logger.info("Appending expense to Google Sheets")
        append_expense_row(expense)

        confirmation = (
            f"✅ Logged: ₹{expense['amount']} for {expense['category']} on "
            f"{expense['date']} ({expense['description']})"
        )
        return PlainTextResponse(build_twiml(confirmation), media_type="application/xml")

    except (WebhookProcessingError, ExpenseExtractionError, SheetsServiceError) as exc:
        logger.exception("Failed processing webhook: %s", exc)
        message = f"⚠️ Could not log expense: {exc}"
        return PlainTextResponse(build_twiml(message), media_type="application/xml", status_code=200)
    except Exception as exc:  # catch-all for unexpected issues
        logger.exception("Unexpected error: %s", exc)
        message = "⚠️ Internal error while processing your expense. Please try again."
        return PlainTextResponse(build_twiml(message), media_type="application/xml", status_code=200)
