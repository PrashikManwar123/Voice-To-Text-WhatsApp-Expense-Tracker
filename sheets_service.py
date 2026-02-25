import json
import os
from typing import Any

import gspread
from google.oauth2.service_account import Credentials


class SheetsServiceError(Exception):
    """Raised for Google Sheets write failures."""


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]


def _build_credentials() -> Credentials:
    credentials_file = os.getenv("GOOGLE_SHEETS_CREDENTIALS_FILE")
    credentials_json = os.getenv("GOOGLE_SHEETS_CREDENTIALS_JSON")

    if credentials_file:
        return Credentials.from_service_account_file(credentials_file, scopes=SCOPES)

    if credentials_json:
        try:
            payload = json.loads(credentials_json)
            return Credentials.from_service_account_info(payload, scopes=SCOPES)
        except json.JSONDecodeError as exc:
            raise SheetsServiceError("GOOGLE_SHEETS_CREDENTIALS_JSON is not valid JSON.") from exc

    raise SheetsServiceError(
        "Google credentials missing. Set GOOGLE_SHEETS_CREDENTIALS_FILE or GOOGLE_SHEETS_CREDENTIALS_JSON."
    )


def append_expense_row(expense: dict[str, Any]) -> None:
    """Append an expense object to the configured Google Sheet worksheet."""
    sheet_id = os.getenv("GOOGLE_SHEET_ID")
    worksheet_name = os.getenv("GOOGLE_SHEET_WORKSHEET", "Sheet1")

    if not sheet_id:
        raise SheetsServiceError("GOOGLE_SHEET_ID is not configured.")

    try:
        credentials = _build_credentials()
        client = gspread.authorize(credentials)
        worksheet = client.open_by_key(sheet_id).worksheet(worksheet_name)

        row = [
            expense.get("date"),
            expense.get("amount"),
            expense.get("category"),
            expense.get("description"),
        ]
        worksheet.append_row(row, value_input_option="USER_ENTERED")
    except SheetsServiceError:
        raise
    except Exception as exc:
        raise SheetsServiceError(f"Failed to append expense to Google Sheets: {exc}") from exc
