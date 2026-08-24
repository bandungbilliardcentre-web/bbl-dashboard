"""Cached, credential-safe Google Sheets access for Streamlit."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials

from .schema import SHEET_NAMES

LOGGER = logging.getLogger("bbl_dashboard.google_sheets")
SCOPES = (
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
)


class DashboardDataError(RuntimeError):
    """A sanitized dashboard data-source error."""


@dataclass(frozen=True)
class LoadResult:
    sheets: dict[str, pd.DataFrame]
    loaded_at: datetime
    source: str


def _frame(values: list[list[Any]]) -> pd.DataFrame:
    if not values:
        return pd.DataFrame()
    headers = [str(value).strip() for value in values[0]]
    width = len(headers)
    rows = [(row + [""] * width)[:width] for row in values[1:]]
    return pd.DataFrame(rows, columns=headers)


class GoogleSheetsReader:
    def __init__(self, spreadsheet_id: str, service_account_info: dict[str, str]):
        credentials = Credentials.from_service_account_info(service_account_info, scopes=SCOPES)
        self._spreadsheet = gspread.authorize(credentials).open_by_key(spreadsheet_id)

    def load_sheet(self, sheet_name: str) -> pd.DataFrame:
        if sheet_name not in SHEET_NAMES:
            raise ValueError(f"Unsupported sheet: {sheet_name}")
        # Apps Script persists money as numeric cells and applies a display-only
        # Rupiah number format. Request raw numbers while keeping dates formatted,
        # otherwise gspread returns strings such as "Rp 150.000".
        values = self._spreadsheet.worksheet(sheet_name).get_all_values(
            value_render_option="UNFORMATTED_VALUE",
            date_time_render_option="FORMATTED_STRING",
        )
        return _frame(values)


class PublicCsvReader:
    def __init__(self, url_template: str):
        if "{sheet_name}" not in url_template:
            raise DashboardDataError("published_csv_url_template must contain {sheet_name}")
        self._url_template = url_template

    def load_sheet(self, sheet_name: str) -> pd.DataFrame:
        url = self._url_template.format(sheet_name=quote(sheet_name, safe=""))
        request = Request(url, headers={"User-Agent": "BBL-Business-Dashboard/1.0"})
        with urlopen(request, timeout=30) as response:  # noqa: S310 - explicitly configured public fallback URL
            body = response.read().decode("utf-8-sig")
        return pd.read_csv(StringIO(body)) if body.strip() else pd.DataFrame()


def _settings() -> tuple[str, dict[str, str] | None, str]:
    try:
        sheets = dict(st.secrets.get("google_sheets", {}))
        credentials = dict(st.secrets.get("gcp_service_account", {}))
    except (FileNotFoundError, KeyError):
        sheets, credentials = {}, {}
    spreadsheet_id = str(sheets.get("spreadsheet_id", "")).strip()
    public_template = str(sheets.get("published_csv_url_template", "")).strip()
    return spreadsheet_id, credentials or None, public_template


def _reader():
    spreadsheet_id, credentials, public_template = _settings()
    if spreadsheet_id and credentials:
        return GoogleSheetsReader(spreadsheet_id, credentials), "Google Sheets API (service account)"
    if public_template:
        return PublicCsvReader(public_template), "Published CSV fallback"
    raise DashboardDataError(
        "Google Sheets is not configured. Create dashboard/.streamlit/secrets.toml from secrets.toml.example."
    )


@st.cache_data(ttl=60, show_spinner=False)
def load_all_sheets() -> LoadResult:
    """Fetch each reporting sheet once per cached refresh cycle."""
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            reader, source = _reader()
            sheets = {name: reader.load_sheet(name) for name in SHEET_NAMES}
            return LoadResult(sheets=sheets, loaded_at=datetime.now().astimezone(), source=source)
        except Exception as error:  # credentials/network errors must not reach the cashier-facing UI
            last_error = error
            LOGGER.exception("Google Sheets dashboard load attempt %s failed", attempt)
            if attempt < 3:
                time.sleep(attempt)
    raise DashboardDataError("Unable to load Google Sheets data after three attempts.") from last_error


def clear_sheet_cache() -> None:
    load_all_sheets.clear()
