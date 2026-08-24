"""Normalize the current Google Sheets archive into reusable analytics frames."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, replace
from datetime import datetime
from numbers import Number
from typing import Any, Mapping

import pandas as pd

from .schema import (
    PAID_STATUSES,
    SHEET_COLUMN_MAPS,
    SHEET_COLUMNS,
    SHEET_FINANCIAL_COLUMNS,
    SHEET_NAMES,
    TRANSACTION_TYPES,
)


LOGGER = logging.getLogger("bbl_dashboard.transforms")


@dataclass(frozen=True)
class DataHealth:
    raw_transactions: int = 0
    raw_sessions: int = 0
    raw_fnb_items: int = 0
    normalized_transactions: int = 0
    normalized_sessions: int = 0
    normalized_fnb_items: int = 0
    duplicate_invoices: int = 0
    duplicate_sessions: int = 0
    invalid_business_dates: int = 0
    transactions_missing_paid_at: int = 0
    transactions_invalid_paid_at: int = 0
    transactions_missing_revenue_date: int = 0
    transactions_revenue_date_mismatches: int = 0
    excluded_unpaid_or_invalid: int = 0
    sessions_without_invoice: int = 0
    session_paid_at_mismatches: int = 0
    session_revenue_date_mismatches: int = 0
    session_business_date_mismatches: int = 0
    fnb_without_invoice: int = 0
    fnb_paid_at_mismatches: int = 0
    fnb_revenue_date_mismatches: int = 0
    fnb_business_date_mismatches: int = 0
    fnb_total_mismatches: int = 0
    session_total_mismatches: int = 0
    transactions_actual_headers: tuple[str, ...] = ()
    transactions_mapped_headers: tuple[str, ...] = ()
    grand_total_non_null_count: int = 0
    grand_total_numeric_count: int = 0
    grand_total_sum: float = 0.0
    billiard_subtotal_sum: float = 0.0
    fnb_subtotal_sum: float = 0.0
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class DashboardData:
    transactions: pd.DataFrame
    sessions: pd.DataFrame
    fnb_items: pd.DataFrame
    daily_summary: pd.DataFrame
    health: DataHealth


def empty_raw_sheets() -> dict[str, pd.DataFrame]:
    return {name: pd.DataFrame(columns=SHEET_COLUMNS[name]) for name in SHEET_NAMES}


def _raw(raw_sheets: Mapping[str, pd.DataFrame], sheet: str) -> pd.DataFrame:
    value = raw_sheets.get(sheet)
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _series(frame: pd.DataFrame, name: str, default: Any = pd.NA) -> pd.Series:
    if name in frame.columns:
        return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def _text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def _non_blank_count(series: pd.Series) -> int:
    return int(_text(series).ne("").sum())


def normalize_currency(value: Any) -> float | None:
    """Parse numeric or commonly formatted Rupiah cells without losing decimals.

    A lone separator followed by three digits is a thousands separator when the
    leading group contains at most three digits (for example ``150.000``). One or
    two decimal digits are preserved, as are numeric float cells from Sheets.
    """
    if value is None or value is pd.NA:
        return None
    if isinstance(value, Number) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    text = str(value).replace("\u00a0", " ").strip()
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()
    text = re.sub(r"(?i)^rp\.?\s*", "", text).replace(" ", "")
    sign = -1.0 if text.startswith("-") else 1.0
    if text[:1] in ("+", "-"):
        text = text[1:]
    if negative:
        sign = -1.0
    if not text or not re.fullmatch(r"[0-9.,]+", text):
        return None

    comma_count = text.count(",")
    dot_count = text.count(".")
    normalized: str
    if comma_count and dot_count:
        # In mixed-locale values, the last separator is the decimal mark and the
        # other separator is grouping (1,234.50 or 1.234,50).
        decimal_separator = "," if text.rfind(",") > text.rfind(".") else "."
        integer_part, fractional_part = text.rsplit(decimal_separator, 1)
        if not fractional_part:
            return None
        integer_digits = integer_part.replace(",", "").replace(".", "")
        if not integer_digits.isdigit() or not fractional_part.isdigit():
            return None
        normalized = f"{integer_digits}.{fractional_part}"
    elif comma_count or dot_count:
        separator = "," if comma_count else "."
        groups = text.split(separator)
        if any(not group.isdigit() or not group for group in groups):
            return None
        if len(groups) > 2:
            if len(groups[0]) <= 3 and all(len(group) == 3 for group in groups[1:]):
                normalized = "".join(groups)
            else:
                return None
        else:
            whole, trailing = groups
            if len(trailing) == 3 and len(whole) <= 3 and whole != "0":
                normalized = whole + trailing
            else:
                normalized = f"{whole}.{trailing}"
    else:
        normalized = text

    try:
        parsed = float(normalized) * sign
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _currency(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.map(normalize_currency), errors="coerce").fillna(0.0)


def _number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def _integer(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def _date(series: pd.Series) -> pd.Series:
    try:
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(series, errors="coerce")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_convert("Asia/Jakarta").dt.tz_localize(None)
    return parsed.dt.normalize()


def _local_datetime(value: Any) -> pd.Timestamp:
    if value is None or value is pd.NA or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT
    text = str(value).strip()
    if not text:
        return pd.NaT
    try:
        parsed = pd.to_datetime(text, errors="coerce")
    except (TypeError, ValueError, OverflowError):
        return pd.NaT
    if pd.isna(parsed):
        return pd.NaT
    timestamp = pd.Timestamp(parsed)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("Asia/Jakarta").tz_localize(None)
    return timestamp


def _datetimes(series: pd.Series) -> pd.Series:
    return series.map(_local_datetime).astype("datetime64[ns]")


def _valid_table(series: pd.Series) -> pd.Series:
    values = _integer(series)
    return values.where(values.between(1, 12))


def _mismatch_count(left: pd.Series, right: pd.Series) -> int:
    """Count unequal values while treating two missing values as equal."""
    equal = left.eq(right) | (left.isna() & right.isna())
    return int((~equal.fillna(False)).sum())


def normalize_transactions(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], list[str]]:
    warnings: list[str] = []
    frame = pd.DataFrame(index=raw.index)
    frame["source_row"] = range(2, len(raw) + 2)
    frame["timestamp"] = _datetimes(_series(raw, "Timestamp"))
    frame["business_date"] = _date(_series(raw, "Business Date"))
    frame["stored_revenue_date"] = _date(_series(raw, "Revenue Date"))
    frame["invoice_number"] = _text(_series(raw, "Invoice Number"))
    frame["transaction_type"] = _text(_series(raw, "Transaction Type")).str.upper()
    frame["table_number"] = _valid_table(_series(raw, "Table Number"))
    frame["cashier"] = _text(_series(raw, "Cashier"))
    frame["payment_method"] = _text(_series(raw, "Payment Method"))
    frame["billiard_revenue"] = _currency(_series(raw, "Billiard Subtotal"))
    frame["fnb_revenue"] = _currency(_series(raw, "F&B Subtotal"))
    frame["subtotal"] = _currency(_series(raw, "Subtotal"))
    frame["discount_name"] = _text(_series(raw, "Discount Name"))
    frame["discount_percent"] = _number(_series(raw, "Discount Percent"))
    frame["discount_amount"] = _currency(_series(raw, "Discount Amount"))
    frame["grand_total"] = _currency(_series(raw, "Grand Total"))
    frame["paid_at"] = _datetimes(_series(raw, "Paid At"))
    frame["revenue_date"] = frame["paid_at"].dt.normalize()
    frame["reporting_date"] = frame["revenue_date"]
    frame["synced_at"] = _datetimes(_series(raw, "Synced At"))

    status_column = "Payment Status" if "Payment Status" in raw.columns else "Status" if "Status" in raw.columns else None
    frame["payment_status"] = _text(_series(raw, status_column, "PAID")) if status_column else "PAID"
    frame["payment_status"] = frame["payment_status"].replace("", "PAID").str.upper()

    invalid_dates = int(frame["business_date"].isna().sum())
    raw_paid_at = _text(_series(raw, "Paid At"))
    raw_revenue_date = _text(_series(raw, "Revenue Date"))
    missing_paid_at = int(raw_paid_at.eq("").sum())
    invalid_paid_at = int((raw_paid_at.ne("") & frame["paid_at"].isna()).sum())
    missing_revenue_date = int(raw_revenue_date.eq("").sum())
    revenue_mismatches = int((
        frame["paid_at"].notna()
        & frame["stored_revenue_date"].notna()
        & frame["stored_revenue_date"].ne(frame["revenue_date"])
    ).sum())
    before = len(frame)
    valid = (
        frame["invoice_number"].ne("")
        & frame["paid_at"].notna()
        & frame["transaction_type"].isin(TRANSACTION_TYPES)
        & frame["payment_status"].isin(PAID_STATUSES)
    )
    frame = frame.loc[valid].copy()
    excluded = before - len(frame)

    duplicate_mask = frame.duplicated("invoice_number", keep=False)
    duplicate_invoices = int(frame.loc[duplicate_mask, "invoice_number"].nunique())
    if duplicate_invoices:
        warnings.append(f"{duplicate_invoices} duplicate canonical invoice(s) were deduplicated by Invoice Number.")
    frame = (
        frame.sort_values(["invoice_number", "synced_at", "timestamp", "source_row"], na_position="first", kind="stable")
        .drop_duplicates("invoice_number", keep="last")
        .sort_values(["revenue_date", "paid_at", "invoice_number"], kind="stable")
        .reset_index(drop=True)
    )
    standalone = frame["transaction_type"].eq("STANDALONE_FNB")
    frame.loc[standalone, "table_number"] = pd.NA
    return frame, {
        "invalid_business_dates": invalid_dates,
        "transactions_missing_paid_at": missing_paid_at,
        "transactions_invalid_paid_at": invalid_paid_at,
        "transactions_missing_revenue_date": missing_revenue_date,
        "transactions_revenue_date_mismatches": revenue_mismatches,
        "excluded_unpaid_or_invalid": excluded,
        "duplicate_invoices": duplicate_invoices,
    }, warnings


def normalize_sessions(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    frame = pd.DataFrame(index=raw.index)
    frame["source_row"] = range(2, len(raw) + 2)
    frame["timestamp"] = _datetimes(_series(raw, "Timestamp"))
    frame["business_date"] = _date(_series(raw, "Business Date"))
    frame["revenue_date"] = _date(_series(raw, "Revenue Date"))
    frame["paid_at"] = _datetimes(_series(raw, "Paid At"))
    frame["invoice_number"] = _text(_series(raw, "Invoice Number"))
    frame["session_number"] = _integer(_series(raw, "Session Number"))
    frame["table_number"] = _valid_table(_series(raw, "Table Number"))
    frame["package_id"] = _integer(_series(raw, "Package ID"))
    frame["package_name"] = _text(_series(raw, "Package Name"))
    frame["pricing_mode"] = _text(_series(raw, "Pricing Mode"))
    frame["start_time"] = _datetimes(_series(raw, "Start Time"))
    frame["end_time"] = _datetimes(_series(raw, "End Time"))
    frame["duration_minutes"] = _number(_series(raw, "Duration Minutes"))
    frame["rate"] = _currency(_series(raw, "Rate"))
    frame["session_subtotal"] = _currency(_series(raw, "Session Subtotal"))
    frame["status"] = _text(_series(raw, "Status")).str.upper()
    frame = frame.loc[frame["invoice_number"].ne("") & ~frame["status"].isin(("CANCELLED", "VOID"))].copy()

    missing_number = frame["session_number"].isna()
    if missing_number.any():
        fallback = frame.groupby("invoice_number", sort=False).cumcount() + 1
        frame.loc[missing_number, "session_number"] = fallback.loc[missing_number]
    duplicate_mask = frame.duplicated(["invoice_number", "session_number"], keep=False)
    duplicate_sessions = int(duplicate_mask.sum())
    frame = (
        frame.sort_values("source_row", kind="stable")
        .drop_duplicates(["invoice_number", "session_number"], keep="last")
        .reset_index(drop=True)
    )
    return frame, duplicate_sessions


def normalize_fnb_items(raw: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(index=raw.index)
    frame["source_row"] = range(2, len(raw) + 2)
    frame["timestamp"] = _datetimes(_series(raw, "Timestamp"))
    frame["business_date"] = _date(_series(raw, "Business Date"))
    frame["revenue_date"] = _date(_series(raw, "Revenue Date"))
    frame["paid_at"] = _datetimes(_series(raw, "Paid At"))
    frame["invoice_number"] = _text(_series(raw, "Invoice Number"))
    frame["transaction_type"] = _text(_series(raw, "Transaction Type")).str.upper()
    frame["table_number"] = _valid_table(_series(raw, "Table Number"))
    frame["menu_id"] = _integer(_series(raw, "Menu ID"))
    frame["menu_name"] = _text(_series(raw, "Menu Name"))
    frame["quantity"] = _number(_series(raw, "Quantity"))
    frame["unit_price"] = _currency(_series(raw, "Unit Price"))
    frame["line_total"] = _currency(_series(raw, "Line Total"))
    line_id_column = "Line ID" if "Line ID" in raw.columns else "Item ID" if "Item ID" in raw.columns else None
    frame["line_id"] = _text(_series(raw, line_id_column)) if line_id_column else ""
    frame = frame.loc[
        frame["invoice_number"].ne("") & frame["menu_name"].ne("") & frame["quantity"].gt(0)
    ].copy()
    if line_id_column:
        has_id = frame["line_id"].ne("")
        identified = frame.loc[has_id].drop_duplicates(["invoice_number", "line_id"], keep="last")
        frame = pd.concat([identified, frame.loc[~has_id]], ignore_index=True)
    return frame.sort_values(["invoice_number", "source_row"], kind="stable").reset_index(drop=True)


def normalize_daily_summary(raw: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(index=raw.index)
    frame["revenue_date"] = _date(_series(raw, "Revenue Date"))
    frame["invoice_count"] = _number(_series(raw, "Invoice Count"))
    for external, internal in (
        ("Billiard Revenue", "billiard_revenue"),
        ("F&B Revenue", "fnb_revenue"),
        ("Discount Total", "discount_total"),
        ("Grand Revenue", "grand_revenue"),
        ("Cash", "cash"),
        ("QRIS", "qris"),
        ("Card", "card"),
        ("Other", "other"),
    ):
        frame[internal] = _currency(_series(raw, external))
    frame["last_updated"] = _datetimes(_series(raw, "Last Updated"))
    return (
        frame.loc[frame["revenue_date"].notna()]
        .sort_values(["revenue_date", "last_updated"], na_position="first", kind="stable")
        .drop_duplicates("revenue_date", keep="last")
        .reset_index(drop=True)
    )


def prepare_data(raw_sheets: Mapping[str, pd.DataFrame]) -> DashboardData:
    raw_transactions = _raw(raw_sheets, "Transactions")
    raw_sessions = _raw(raw_sheets, "Billiard_Sessions")
    raw_fnb = _raw(raw_sheets, "FNB_Items")
    raw_daily = _raw(raw_sheets, "Daily_Summary")

    transactions, transaction_stats, warnings = normalize_transactions(raw_transactions)
    sessions, duplicate_sessions = normalize_sessions(raw_sessions)
    fnb_items = normalize_fnb_items(raw_fnb)
    daily = normalize_daily_summary(raw_daily)

    for sheet_name, raw_frame in (
        ("Transactions", raw_transactions),
        ("Billiard_Sessions", raw_sessions),
        ("FNB_Items", raw_fnb),
        ("Daily_Summary", raw_daily),
    ):
        actual_headers = tuple(str(column) for column in raw_frame.columns)
        mapped_headers = tuple(
            SHEET_COLUMN_MAPS[sheet_name][column]
            for column in actual_headers
            if column in SHEET_COLUMN_MAPS[sheet_name]
        )
        LOGGER.info(
            "sheet=%s actual_headers=%s mapped_canonical_columns=%s",
            sheet_name,
            actual_headers,
            mapped_headers,
        )
        missing = [column for column in SHEET_COLUMNS[sheet_name] if column not in raw_frame.columns]
        if missing:
            LOGGER.warning(
                "sheet=%s missing_expected_columns=%s actual_headers=%s mapped_canonical_columns=%s",
                sheet_name,
                missing,
                actual_headers,
                mapped_headers,
            )
            financial = set(SHEET_FINANCIAL_COLUMNS[sheet_name])
            for column in missing:
                if column in financial:
                    warnings.append(f"Missing expected column: {column}")
                else:
                    warnings.append(f"{sheet_name} is missing expected column: {column}.")
        for column in SHEET_FINANCIAL_COLUMNS[sheet_name]:
            if column not in raw_frame.columns:
                continue
            non_blank_count = _non_blank_count(raw_frame[column])
            numeric_count = int(raw_frame[column].map(normalize_currency).notna().sum())
            if non_blank_count > numeric_count:
                message = (
                    f"{sheet_name}.{column} contains "
                    f"{non_blank_count - numeric_count} unparseable financial value(s)."
                )
                warnings.append(message)
                LOGGER.warning(
                    "sheet=%s column=%s non_blank_count=%s numeric_count=%s",
                    sheet_name,
                    column,
                    non_blank_count,
                    numeric_count,
                )

    transaction_actual_headers = tuple(str(column) for column in raw_transactions.columns)
    transaction_mapped_headers = tuple(
        SHEET_COLUMN_MAPS["Transactions"][column]
        for column in transaction_actual_headers
        if column in SHEET_COLUMN_MAPS["Transactions"]
    )
    grand_total_raw = _series(raw_transactions, "Grand Total")
    grand_total_parsed = grand_total_raw.map(normalize_currency)
    grand_total_non_null = _non_blank_count(grand_total_raw)

    canonical_invoices = set(transactions["invoice_number"])
    session_orphans = int((~sessions["invoice_number"].isin(canonical_invoices)).sum())
    fnb_orphans = int((~fnb_items["invoice_number"].isin(canonical_invoices)).sum())
    sessions = sessions.loc[sessions["invoice_number"].isin(canonical_invoices)].copy()
    fnb_items = fnb_items.loc[fnb_items["invoice_number"].isin(canonical_invoices)].copy()

    invoice_context = transactions[[
        "invoice_number", "timestamp", "business_date", "revenue_date", "paid_at",
        "transaction_type", "table_number",
    ]].rename(columns={
        "timestamp": "parent_timestamp",
        "business_date": "parent_business_date",
        "revenue_date": "parent_revenue_date",
        "paid_at": "parent_paid_at",
    })
    sessions = sessions.merge(
        invoice_context[[
            "invoice_number", "parent_timestamp", "parent_business_date",
            "parent_revenue_date", "parent_paid_at",
        ]],
        on="invoice_number",
        how="left",
    )
    session_paid_mismatch = _mismatch_count(sessions["paid_at"], sessions["parent_paid_at"])
    session_revenue_mismatch = _mismatch_count(sessions["revenue_date"], sessions["parent_revenue_date"])
    session_business_mismatch = _mismatch_count(sessions["business_date"], sessions["parent_business_date"])
    sessions["timestamp"] = sessions["parent_timestamp"]
    sessions["business_date"] = sessions["parent_business_date"]
    sessions["revenue_date"] = sessions["parent_revenue_date"]
    sessions["paid_at"] = sessions["parent_paid_at"]
    sessions = sessions.drop(columns=[
        "parent_timestamp", "parent_business_date", "parent_revenue_date", "parent_paid_at",
    ])
    fnb_items = fnb_items.merge(
        invoice_context,
        on="invoice_number",
        how="left",
        suffixes=("", "_parent"),
    )
    fnb_paid_mismatch = _mismatch_count(fnb_items["paid_at"], fnb_items["parent_paid_at"])
    fnb_revenue_mismatch = _mismatch_count(fnb_items["revenue_date"], fnb_items["parent_revenue_date"])
    fnb_business_mismatch = _mismatch_count(fnb_items["business_date"], fnb_items["parent_business_date"])
    fnb_items["timestamp"] = fnb_items["parent_timestamp"]
    fnb_items["business_date"] = fnb_items["parent_business_date"]
    fnb_items["revenue_date"] = fnb_items["parent_revenue_date"]
    fnb_items["paid_at"] = fnb_items["parent_paid_at"]
    fnb_items["transaction_type"] = fnb_items["transaction_type"].where(
        fnb_items["transaction_type"].isin(TRANSACTION_TYPES), fnb_items["transaction_type_parent"]
    )
    fnb_items["table_number"] = fnb_items["table_number"].fillna(fnb_items["table_number_parent"])
    fnb_items = fnb_items.drop(columns=[
        "parent_timestamp", "parent_business_date", "parent_revenue_date", "parent_paid_at",
        "transaction_type_parent", "table_number_parent",
    ])

    fnb_totals = fnb_items.groupby("invoice_number", as_index=False)["line_total"].sum()
    fnb_expected = transactions[["invoice_number", "fnb_revenue"]].merge(fnb_totals, on="invoice_number", how="left").fillna({"line_total": 0})
    fnb_mismatch = int((fnb_expected["fnb_revenue"].sub(fnb_expected["line_total"]).abs() > 0.5).sum())
    session_totals = sessions.groupby("invoice_number", as_index=False)["session_subtotal"].sum()
    session_expected = transactions[["invoice_number", "billiard_revenue"]].merge(session_totals, on="invoice_number", how="left").fillna({"session_subtotal": 0})
    session_mismatch = int((session_expected["billiard_revenue"].sub(session_expected["session_subtotal"]).abs() > 0.5).sum())

    if session_orphans:
        warnings.append(f"{session_orphans} session row(s) do not have a canonical transaction.")
    if fnb_orphans:
        warnings.append(f"{fnb_orphans} F&B row(s) do not have a canonical transaction.")
    if transaction_stats["transactions_missing_paid_at"]:
        warnings.append(f"{transaction_stats['transactions_missing_paid_at']} transaction row(s) have missing Paid At.")
    if transaction_stats["transactions_invalid_paid_at"]:
        warnings.append(f"{transaction_stats['transactions_invalid_paid_at']} transaction row(s) have invalid Paid At.")
    if transaction_stats["transactions_missing_revenue_date"]:
        warnings.append(f"{transaction_stats['transactions_missing_revenue_date']} transaction row(s) have missing Revenue Date.")
    if transaction_stats["transactions_revenue_date_mismatches"]:
        warnings.append(
            f"{transaction_stats['transactions_revenue_date_mismatches']} transaction row(s) have Revenue Date inconsistent with Paid At."
        )
    if session_paid_mismatch or session_revenue_mismatch or session_business_mismatch:
        warnings.append("Billiard_Sessions contains transaction timing that does not match its parent invoice.")
    if fnb_paid_mismatch or fnb_revenue_mismatch or fnb_business_mismatch:
        warnings.append("FNB_Items contains transaction timing that does not match its parent invoice.")
    if fnb_mismatch:
        warnings.append(f"{fnb_mismatch} invoice(s) have incomplete or inconsistent F&B child totals.")
    if session_mismatch:
        warnings.append(f"{session_mismatch} invoice(s) have incomplete or inconsistent session totals.")
    if "Line ID" not in raw_fnb.columns and "Item ID" not in raw_fnb.columns and len(raw_fnb):
        warnings.append("FNB_Items has no immutable line identifier; identical legitimate lines are preserved.")

    health = DataHealth(
        raw_transactions=len(raw_transactions),
        raw_sessions=len(raw_sessions),
        raw_fnb_items=len(raw_fnb),
        normalized_transactions=len(transactions),
        normalized_sessions=len(sessions),
        normalized_fnb_items=len(fnb_items),
        duplicate_invoices=transaction_stats["duplicate_invoices"],
        duplicate_sessions=duplicate_sessions,
        invalid_business_dates=transaction_stats["invalid_business_dates"],
        transactions_missing_paid_at=transaction_stats["transactions_missing_paid_at"],
        transactions_invalid_paid_at=transaction_stats["transactions_invalid_paid_at"],
        transactions_missing_revenue_date=transaction_stats["transactions_missing_revenue_date"],
        transactions_revenue_date_mismatches=transaction_stats["transactions_revenue_date_mismatches"],
        excluded_unpaid_or_invalid=transaction_stats["excluded_unpaid_or_invalid"],
        sessions_without_invoice=session_orphans,
        session_paid_at_mismatches=session_paid_mismatch,
        session_revenue_date_mismatches=session_revenue_mismatch,
        session_business_date_mismatches=session_business_mismatch,
        fnb_without_invoice=fnb_orphans,
        fnb_paid_at_mismatches=fnb_paid_mismatch,
        fnb_revenue_date_mismatches=fnb_revenue_mismatch,
        fnb_business_date_mismatches=fnb_business_mismatch,
        fnb_total_mismatches=fnb_mismatch,
        session_total_mismatches=session_mismatch,
        transactions_actual_headers=transaction_actual_headers,
        transactions_mapped_headers=transaction_mapped_headers,
        grand_total_non_null_count=grand_total_non_null,
        grand_total_numeric_count=int(grand_total_parsed.notna().sum()),
        grand_total_sum=float(pd.to_numeric(grand_total_parsed, errors="coerce").fillna(0).sum()),
        billiard_subtotal_sum=float(_currency(_series(raw_transactions, "Billiard Subtotal")).sum()),
        fnb_subtotal_sum=float(_currency(_series(raw_transactions, "F&B Subtotal")).sum()),
        warnings=tuple(warnings),
    )
    return DashboardData(transactions, sessions, fnb_items, daily, health)


def filter_data(
    data: DashboardData,
    start_date: Any,
    end_date: Any,
    transaction_type: str = "All",
    cashier: str = "All",
    payment_method: str = "All",
    table_number: str | int = "All",
    date_basis: str = "Calendar / Paid At",
) -> DashboardData:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    date_column = "business_date" if date_basis == "Business Date" else "revenue_date"
    attribute_mask = pd.Series(True, index=data.transactions.index)
    if transaction_type != "All":
        attribute_mask &= data.transactions["transaction_type"].eq(transaction_type)
    if cashier != "All":
        attribute_mask &= data.transactions["cashier"].eq(cashier)
    if payment_method != "All":
        attribute_mask &= data.transactions["payment_method"].eq(payment_method)
    if table_number != "All":
        attribute_mask &= data.transactions["table_number"].eq(int(table_number))
    mask = attribute_mask & data.transactions[date_column].between(start, end)
    transactions = data.transactions.loc[mask].copy()
    transactions["reporting_date"] = transactions[date_column]
    attribute_invoices = set(data.transactions.loc[attribute_mask, "invoice_number"])
    range_end = end + pd.Timedelta(days=1)
    session_overlap = data.sessions["start_time"].lt(range_end) & data.sessions["end_time"].gt(start)
    sessions = data.sessions.loc[
        data.sessions["invoice_number"].isin(attribute_invoices) & session_overlap
    ].copy()
    invoice_numbers = set(transactions["invoice_number"])
    fnb_items = data.fnb_items.loc[data.fnb_items["invoice_number"].isin(invoice_numbers)].copy()
    daily = (
        data.daily_summary.loc[data.daily_summary["revenue_date"].between(start, end)].copy()
        if date_basis != "Business Date"
        else data.daily_summary.iloc[0:0].copy()
    )
    return replace(data, transactions=transactions, sessions=sessions, fnb_items=fnb_items, daily_summary=daily)


def data_health_rows(data: DashboardData, last_refresh: datetime | None = None) -> pd.DataFrame:
    health = data.health
    earliest = data.transactions["revenue_date"].min() if len(data.transactions) else pd.NaT
    latest = data.transactions["revenue_date"].max() if len(data.transactions) else pd.NaT
    values = (
        ("Transactions rows", health.raw_transactions),
        ("Canonical paid invoices", health.normalized_transactions),
        ("Session rows", health.raw_sessions),
        ("Canonical session rows", health.normalized_sessions),
        ("F&B rows", health.raw_fnb_items),
        ("Canonical F&B rows", health.normalized_fnb_items),
        ("Earliest revenue date", earliest.date().isoformat() if pd.notna(earliest) else "—"),
        ("Latest revenue date", latest.date().isoformat() if pd.notna(latest) else "—"),
        ("Last refresh", last_refresh.isoformat(timespec="seconds") if last_refresh else "—"),
        ("Duplicate invoice identities", health.duplicate_invoices),
        ("Transactions missing Paid At", health.transactions_missing_paid_at),
        ("Transactions invalid Paid At", health.transactions_invalid_paid_at),
        ("Transactions missing Revenue Date", health.transactions_missing_revenue_date),
        ("Transactions Revenue Date mismatch", health.transactions_revenue_date_mismatches),
        ("Sessions without invoice", health.sessions_without_invoice),
        ("Session Paid At mismatch", health.session_paid_at_mismatches),
        ("Session Revenue Date mismatch", health.session_revenue_date_mismatches),
        ("Session Business Date mismatch", health.session_business_date_mismatches),
        ("F&B rows without invoice", health.fnb_without_invoice),
        ("F&B Paid At mismatch", health.fnb_paid_at_mismatches),
        ("F&B Revenue Date mismatch", health.fnb_revenue_date_mismatches),
        ("F&B Business Date mismatch", health.fnb_business_date_mismatches),
        ("F&B total mismatches", health.fnb_total_mismatches),
        ("Session total mismatches", health.session_total_mismatches),
        ("Transactions actual headers", ", ".join(health.transactions_actual_headers) or "—"),
        ("Transactions mapped headers", ", ".join(health.transactions_mapped_headers) or "—"),
        ("Grand Total non-null count", health.grand_total_non_null_count),
        ("Grand Total numeric count", health.grand_total_numeric_count),
        ("Grand Total sum", health.grand_total_sum),
        ("Billiard subtotal sum", health.billiard_subtotal_sum),
        ("F&B subtotal sum", health.fnb_subtotal_sum),
    )
    result = pd.DataFrame(values, columns=["Check", "Value"])
    result["Check"] = result["Check"].astype("string")
    result["Value"] = result["Value"].map(str).astype("string")
    return result
