"""Central financial, session, occupancy, and invoice calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

from .schema import OPERATING_HOURS, TOTAL_TABLES
from .transforms import DashboardData


@dataclass(frozen=True)
class SummaryMetrics:
    grand_revenue: float
    billiard_revenue: float
    fnb_revenue: float
    discount_total: float
    invoice_count: int
    average_invoice: float
    average_daily_revenue: float


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def summarize(transactions: pd.DataFrame, calendar_days: int | None = None) -> SummaryMetrics:
    invoice_count = int(len(transactions))
    grand = float(transactions["grand_total"].sum()) if invoice_count else 0.0
    billiard = float(transactions["billiard_revenue"].sum()) if invoice_count else 0.0
    fnb = float(transactions["fnb_revenue"].sum()) if invoice_count else 0.0
    discount = float(transactions["discount_amount"].sum()) if invoice_count else 0.0
    if calendar_days is None:
        calendar_days = max(1, int(transactions["reporting_date"].nunique())) if invoice_count else 0
    return SummaryMetrics(
        grand_revenue=grand,
        billiard_revenue=billiard,
        fnb_revenue=fnb,
        discount_total=discount,
        invoice_count=invoice_count,
        average_invoice=safe_divide(grand, invoice_count),
        average_daily_revenue=safe_divide(grand, calendar_days),
    )


def transactions_for_date(transactions: pd.DataFrame, reporting_date: Any) -> pd.DataFrame:
    target = pd.Timestamp(reporting_date).normalize()
    return transactions.loc[transactions["reporting_date"].eq(target)].copy()


TREND_METRICS = {
    "Total Revenue": "grand_revenue",
    "Billiard Revenue": "billiard_revenue",
    "F&B Revenue": "fnb_revenue",
    "Invoice Count": "invoice_count",
    "Average Invoice Value": "average_invoice",
}


def revenue_trend(transactions: pd.DataFrame, grain: str = "Daily") -> pd.DataFrame:
    columns = ["period_start", "grand_revenue", "billiard_revenue", "fnb_revenue", "invoice_count", "average_invoice"]
    if transactions.empty:
        return pd.DataFrame(columns=columns)
    frame = transactions.copy()
    dates = frame["reporting_date"].dt.normalize()
    if grain == "Weekly":
        frame["period_start"] = dates - pd.to_timedelta(dates.dt.dayofweek, unit="D")
    elif grain == "Monthly":
        frame["period_start"] = dates.dt.to_period("M").dt.to_timestamp()
    else:
        frame["period_start"] = dates
    grouped = frame.groupby("period_start", as_index=False).agg(
        grand_revenue=("grand_total", "sum"),
        billiard_revenue=("billiard_revenue", "sum"),
        fnb_revenue=("fnb_revenue", "sum"),
        invoice_count=("invoice_number", "nunique"),
    )
    grouped["average_invoice"] = grouped["grand_revenue"].div(grouped["invoice_count"]).fillna(0)
    return grouped.sort_values("period_start").reset_index(drop=True)[columns]


def trend_comparison(trend: pd.DataFrame, column: str) -> dict[str, float | None]:
    if trend.empty:
        return {"current": 0.0, "previous": None, "growth_percent": None}
    current = float(trend.iloc[-1][column])
    if len(trend) < 2:
        return {"current": current, "previous": None, "growth_percent": None}
    previous = float(trend.iloc[-2][column])
    growth = ((current - previous) / previous * 100.0) if previous else None
    return {"current": current, "previous": previous, "growth_percent": growth}


def table_performance(sessions: pd.DataFrame) -> pd.DataFrame:
    base = pd.DataFrame({"table_number": range(1, TOTAL_TABLES + 1)})
    if sessions.empty:
        for column in ("billiard_revenue", "session_count", "playing_hours", "average_duration_minutes", "revenue_per_hour"):
            base[column] = 0.0
        return base
    grouped = sessions.groupby("table_number", as_index=False).agg(
        billiard_revenue=("session_subtotal", "sum"),
        session_count=("session_number", "count"),
        duration_minutes=("duration_minutes", "sum"),
        average_duration_minutes=("duration_minutes", "mean"),
    )
    grouped["playing_hours"] = grouped["duration_minutes"] / 60.0
    grouped["revenue_per_hour"] = grouped.apply(
        lambda row: safe_divide(row["billiard_revenue"], row["playing_hours"]), axis=1
    )
    result = base.merge(grouped.drop(columns="duration_minutes"), on="table_number", how="left").fillna(0)
    result["session_count"] = result["session_count"].astype(int)
    return result


def table_heatmap(sessions: pd.DataFrame) -> pd.DataFrame:
    result = table_performance(sessions)
    total = float(result["billiard_revenue"].sum())
    result["revenue_share_percent"] = result["billiard_revenue"].map(lambda value: safe_divide(value * 100, total))
    result["average_revenue_per_session"] = result.apply(
        lambda row: safe_divide(row["billiard_revenue"], row["session_count"]), axis=1
    )
    return result


def fnb_performance(fnb_items: pd.DataFrame) -> pd.DataFrame:
    columns = ["menu_name", "quantity_sold", "revenue", "average_selling_price", "contribution_percent"]
    if fnb_items.empty:
        return pd.DataFrame(columns=columns)
    grouped = fnb_items.groupby("menu_name", as_index=False).agg(
        quantity_sold=("quantity", "sum"),
        revenue=("line_total", "sum"),
    )
    grouped["average_selling_price"] = grouped.apply(
        lambda row: safe_divide(row["revenue"], row["quantity_sold"]), axis=1
    )
    total = float(grouped["revenue"].sum())
    grouped["contribution_percent"] = grouped["revenue"].map(lambda value: safe_divide(value * 100, total))
    return grouped.sort_values(["revenue", "quantity_sold"], ascending=False).reset_index(drop=True)[columns]


def fnb_summary(transactions: pd.DataFrame, fnb_items: pd.DataFrame) -> dict[str, Any]:
    total = float(transactions["fnb_revenue"].sum()) if len(transactions) else 0.0
    invoices = int(transactions.loc[transactions["fnb_revenue"].gt(0), "invoice_number"].nunique()) if len(transactions) else 0
    quantity = float(fnb_items["quantity"].sum()) if len(fnb_items) else 0.0
    performance = fnb_performance(fnb_items)
    top_item = performance.iloc[0]["menu_name"] if len(performance) else "—"
    return {
        "total_revenue": total,
        "items_sold": quantity,
        "average_per_fnb_invoice": safe_divide(total, invoices),
        "top_item": top_item,
    }


def cashier_performance(transactions: pd.DataFrame) -> pd.DataFrame:
    columns = ["cashier", "invoice_count", "grand_revenue", "average_invoice", "discount_total"]
    if transactions.empty:
        return pd.DataFrame(columns=columns)
    frame = transactions.copy()
    frame["cashier"] = frame["cashier"].replace("", "Unknown / Legacy")
    grouped = frame.groupby("cashier", as_index=False).agg(
        invoice_count=("invoice_number", "nunique"),
        grand_revenue=("grand_total", "sum"),
        discount_total=("discount_amount", "sum"),
    )
    grouped["average_invoice"] = grouped["grand_revenue"].div(grouped["invoice_count"]).fillna(0)
    return grouped.sort_values("grand_revenue", ascending=False).reset_index(drop=True)[columns]


def payment_method_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    columns = ["payment_method", "transaction_count", "revenue", "revenue_percent"]
    if transactions.empty:
        return pd.DataFrame(columns=columns)
    frame = transactions.copy()
    frame["payment_method"] = frame["payment_method"].replace("", "Other")
    grouped = frame.groupby("payment_method", as_index=False).agg(
        transaction_count=("invoice_number", "nunique"),
        revenue=("grand_total", "sum"),
    )
    total = float(grouped["revenue"].sum())
    grouped["revenue_percent"] = grouped["revenue"].map(lambda value: safe_divide(value * 100, total))
    return grouped.sort_values("revenue", ascending=False).reset_index(drop=True)[columns]


def discount_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    columns = ["discount_name", "times_used", "gross_before_discount", "discount_amount", "net_revenue"]
    if transactions.empty:
        return pd.DataFrame(columns=columns)
    frame = transactions.loc[transactions["discount_amount"].gt(0)].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["discount_name"] = frame["discount_name"].replace("", "Unnamed discount")
    grouped = frame.groupby("discount_name", as_index=False).agg(
        times_used=("invoice_number", "nunique"),
        gross_before_discount=("subtotal", "sum"),
        discount_amount=("discount_amount", "sum"),
        net_revenue=("grand_total", "sum"),
    )
    return grouped.sort_values("discount_amount", ascending=False).reset_index(drop=True)[columns]


def standalone_summary(transactions: pd.DataFrame) -> dict[str, float | int]:
    standalone = transactions.loc[transactions["transaction_type"].eq("STANDALONE_FNB")]
    table = transactions.loc[transactions["transaction_type"].eq("TABLE")]
    count = int(standalone["invoice_number"].nunique())
    standalone_revenue = float(standalone["grand_total"].sum())
    return {
        "standalone_invoice_count": count,
        "standalone_revenue": standalone_revenue,
        "average_standalone_order": safe_divide(standalone_revenue, count),
        "table_attached_fnb_revenue": float(table["fnb_revenue"].sum()),
        "standalone_fnb_revenue": float(standalone["fnb_revenue"].sum()),
    }


def package_performance(sessions: pd.DataFrame) -> pd.DataFrame:
    columns = ["package_name", "session_count", "revenue", "playing_hours", "average_duration_minutes", "average_revenue_per_session"]
    if sessions.empty:
        return pd.DataFrame(columns=columns)
    frame = sessions.copy()
    frame["package_name"] = frame["package_name"].replace("", "Unknown / Legacy")
    grouped = frame.groupby("package_name", as_index=False).agg(
        session_count=("session_number", "count"),
        revenue=("session_subtotal", "sum"),
        duration_minutes=("duration_minutes", "sum"),
        average_duration_minutes=("duration_minutes", "mean"),
    )
    grouped["playing_hours"] = grouped["duration_minutes"] / 60.0
    grouped["average_revenue_per_session"] = grouped["revenue"].div(grouped["session_count"]).fillna(0)
    return grouped.drop(columns="duration_minutes").sort_values("revenue", ascending=False).reset_index(drop=True)[columns]


def business_kpis(transactions: pd.DataFrame, sessions: pd.DataFrame) -> dict[str, float]:
    table_invoices = transactions.loc[transactions["transaction_type"].eq("TABLE")]
    playing_hours = float(sessions["duration_minutes"].sum()) / 60.0 if len(sessions) else 0.0
    session_count = int(len(sessions))
    table_invoice_count = int(table_invoices["invoice_number"].nunique())
    with_fnb = int(table_invoices.loc[table_invoices["fnb_revenue"].gt(0), "invoice_number"].nunique())
    return {
        "revenue_per_table_hour": safe_divide(float(sessions["session_subtotal"].sum()) if len(sessions) else 0, playing_hours),
        "fnb_revenue_per_table_invoice": safe_divide(float(table_invoices["fnb_revenue"].sum()), table_invoice_count),
        "fnb_attach_rate_percent": safe_divide(with_fnb * 100, table_invoice_count),
        "average_session_duration_minutes": safe_divide(float(sessions["duration_minutes"].sum()) if len(sessions) else 0, session_count),
        "average_sessions_per_invoice": safe_divide(session_count, table_invoice_count),
    }


def _calendar_dates(values: Iterable[Any]) -> list[pd.Timestamp]:
    return sorted({pd.Timestamp(value).normalize() for value in values})


def occupancy_contributions(sessions: pd.DataFrame, calendar_dates: Iterable[Any]) -> pd.DataFrame:
    """Return actual calendar-timeline table-minutes for every date/hour/table."""
    dates = _calendar_dates(calendar_dates)
    columns = ["calendar_date", "weekday", "hour", "hour_label", "table_number", "occupied_minutes"]
    if sessions.empty or not dates:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for session in sessions.itertuples(index=False):
        start = pd.Timestamp(session.start_time) if pd.notna(session.start_time) else pd.NaT
        end = pd.Timestamp(session.end_time) if pd.notna(session.end_time) else pd.NaT
        if pd.isna(start) or pd.isna(end) or end <= start or pd.isna(session.table_number):
            continue
        for calendar_date in dates:
            for hour in OPERATING_HOURS:
                bucket_start = calendar_date + pd.Timedelta(hours=hour)
                bucket_end = bucket_start + pd.Timedelta(hours=1)
                overlap_start = max(start, bucket_start)
                overlap_end = min(end, bucket_end)
                minutes = max(0.0, (overlap_end - overlap_start).total_seconds() / 60.0)
                if minutes:
                    rows.append({
                        "calendar_date": calendar_date,
                        "weekday": calendar_date.day_name(),
                        "hour": hour,
                        "hour_label": f"{hour:02d}:00",
                        "table_number": int(session.table_number),
                        "occupied_minutes": minutes,
                    })
    if not rows:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(rows).groupby(
        ["calendar_date", "weekday", "hour", "hour_label", "table_number"], as_index=False
    )["occupied_minutes"].sum()
    result["occupied_minutes"] = result["occupied_minutes"].clip(upper=60.0)
    return result


def hourly_occupancy(sessions: pd.DataFrame, calendar_dates: Iterable[Any]) -> pd.DataFrame:
    dates = _calendar_dates(calendar_dates)
    contributions = occupancy_contributions(sessions, dates)
    totals = contributions.groupby(["hour", "hour_label"], as_index=False)["occupied_minutes"].sum() if len(contributions) else pd.DataFrame()
    base = pd.DataFrame({"hour": OPERATING_HOURS, "hour_label": [f"{hour:02d}:00" for hour in OPERATING_HOURS]})
    if len(totals):
        base = base.merge(totals, on=["hour", "hour_label"], how="left")
    else:
        base["occupied_minutes"] = 0.0
    base["occupied_minutes"] = base["occupied_minutes"].fillna(0.0)
    day_count = len(dates)
    base["average_active_tables"] = base["occupied_minutes"].map(lambda value: safe_divide(value, 60 * day_count))
    base["occupancy_percent"] = base["occupied_minutes"].map(
        lambda value: safe_divide(value * 100, TOTAL_TABLES * 60 * day_count)
    )
    return base


def occupancy_day_hour(sessions: pd.DataFrame, calendar_dates: Iterable[Any]) -> pd.DataFrame:
    dates = _calendar_dates(calendar_dates)
    contributions = occupancy_contributions(sessions, dates)
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    denominators = pd.Series([date.day_name() for date in dates]).value_counts().to_dict() if dates else {}
    totals = contributions.groupby(["weekday", "hour"], as_index=False)["occupied_minutes"].sum() if len(contributions) else pd.DataFrame()
    grid = pd.MultiIndex.from_product([weekdays, OPERATING_HOURS], names=["weekday", "hour"]).to_frame(index=False)
    if len(totals):
        grid = grid.merge(totals, on=["weekday", "hour"], how="left")
    else:
        grid["occupied_minutes"] = 0.0
    grid["occupied_minutes"] = grid["occupied_minutes"].fillna(0.0)
    grid["hour_label"] = grid["hour"].map(lambda value: f"{value:02d}:00")
    grid["occupancy_percent"] = grid.apply(
        lambda row: safe_divide(
            row["occupied_minutes"] * 100,
            TOTAL_TABLES * 60 * denominators.get(row["weekday"], 0),
        ),
        axis=1,
    )
    return grid


def invoice_detail(data: DashboardData, invoice_number: str) -> dict[str, Any] | None:
    transaction = data.transactions.loc[data.transactions["invoice_number"].eq(invoice_number)]
    if transaction.empty:
        return None
    return {
        "transaction": transaction.iloc[0].to_dict(),
        "sessions": data.sessions.loc[data.sessions["invoice_number"].eq(invoice_number)].sort_values("session_number").copy(),
        "fnb_items": data.fnb_items.loc[data.fnb_items["invoice_number"].eq(invoice_number)].sort_values("source_row").copy(),
    }


def daily_comparison(transactions: pd.DataFrame, selected_date: Any) -> dict[str, SummaryMetrics]:
    target = pd.Timestamp(selected_date).normalize()
    return {
        "selected": summarize(transactions_for_date(transactions, target), 1),
        "previous_day": summarize(transactions_for_date(transactions, target - pd.Timedelta(days=1)), 1),
        "previous_week": summarize(transactions_for_date(transactions, target - pd.Timedelta(days=7)), 1),
    }
