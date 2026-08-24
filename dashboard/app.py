"""Production Streamlit dashboard for the current POS BBL Google Sheets archive."""

from __future__ import annotations

import logging
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dashboard.charts import (  # noqa: E402
    category_bar_figure,
    fnb_performance_figure,
    fnb_popularity_figure,
    hourly_occupancy_figure,
    occupancy_heatmap_figure,
    payment_donut_figure,
    revenue_composition_figure,
    revenue_trend_figure,
    standalone_comparison_figure,
    table_heatmap_figure,
    table_performance_figure,
)
from dashboard.data_loader import DashboardDataError, clear_sheet_cache, load_all_sheets  # noqa: E402
from dashboard.formatting import format_duration, format_percent, format_rupiah  # noqa: E402
from dashboard.metrics import (  # noqa: E402
    TREND_METRICS,
    business_kpis,
    cashier_performance,
    daily_comparison,
    discount_summary,
    fnb_performance,
    fnb_summary,
    hourly_occupancy,
    invoice_detail,
    occupancy_day_hour,
    package_performance,
    payment_method_summary,
    revenue_trend,
    standalone_summary,
    summarize,
    table_heatmap,
    table_performance,
    transactions_for_date,
    trend_comparison,
)
from dashboard.sample_data import demo_load_result  # noqa: E402
from dashboard.schema import TOTAL_TABLES  # noqa: E402
from dashboard.transforms import data_health_rows, filter_data, prepare_data  # noqa: E402


st.set_page_config(
    page_title="BBL Business Dashboard",
    page_icon="🎱",
    layout="wide",
)

logging.basicConfig(level=os.getenv("BBL_DASHBOARD_LOG_LEVEL", "INFO"))

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px;}
      [data-testid="stMetric"] {border: 1px solid rgba(148,163,184,.25); border-radius: 14px; padding: 14px 16px; background: rgba(22,163,74,.035);}
      [data-testid="stMetricLabel"] {font-weight: 650;}
      .bbl-kicker {color:#16a34a; font-weight:800; letter-spacing:.12em; font-size:.78rem; text-transform:uppercase;}
      .bbl-subtle {color:#64748b; margin-top:-.6rem;}
      .bbl-status {display:inline-block; border-radius:999px; padding:.25rem .65rem; background:rgba(22,163,74,.12); color:#16a34a; font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)


def display_metric(container, label: str, value: str, delta: str | None = None, help_text: str | None = None) -> None:
    container.metric(label, value, delta=delta, help=help_text)


def dataframe_with_currency(frame: pd.DataFrame, currency_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    result = frame.copy()
    for column in currency_columns:
        if column in result.columns:
            result[column] = result[column].map(format_rupiah)
    return result


def load_dashboard_data():
    demo_mode = os.getenv("BBL_DASHBOARD_DEMO", "").strip() == "1"
    if demo_mode:
        result = demo_load_result()
        st.warning("Demo mode is active. Values below are local fixtures, not production Google Sheets data.")
    else:
        try:
            result = load_all_sheets()
            st.session_state["last_good_load"] = result
        except DashboardDataError as error:
            result = st.session_state.get("last_good_load")
            if result is None:
                st.error("Unable to load Google Sheets data.")
                st.info(str(error))
                st.caption("Technical details were written to the server log; no credentials are displayed here.")
                st.stop()
            st.warning(
                "Google Sheets is temporarily unavailable. Showing the last successful refresh from "
                f"{result.loaded_at.strftime('%d %b %Y %H:%M:%S %Z')}."
            )
    return result, prepare_data(result.sheets)


def preset_dates(name: str, minimum: pd.Timestamp, maximum: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    if name == "Latest date":
        return maximum, maximum
    if name == "Previous date":
        target = max(minimum, maximum - pd.Timedelta(days=1))
        return target, target
    if name == "Last 7 days":
        return max(minimum, maximum - pd.Timedelta(days=6)), maximum
    if name == "This month":
        return max(minimum, maximum.to_period("M").start_time), maximum
    if name == "Last month":
        previous = maximum.to_period("M") - 1
        start = max(minimum, previous.start_time)
        end = min(maximum, previous.end_time.normalize())
        return (start, end) if start <= end else (minimum, minimum)
    if name == "All time":
        return minimum, maximum
    return max(minimum, maximum - pd.Timedelta(days=6)), maximum


def global_filters(data):
    st.sidebar.header("Filters")
    date_basis = st.sidebar.selectbox(
        "Reporting Date Basis",
        ["Calendar / Paid At", "Business Date"],
        index=0,
        key="filters_reporting_date_basis",
        help="Financial analytics default to the Jakarta calendar date derived from Paid At.",
    )
    date_column = "business_date" if date_basis == "Business Date" else "revenue_date"
    if data.transactions.empty:
        today = pd.Timestamp(date.today())
        minimum = maximum = today
    else:
        dates = data.transactions[date_column].dropna()
        minimum = dates.min() if len(dates) else pd.Timestamp(date.today())
        maximum = dates.max() if len(dates) else pd.Timestamp(date.today())

    preset = st.sidebar.selectbox(
        "Quick period",
        ["Latest date", "Previous date", "Last 7 days", "This month", "Last month", "All time", "Custom"],
        index=2,
        key="filters_quick_period",
    )
    default_start, default_end = preset_dates(preset, minimum, maximum)
    if preset == "Custom":
        start_date = st.sidebar.date_input(
            "Start Date",
            value=default_start.date(),
            min_value=minimum.date(),
            max_value=maximum.date(),
            key="filters_custom_start_date",
        )
        end_date = st.sidebar.date_input(
            "End Date",
            value=default_end.date(),
            min_value=minimum.date(),
            max_value=maximum.date(),
            key="filters_custom_end_date",
        )
    else:
        start_date = st.sidebar.date_input(
            "Start Date",
            value=default_start.date(),
            disabled=True,
            key="filters_preset_start_date",
        )
        end_date = st.sidebar.date_input(
            "End Date",
            value=default_end.date(),
            disabled=True,
            key="filters_preset_end_date",
        )
    if start_date > end_date:
        st.sidebar.error("Start Date must not be after End Date.")
        st.stop()

    type_options = ["All"] + sorted(data.transactions["transaction_type"].dropna().unique().tolist())
    cashier_options = ["All"] + sorted(value for value in data.transactions["cashier"].dropna().unique().tolist() if value)
    payment_options = ["All"] + sorted(value for value in data.transactions["payment_method"].dropna().unique().tolist() if value)
    transaction_type = st.sidebar.selectbox("Transaction Type", type_options, key="filters_transaction_type")
    cashier = st.sidebar.selectbox("Cashier", cashier_options, key="filters_cashier")
    payment_method = st.sidebar.selectbox("Payment Method", payment_options, key="filters_payment_method")
    table_number = st.sidebar.selectbox(
        "Table Number",
        ["All"] + list(range(1, TOTAL_TABLES + 1)),
        key="filters_table_number",
    )
    return start_date, end_date, transaction_type, cashier, payment_method, table_number, date_basis


def render_overview(filtered, start_date, end_date):
    latest_date = filtered.transactions["reporting_date"].max() if len(filtered.transactions) else pd.Timestamp(end_date)
    latest_transactions = transactions_for_date(filtered.transactions, latest_date)
    daily = summarize(latest_transactions, 1)
    calendar_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days + 1
    period = summarize(filtered.transactions, calendar_days)

    st.subheader(f"Daily Summary · {latest_date.strftime('%d %b %Y')}")
    columns = st.columns(6)
    display_metric(columns[0], "Total Revenue", format_rupiah(daily.grand_revenue))
    display_metric(columns[1], "Billiard Revenue", format_rupiah(daily.billiard_revenue))
    display_metric(columns[2], "F&B Revenue", format_rupiah(daily.fnb_revenue))
    display_metric(columns[3], "Invoice Count", f"{daily.invoice_count:,}")
    display_metric(columns[4], "Discount Total", format_rupiah(daily.discount_total))
    display_metric(columns[5], "Average Invoice", format_rupiah(daily.average_invoice))

    st.subheader("Selected Period")
    columns = st.columns(6)
    display_metric(columns[0], "Period Revenue", format_rupiah(period.grand_revenue))
    display_metric(columns[1], "Billiard Revenue", format_rupiah(period.billiard_revenue))
    display_metric(columns[2], "F&B Revenue", format_rupiah(period.fnb_revenue))
    display_metric(columns[3], "Invoice Count", f"{period.invoice_count:,}")
    display_metric(columns[4], "Average Daily Revenue", format_rupiah(period.average_daily_revenue))
    display_metric(columns[5], "Average Invoice", format_rupiah(period.average_invoice))

    left, right = st.columns([2, 1])
    with left:
        chart_controls = st.columns(2)
        metric_label = chart_controls[0].selectbox("Trend metric", list(TREND_METRICS), key="overview_trend_metric")
        grain = chart_controls[1].selectbox("Trend grain", ["Daily", "Weekly", "Monthly"], key="overview_trend_grain")
        trend = revenue_trend(filtered.transactions, grain)
        metric_column = TREND_METRICS[metric_label]
        comparison = trend_comparison(trend, metric_column)
        if metric_column == "invoice_count":
            current_label = f"{comparison['current']:,.0f}"
        else:
            current_label = format_rupiah(comparison["current"])
        delta = f"{comparison['growth_percent']:+.1f}% vs previous period" if comparison["growth_percent"] is not None else None
        st.metric(f"Current {metric_label}", current_label, delta=delta)
        if comparison["previous"] == 0:
            st.caption("Growth percentage is hidden because the previous period value is zero.")
        st.plotly_chart(
            revenue_trend_figure(trend, metric_column, f"{metric_label} · {grain}"),
            width="stretch",
            key="overview_revenue_trend",
        )
    with right:
        st.plotly_chart(
            revenue_composition_figure(filtered.transactions),
            width="stretch",
            key="overview_revenue_composition",
        )
        payment = payment_method_summary(filtered.transactions)
        st.plotly_chart(
            payment_donut_figure(payment),
            width="stretch",
            key="overview_payment_donut",
        )


def render_tables(filtered, start_date, end_date):
    latest_date = pd.Timestamp(end_date).normalize()
    mode = st.radio("Table analytics scope", ["Daily", "Selected Period"], horizontal=True, key="tables_analytics_scope")
    sessions = filtered.sessions if mode == "Selected Period" else filtered.sessions.loc[
        filtered.sessions["start_time"].lt(latest_date + pd.Timedelta(days=1))
        & filtered.sessions["end_time"].gt(latest_date)
    ]
    heat = table_heatmap(sessions)
    st.plotly_chart(
        table_heatmap_figure(heat),
        width="stretch",
        key="tables_revenue_heatmap",
    )
    st.caption("Heatmap revenue is session subtotal only. F&B is intentionally excluded. Layout: Tables 1–4, 5–8, 9–12.")

    performance = table_performance(sessions)
    mapping = {
        "Revenue": "billiard_revenue",
        "Sessions": "session_count",
        "Playing Hours": "playing_hours",
        "Revenue / Hour": "revenue_per_hour",
    }
    sort_label = st.selectbox("Rank tables by", list(mapping), key="tables_rank_metric")
    left, right = st.columns([1.35, 1])
    with left:
        st.plotly_chart(
            table_performance_figure(performance, mapping[sort_label]),
            width="stretch",
            key="tables_performance_bar",
        )
    with right:
        display = performance.rename(columns={
            "table_number": "Table",
            "billiard_revenue": "Billiard Revenue",
            "session_count": "Sessions",
            "playing_hours": "Playing Hours",
            "average_duration_minutes": "Avg Duration",
            "revenue_per_hour": "Revenue / Hour",
        })
        display["Table"] = display["Table"].map(lambda value: f"Table {int(value)}")
        display["Billiard Revenue"] = display["Billiard Revenue"].map(format_rupiah)
        display["Playing Hours"] = display["Playing Hours"].map(lambda value: f"{value:.1f}")
        display["Avg Duration"] = display["Avg Duration"].map(format_duration)
        display["Revenue / Hour"] = display["Revenue / Hour"].map(format_rupiah)
        st.dataframe(display, hide_index=True, width="stretch", key="tables_performance_table")

    st.subheader("Hourly Table Occupancy")
    occupancy_mode = st.radio("Occupancy scope", ["Daily", "Selected Period"], horizontal=True, key="tables_occupancy_scope")
    calendar_dates = [latest_date] if occupancy_mode == "Daily" else pd.date_range(start_date, end_date, freq="D")
    hourly = hourly_occupancy(filtered.sessions, calendar_dates)
    st.plotly_chart(
        hourly_occupancy_figure(hourly),
        width="stretch",
        key="tables_hourly_occupancy",
    )
    st.caption("Occupancy uses actual minute overlap in each 12:00–03:00 hourly bucket and divides by 12 tables.")

    matrix = occupancy_day_hour(filtered.sessions, pd.date_range(start_date, end_date, freq="D"))
    st.plotly_chart(
        occupancy_heatmap_figure(matrix),
        width="stretch",
        key="tables_weekday_hour_heatmap",
    )

    packages = package_performance(sessions)
    st.subheader("Package / Session Analysis")
    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(
            category_bar_figure(packages, "package_name", "session_count", "Package Popularity", "blue"),
            width="stretch",
            key="tables_package_popularity",
        )
    with right:
        display = packages.rename(columns={
            "package_name": "Package",
            "session_count": "Sessions",
            "revenue": "Revenue",
            "playing_hours": "Playing Hours",
            "average_duration_minutes": "Avg Duration",
            "average_revenue_per_session": "Avg Revenue / Session",
        })
        display["Revenue"] = display["Revenue"].map(format_rupiah)
        display["Playing Hours"] = display["Playing Hours"].map(lambda value: f"{value:.1f}")
        display["Avg Duration"] = display["Avg Duration"].map(format_duration)
        display["Avg Revenue / Session"] = display["Avg Revenue / Session"].map(format_rupiah)
        st.dataframe(display, hide_index=True, width="stretch", key="tables_package_table")


def render_fnb(filtered):
    summary = fnb_summary(filtered.transactions, filtered.fnb_items)
    columns = st.columns(4)
    display_metric(columns[0], "Total F&B Revenue", format_rupiah(summary["total_revenue"]))
    display_metric(columns[1], "Items Sold", f"{summary['items_sold']:,.0f}")
    display_metric(columns[2], "Average F&B / Invoice", format_rupiah(summary["average_per_fnb_invoice"]))
    display_metric(columns[3], "Top Selling Item", str(summary["top_item"]))

    performance = fnb_performance(filtered.fnb_items)
    metric_label = st.radio(
        "Top / Bottom metric",
        ["Revenue", "Quantity"],
        horizontal=True,
        key="fnb_top_bottom_metric",
    )
    metric = "revenue" if metric_label == "Revenue" else "quantity_sold"
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            fnb_performance_figure(performance, metric, 10, False),
            width="stretch",
            key="fnb_top_items",
        )
    with right:
        st.plotly_chart(
            fnb_performance_figure(performance, metric, 10, True),
            width="stretch",
            key="fnb_bottom_items",
        )

    st.subheader("F&B Popularity vs Revenue")
    st.plotly_chart(
        fnb_popularity_figure(performance),
        width="stretch",
        key="fnb_popularity_revenue",
    )
    st.info("Historical cost/modal is not synchronized to FNB_Items, so this dashboard does not calculate profit or margin. Add a costSnapshot field in a future POS schema if historical margin analysis is required.")

    display = performance.rename(columns={
        "menu_name": "Menu",
        "quantity_sold": "Quantity Sold",
        "revenue": "Revenue",
        "average_selling_price": "Average Selling Price",
        "contribution_percent": "Contribution",
    })
    for column in ("Revenue", "Average Selling Price"):
        display[column] = display[column].map(format_rupiah)
    display["Contribution"] = display["Contribution"].map(format_percent)
    st.dataframe(display, hide_index=True, width="stretch", key="fnb_performance_table")

    standalone = standalone_summary(filtered.transactions)
    st.subheader("Standalone F&B")
    columns = st.columns(3)
    display_metric(columns[0], "Standalone Invoice Count", f"{standalone['standalone_invoice_count']:,}")
    display_metric(columns[1], "Standalone Revenue", format_rupiah(standalone["standalone_revenue"]))
    display_metric(columns[2], "Average Standalone Order", format_rupiah(standalone["average_standalone_order"]))
    st.plotly_chart(
        standalone_comparison_figure(standalone),
        width="stretch",
        key="fnb_standalone_comparison",
    )
    st.caption("Classification uses the finalized Transactions.Transaction Type, so attached waiting-list F&B is counted once under its final table invoice.")


def render_operations(filtered, all_data, latest_date, date_basis):
    st.subheader("Transactions Closed by Cashier")
    st.caption("This reports which cashier finalized each transaction; it is not an employee sales-attribution metric.")
    cashiers = cashier_performance(filtered.transactions)
    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(
            category_bar_figure(cashiers, "cashier", "grand_revenue", "Revenue Closed by Cashier", "green"),
            width="stretch",
            key="operations_cashier_summary",
        )
    with right:
        display = cashiers.rename(columns={
            "cashier": "Cashier", "invoice_count": "Invoices", "grand_revenue": "Grand Revenue",
            "average_invoice": "Average Invoice", "discount_total": "Discount Total",
        })
        for column in ("Grand Revenue", "Average Invoice", "Discount Total"):
            display[column] = display[column].map(format_rupiah)
        st.dataframe(display, hide_index=True, width="stretch", key="operations_cashier_table")

    payment = payment_method_summary(filtered.transactions)
    discount = discount_summary(filtered.transactions)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            payment_donut_figure(payment),
            width="stretch",
            key="operations_payment_donut",
        )
        display = payment.rename(columns={"payment_method": "Method", "transaction_count": "Transactions", "revenue": "Revenue", "revenue_percent": "% Revenue"})
        display["Revenue"] = display["Revenue"].map(format_rupiah)
        display["% Revenue"] = display["% Revenue"].map(format_percent)
        st.dataframe(display, hide_index=True, width="stretch", key="operations_payment_table")
    with right:
        st.plotly_chart(
            category_bar_figure(discount, "discount_name", "discount_amount", "Discount Usage", "amber"),
            width="stretch",
            key="operations_discount_chart",
        )
        display = discount.rename(columns={
            "discount_name": "Discount", "times_used": "Times Used", "gross_before_discount": "Gross",
            "discount_amount": "Discount Amount", "net_revenue": "Net Revenue",
        })
        for column in ("Gross", "Discount Amount", "Net Revenue"):
            if column in display:
                display[column] = display[column].map(format_rupiah)
        st.dataframe(display, hide_index=True, width="stretch", key="operations_discount_table")

    st.subheader("Business KPIs")
    kpis = business_kpis(filtered.transactions, filtered.sessions)
    columns = st.columns(5)
    display_metric(columns[0], "Revenue / Table Hour", format_rupiah(kpis["revenue_per_table_hour"]))
    display_metric(columns[1], "F&B / Table Invoice", format_rupiah(kpis["fnb_revenue_per_table_invoice"]))
    display_metric(columns[2], "F&B Attach Rate", format_percent(kpis["fnb_attach_rate_percent"]))
    display_metric(columns[3], "Average Session", format_duration(kpis["average_session_duration_minutes"]))
    display_metric(columns[4], "Sessions / Table Invoice", f"{kpis['average_sessions_per_invoice']:.2f}")
    with st.expander("KPI definitions"):
        st.markdown(
            """
            - **Revenue / Table Hour:** sum of session snapshot subtotals ÷ finalized playing hours.
            - **F&B / Table Invoice:** TABLE invoice F&B subtotal ÷ TABLE invoice count.
            - **F&B Attach Rate:** TABLE invoices with F&B subtotal above zero ÷ all TABLE invoices.
            - **Average Session:** finalized session minutes ÷ session count.
            - **Sessions / Table Invoice:** continuation segments count separately; denominator remains unique TABLE invoices.
            """
        )

    st.subheader("Daily Closing Comparison")
    st.caption("Daily closing comparison is venue-wide and intentionally ignores cashier/payment/table filters.")
    comparison_transactions = all_data.transactions.copy()
    comparison_transactions["reporting_date"] = comparison_transactions[
        "business_date" if date_basis == "Business Date" else "revenue_date"
    ]
    comparison = daily_comparison(comparison_transactions, latest_date)
    selected = comparison["selected"]
    previous = comparison["previous_day"]
    previous_week = comparison["previous_week"]
    columns = st.columns(3)
    display_metric(columns[0], "Selected Day", format_rupiah(selected.grand_revenue), f"{selected.invoice_count} invoices")
    delta_day = ((selected.grand_revenue - previous.grand_revenue) / previous.grand_revenue * 100) if previous.grand_revenue else None
    delta_week = ((selected.grand_revenue - previous_week.grand_revenue) / previous_week.grand_revenue * 100) if previous_week.grand_revenue else None
    display_metric(columns[1], "Previous Day", format_rupiah(previous.grand_revenue), f"{delta_day:+.1f}% selected vs prior" if delta_day is not None else None)
    display_metric(columns[2], "Same Weekday Previous Week", format_rupiah(previous_week.grand_revenue), f"{delta_week:+.1f}% selected vs prior" if delta_week is not None else None)
    if delta_day is None or delta_week is None:
        st.caption("A percentage comparison is omitted when the comparison period has zero revenue.")


def render_invoices(filtered):
    st.subheader("Invoice Explorer")
    search = st.text_input(
        "Search Invoice Number",
        placeholder="INV-... or FNB-...",
        key="invoices_search",
    ).strip().upper()
    transactions = filtered.transactions.copy()
    if search:
        transactions = transactions.loc[transactions["invoice_number"].str.upper().str.contains(search, regex=False)]
    explorer = transactions[[
        "invoice_number", "timestamp", "business_date", "paid_at", "revenue_date",
        "transaction_type", "table_number", "cashier", "payment_method",
        "billiard_revenue", "fnb_revenue", "discount_amount", "grand_total",
    ]].rename(columns={
        "invoice_number": "Invoice Number", "timestamp": "Timestamp", "business_date": "Business Date",
        "paid_at": "Paid At", "revenue_date": "Revenue Date", "transaction_type": "Type",
        "table_number": "Table", "cashier": "Cashier", "payment_method": "Payment Method",
        "billiard_revenue": "Billiard Revenue", "fnb_revenue": "F&B Revenue",
        "discount_amount": "Discount", "grand_total": "Grand Total",
    })
    display = explorer.copy()
    display["Timestamp"] = display["Timestamp"].dt.strftime("%Y-%m-%d %H:%M").fillna("—")
    display["Business Date"] = display["Business Date"].dt.strftime("%Y-%m-%d")
    display["Revenue Date"] = display["Revenue Date"].dt.strftime("%Y-%m-%d")
    display["Table"] = display["Table"].map(lambda value: "Standalone" if pd.isna(value) else f"Table {int(value)}")
    for column in ("Billiard Revenue", "F&B Revenue", "Discount", "Grand Total"):
        display[column] = display[column].map(format_rupiah)
    display["Paid At"] = display["Paid At"].dt.strftime("%Y-%m-%d %H:%M").fillna("—")
    st.dataframe(display, hide_index=True, width="stretch", key="invoices_explorer_table")
    if transactions.empty:
        st.info("No invoice matches the current filters and search.")
        return

    selected_invoice = st.selectbox(
        "Open invoice",
        transactions["invoice_number"].tolist(),
        key="invoices_selected_invoice",
    )
    detail = invoice_detail(filtered, selected_invoice)
    if not detail:
        return
    transaction = detail["transaction"]
    columns = st.columns(5)
    display_metric(columns[0], "Invoice", selected_invoice)
    display_metric(columns[1], "Table / Type", "Standalone" if pd.isna(transaction["table_number"]) else f"Table {int(transaction['table_number'])}")
    display_metric(columns[2], "Cashier", transaction["cashier"] or "Unknown / Legacy")
    display_metric(columns[3], "Payment", transaction["payment_method"] or "Other")
    display_metric(columns[4], "Grand Total", format_rupiah(transaction["grand_total"]))
    timestamp_text = transaction["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if pd.notna(transaction["timestamp"]) else "—"
    business_text = transaction["business_date"].strftime("%Y-%m-%d") if pd.notna(transaction["business_date"]) else "—"
    paid_text = transaction["paid_at"].strftime("%Y-%m-%d %H:%M:%S") if pd.notna(transaction["paid_at"]) else "—"
    revenue_text = transaction["revenue_date"].strftime("%Y-%m-%d") if pd.notna(transaction["revenue_date"]) else "—"
    st.caption(
        f"Timestamp: {timestamp_text} · Business Date: {business_text} · "
        f"Paid At: {paid_text} · Revenue Date: {revenue_text}"
    )

    sessions = detail["sessions"]
    items = detail["fnb_items"]
    left, right = st.columns(2)
    with left:
        st.markdown("#### Billiard Sessions")
        if sessions.empty:
            st.caption("No billiard sessions (standalone F&B invoice).")
        else:
            session_display = sessions[["session_number", "table_number", "package_name", "start_time", "end_time", "duration_minutes", "rate", "session_subtotal"]].copy()
            session_display.columns = ["#", "Table", "Package", "Start", "End", "Duration", "Rate", "Subtotal"]
            session_display["Duration"] = session_display["Duration"].map(format_duration)
            session_display["Rate"] = session_display["Rate"].map(format_rupiah)
            session_display["Subtotal"] = session_display["Subtotal"].map(format_rupiah)
            st.dataframe(
                session_display,
                hide_index=True,
                width="stretch",
                key=f"invoices_sessions_{selected_invoice}",
            )
    with right:
        st.markdown("#### F&B Items")
        if items.empty:
            st.caption("No F&B items on this invoice.")
        else:
            item_display = items[["menu_name", "quantity", "unit_price", "line_total"]].copy()
            item_display.columns = ["Menu", "Qty", "Unit Price", "Line Total"]
            item_display["Unit Price"] = item_display["Unit Price"].map(format_rupiah)
            item_display["Line Total"] = item_display["Line Total"].map(format_rupiah)
            st.dataframe(
                item_display,
                hide_index=True,
                width="stretch",
                key=f"invoices_fnb_{selected_invoice}",
            )

    totals = st.columns(4)
    display_metric(totals[0], "Gross Subtotal", format_rupiah(transaction["subtotal"]))
    display_metric(totals[1], "Discount", f"{transaction['discount_name'] or 'No discount'} · {format_rupiah(transaction['discount_amount'])}")
    display_metric(totals[2], "Billiard", format_rupiah(transaction["billiard_revenue"]))
    display_metric(totals[3], "F&B", format_rupiah(transaction["fnb_revenue"]))


result, all_data = load_dashboard_data()

header_left, header_right = st.columns([4, 1])
with header_left:
    st.markdown('<div class="bbl-kicker">BBL reporting workspace</div>', unsafe_allow_html=True)
    st.title("🎱 BBL Business Dashboard")
    st.markdown('<div class="bbl-subtle">Finalized invoice, billiard session, and F&B snapshot analytics</div>', unsafe_allow_html=True)
with header_right:
    st.markdown(f'<div class="bbl-status">● {result.source}</div>', unsafe_allow_html=True)
    st.caption(f"Refreshed {result.loaded_at.strftime('%d %b %Y %H:%M:%S %Z')}")

if st.sidebar.button("↻ Refresh Data", width="stretch", key="sidebar_refresh_data"):
    clear_sheet_cache()
    st.rerun()
st.sidebar.caption("Automatic cache TTL: 60 seconds")

start_date, end_date, transaction_type, cashier, payment_method, table_number, date_basis = global_filters(all_data)
filtered = filter_data(
    all_data, start_date, end_date, transaction_type, cashier, payment_method, table_number, date_basis
)
st.caption(
    f"{date_basis} dates {start_date:%d %b %Y} → {end_date:%d %b %Y} · "
    f"{len(filtered.transactions):,} canonical paid invoice(s)"
)
if filtered.transactions.empty:
    st.warning("No paid invoices match the current filters. Charts will remain empty without failing.")

tabs = st.tabs(["Overview", "Tables", "F&B", "Operations", "Invoices"])
with tabs[0]:
    render_overview(filtered, start_date, end_date)
with tabs[1]:
    render_tables(filtered, start_date, end_date)
with tabs[2]:
    render_fnb(filtered)
with tabs[3]:
    latest = filtered.transactions["reporting_date"].max() if len(filtered.transactions) else pd.Timestamp(end_date)
    render_operations(filtered, all_data, latest, date_basis)
with tabs[4]:
    render_invoices(filtered)

with st.expander("Data Health"):
    st.dataframe(
        data_health_rows(all_data, result.loaded_at),
        hide_index=True,
        width="stretch",
        key="data_health_table",
    )
    if all_data.health.warnings:
        for warning in all_data.health.warnings:
            st.warning(warning)
    else:
        st.success("No duplicate invoices, missing transaction dates, child timing mismatches, or snapshot total mismatches detected.")
    st.caption("Data Health never displays service-account credentials, webhook secrets, or management PIN data.")
