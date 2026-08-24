"""Plotly figures for the BBL dashboard. Financial calculations stay in metrics.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


COLORS = {
    "green": "#16a34a",
    "teal": "#0d9488",
    "blue": "#2563eb",
    "amber": "#d97706",
    "red": "#dc2626",
    "slate": "#64748b",
}


def _style(figure: go.Figure, height: int = 430) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=55, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        hoverlabel=dict(namelength=-1),
    )
    return figure


def empty_figure(message: str, height: int = 360) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return _style(figure, height)


def revenue_trend_figure(trend: pd.DataFrame, metric_column: str, title: str) -> go.Figure:
    if trend.empty:
        return empty_figure("No paid invoices in this period")
    figure = px.line(
        trend,
        x="period_start",
        y=metric_column,
        markers=True,
        title=title,
        color_discrete_sequence=[COLORS["green"]],
    )
    figure.update_traces(line_width=3, hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f}<extra></extra>")
    figure.update_xaxes(title=None)
    figure.update_yaxes(title=None, rangemode="tozero", gridcolor="rgba(148,163,184,0.2)")
    return _style(figure)


def revenue_composition_figure(transactions: pd.DataFrame) -> go.Figure:
    if transactions.empty:
        return empty_figure("No revenue composition available")
    values = [transactions["billiard_revenue"].sum(), transactions["fnb_revenue"].sum()]
    figure = go.Figure(go.Bar(
        x=["Billiard", "F&B"],
        y=values,
        marker_color=[COLORS["green"], COLORS["teal"]],
        text=[f"Rp {value:,.0f}" for value in values],
        textposition="outside",
        hovertemplate="%{x}<br>Rp %{y:,.0f}<extra></extra>",
    ))
    figure.update_layout(title="Revenue Composition")
    figure.update_yaxes(title=None, gridcolor="rgba(148,163,184,0.2)")
    return _style(figure, 360)


def table_heatmap_figure(table_data: pd.DataFrame) -> go.Figure:
    if table_data.empty:
        return empty_figure("No table session data")
    ordered = table_data.sort_values("table_number")
    revenue = ordered["billiard_revenue"].to_numpy().reshape(3, 4)
    text = []
    custom = []
    for row_index in range(3):
        text_row = []
        custom_row = []
        for column_index in range(4):
            row = ordered.iloc[row_index * 4 + column_index]
            text_row.append(
                f"<b>Table {int(row.table_number)}</b><br>Rp {row.billiard_revenue:,.0f}<br>{row.revenue_share_percent:.1f}%"
            )
            custom_row.append([row.session_count, row.playing_hours, row.average_revenue_per_session])
        text.append(text_row)
        custom.append(custom_row)
    figure = go.Figure(go.Heatmap(
        z=revenue,
        text=text,
        customdata=np.array(custom, dtype=object),
        texttemplate="%{text}",
        colorscale="Greens",
        xgap=7,
        ygap=7,
        showscale=False,
        hovertemplate=(
            "%{text}<br>Sessions: %{customdata[0]}<br>Playing hours: %{customdata[1]:.1f}"
            "<br>Avg/session: Rp %{customdata[2]:,.0f}<extra></extra>"
        ),
    ))
    figure.update_xaxes(showticklabels=False, fixedrange=True)
    figure.update_yaxes(showticklabels=False, autorange="reversed", fixedrange=True)
    figure.update_layout(title="Billiard Revenue by Physical Table Layout")
    return _style(figure, 470)


def table_performance_figure(frame: pd.DataFrame, sort_column: str) -> go.Figure:
    if frame.empty:
        return empty_figure("No table performance data")
    ordered = frame.sort_values(sort_column, ascending=True)
    figure = px.bar(
        ordered,
        x=sort_column,
        y=ordered["table_number"].map(lambda value: f"Table {int(value)}"),
        orientation="h",
        color=sort_column,
        color_continuous_scale="Greens",
    )
    figure.update_layout(title="Ranked Table Performance", coloraxis_showscale=False)
    figure.update_xaxes(title=None, gridcolor="rgba(148,163,184,0.2)")
    figure.update_yaxes(title=None)
    return _style(figure, 540)


def hourly_occupancy_figure(frame: pd.DataFrame) -> go.Figure:
    if frame.empty:
        return empty_figure("No session intervals available")
    figure = px.bar(
        frame,
        x="hour_label",
        y="occupancy_percent",
        color="occupancy_percent",
        color_continuous_scale="Blues",
        text=frame.apply(lambda row: f"{row.occupancy_percent:.1f}%<br>{row.average_active_tables:.2f}/12", axis=1),
    )
    figure.update_traces(textposition="outside", hovertemplate="%{x}<br>Occupancy %{y:.2f}%<extra></extra>")
    figure.update_layout(title="Minute-Weighted Hourly Occupancy", coloraxis_showscale=False)
    figure.update_xaxes(title=None)
    figure.update_yaxes(title="Occupancy %", range=[0, max(105, frame["occupancy_percent"].max() * 1.25)])
    return _style(figure, 480)


def occupancy_heatmap_figure(frame: pd.DataFrame) -> go.Figure:
    if frame.empty:
        return empty_figure("No occupancy matrix available")
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    matrix = frame.pivot(index="weekday", columns="hour_label", values="occupancy_percent").reindex(weekdays)
    figure = go.Figure(go.Heatmap(
        z=matrix.to_numpy(),
        x=matrix.columns,
        y=matrix.index,
        colorscale="YlGnBu",
        text=np.vectorize(lambda value: f"{value:.1f}%")(matrix.fillna(0).to_numpy()),
        texttemplate="%{text}",
        hovertemplate="%{y} %{x}<br>Average occupancy %{z:.2f}%<extra></extra>",
    ))
    figure.update_layout(title="Average Occupancy by Business Day and Hour")
    figure.update_xaxes(title=None)
    figure.update_yaxes(title=None, autorange="reversed")
    return _style(figure, 500)


def fnb_performance_figure(frame: pd.DataFrame, metric: str = "revenue", limit: int = 10, ascending: bool = False) -> go.Figure:
    if frame.empty:
        return empty_figure("No finalized F&B item snapshots")
    selected = frame.sort_values(metric, ascending=ascending).head(limit).sort_values(metric)
    figure = px.bar(
        selected,
        x=metric,
        y="menu_name",
        orientation="h",
        color=metric,
        color_continuous_scale="Tealgrn",
    )
    figure.update_layout(title=("Bottom" if ascending else "Top") + f" {limit} F&B Items", coloraxis_showscale=False)
    figure.update_xaxes(title=None, gridcolor="rgba(148,163,184,0.2)")
    figure.update_yaxes(title=None)
    return _style(figure, 500)


def fnb_popularity_figure(frame: pd.DataFrame) -> go.Figure:
    if frame.empty:
        return empty_figure("No finalized F&B item snapshots")
    figure = px.scatter(
        frame,
        x="average_selling_price",
        y="quantity_sold",
        size="revenue",
        color="revenue",
        text="menu_name",
        color_continuous_scale="Viridis",
        size_max=60,
    )
    figure.add_vline(x=frame["average_selling_price"].mean(), line_dash="dash", line_color=COLORS["slate"])
    figure.add_hline(y=frame["quantity_sold"].mean(), line_dash="dash", line_color=COLORS["slate"])
    figure.update_traces(textposition="top center")
    figure.update_layout(title="F&B Popularity vs Average Selling Price")
    figure.update_xaxes(title="Average Selling Price")
    figure.update_yaxes(title="Quantity Sold")
    return _style(figure, 520)


def category_bar_figure(frame: pd.DataFrame, category: str, value: str, title: str, color: str = "green") -> go.Figure:
    if frame.empty:
        return empty_figure(f"No data for {title.lower()}")
    ordered = frame.sort_values(value, ascending=True)
    figure = px.bar(
        ordered,
        x=value,
        y=category,
        orientation="h",
        color_discrete_sequence=[COLORS[color]],
    )
    figure.update_layout(title=title)
    figure.update_xaxes(title=None, gridcolor="rgba(148,163,184,0.2)")
    figure.update_yaxes(title=None)
    return _style(figure, max(350, 45 * len(ordered) + 100))


def payment_donut_figure(frame: pd.DataFrame) -> go.Figure:
    if frame.empty:
        return empty_figure("No payment method data")
    figure = px.pie(frame, names="payment_method", values="revenue", hole=0.55)
    figure.update_traces(textinfo="label+percent", hovertemplate="%{label}<br>Rp %{value:,.0f}<br>%{percent}<extra></extra>")
    figure.update_layout(title="Revenue by Payment Method")
    return _style(figure, 400)


def standalone_comparison_figure(summary: dict[str, float]) -> go.Figure:
    frame = pd.DataFrame({
        "source": ["Table-attached F&B", "Standalone F&B"],
        "revenue": [summary["table_attached_fnb_revenue"], summary["standalone_fnb_revenue"]],
    })
    figure = px.bar(frame, x="source", y="revenue", color="source", color_discrete_sequence=[COLORS["green"], COLORS["teal"]])
    figure.update_layout(title="Finalized F&B Revenue Source", showlegend=False)
    figure.update_xaxes(title=None)
    figure.update_yaxes(title=None, gridcolor="rgba(148,163,184,0.2)")
    return _style(figure, 360)
