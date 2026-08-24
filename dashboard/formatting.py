"""Indonesian display formatting; source DataFrames remain numeric."""

from __future__ import annotations

import math


def format_rupiah(value: float | int | None) -> str:
    number = 0 if value is None or (isinstance(value, float) and math.isnan(value)) else round(float(value))
    return f"Rp {number:,.0f}".replace(",", ".")


def format_percent(value: float | int | None) -> str:
    number = 0.0 if value is None else float(value)
    return f"{number:.1f}%"


def format_duration(minutes: float | int | None) -> str:
    total = max(0, round(float(minutes or 0)))
    hours, remaining = divmod(total, 60)
    return f"{hours}h {remaining:02d}m" if hours else f"{remaining}m"

