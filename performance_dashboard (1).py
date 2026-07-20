"""Streamlit verified-performance section.

Usage in app.py:
    from performance_dashboard import render_performance_dashboard
    render_performance_dashboard(history)
"""

from __future__ import annotations

import calendar
from html import escape
from typing import Any

import pandas as pd
import streamlit as st


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    value = str(value).strip()
    return "" if value.lower() in {"nan", "<na>", "none"} else value


def _tier(row: pd.Series) -> str:
    confidence = _text(row.get("confidence_tier")).title()
    grade = _text(row.get("grade")).upper()
    elite = row.get("elite_eligible", False)

    if isinstance(elite, str):
        elite = elite.strip().lower() in {"true", "1", "yes"}

    if confidence == "Elite" or bool(elite):
        return "Elite"
    if grade == "A+":
        return "A+"
    return confidence or grade or "Unclassified"


def _prepare(history: pd.DataFrame) -> pd.DataFrame:
    if history is None or history.empty or "outcome" not in history.columns:
        return pd.DataFrame()

    frame = history.copy()
    frame["outcome"] = frame["outcome"].astype("string").str.strip().str.upper()
    frame = frame.loc[frame["outcome"].isin({"WIN", "LOSS", "PUSH"})].copy()

    date_col = next(
        (c for c in ["event_date", "date", "game_date", "graded_date"] if c in frame.columns),
        None,
    )
    if date_col is None:
        return pd.DataFrame()

    frame["_date"] = pd.to_datetime(frame[date_col], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["_date"]).copy()

    for col in ["profit", "stake", "probability"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    if "profit" not in frame.columns:
        frame["profit"] = 0.0
    if "stake" not in frame.columns:
        frame["stake"] = 1.0

    frame["profit"] = frame["profit"].fillna(0.0)
    frame["stake"] = frame["stake"].fillna(1.0)
    frame.loc[frame["stake"] <= 0, "stake"] = 1.0
    frame["_tier"] = frame.apply(_tier, axis=1)
    frame["_month"] = frame["_date"].dt.to_period("M").astype(str)
    return frame.sort_values("_date").reset_index(drop=True)


def _summary(group: pd.DataFrame) -> dict[str, Any]:
    wins = int(group["outcome"].eq("WIN").sum())
    losses = int(group["outcome"].eq("LOSS").sum())
    pushes = int(group["outcome"].eq("PUSH").sum())
    decisions = wins + losses
    units = float(group["profit"].sum())
    staked = float(group["stake"].sum())

    return {
        "Picks": len(group),
        "Wins": wins,
        "Losses": losses,
        "Pushes": pushes,
        "Record": f"{wins}-{losses}-{pushes}",
        "Hit Rate": wins / decisions * 100 if decisions else 0.0,
        "Units": units,
        "ROI": units / staked * 100 if staked else 0.0,
    }


def _calendar_html(frame: pd.DataFrame, year: int, month: int) -> str:
    daily = {}
    for day, group in frame.groupby(frame["_date"].dt.day):
        daily[int(day)] = _summary(group)

    html = ['<div class="units-calendar"><table><thead><tr>']
    for label in ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]:
        html.append(f"<th>{label}</th>")
    html.append("</tr></thead><tbody>")

    for week in calendar.Calendar(firstweekday=6).monthdayscalendar(year, month):
        html.append("<tr>")
        for day in week:
            if day == 0:
                html.append('<td class="empty"></td>')
                continue

            data = daily.get(day)
            if not data:
                html.append(
                    f'<td class="day neutral"><b>{day}</b><small>No picks</small></td>'
                )
                continue

            units = data["Units"]
            cls = "positive" if units > 0 else "negative" if units < 0 else "neutral"
            tooltip = (
                f"{data['Picks']} picks | {data['Record']} | "
                f"{data['Hit Rate']:.1f}% hit rate"
            )
            html.append(
                f'<td class="day {cls}" title="{escape(tooltip)}">'
                f'<b>{day}</b><strong>{units:+.2f}u</strong>'
                f'<small>{escape(data["Record"])}</small></td>'
            )
        html.append("</tr>")

    html.append("</tbody></table></div>")
    return "".join(html)


def render_performance_dashboard(history: pd.DataFrame) -> None:
    frame = _prepare(history)

    st.markdown(
        """
        <style>
        .units-calendar table{width:100%;table-layout:fixed;border-spacing:6px}
        .units-calendar th{text-align:center;color:#6b7280;padding:5px}
        .units-calendar td{height:95px;border:1px solid rgba(128,128,128,.25);
            border-radius:11px;padding:8px;vertical-align:top}
        .units-calendar .positive{background:rgba(34,197,94,.14)}
        .units-calendar .negative{background:rgba(239,68,68,.13)}
        .units-calendar .neutral{background:rgba(128,128,128,.06)}
        .units-calendar .empty{border:0}
        .units-calendar strong,.units-calendar small{display:block;margin-top:9px}
        .units-calendar small{color:#6b7280;font-size:.75rem}
        @media(max-width:850px){
          .units-calendar td{height:70px;padding:4px}
          .units-calendar small{display:none}
          .units-calendar strong{font-size:.8rem}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.header("🏆 Verified Performance Center")

    if frame.empty:
        st.info("This section will populate after settled picks with valid dates are saved.")
        return

    overall = _summary(frame)
    elite = _summary(frame.loc[frame["_tier"].eq("Elite")])
    a_plus = _summary(frame.loc[frame["_tier"].eq("A+")])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Overall Record", overall["Record"])
    c2.metric("Elite Record", elite["Record"])
    c3.metric("A+ Record", a_plus["Record"])
    c4.metric("Tracked Units", f"{overall['Units']:+.2f}u")
    c5.metric("Verified ROI", f"{overall['ROI']:+.1f}%")

    tier_tab, calendar_tab, monthly_tab, streak_tab, picks_tab = st.tabs(
        ["Elite / A+ Records", "Units Calendar", "Monthly Reports", "Streaks", "Verified Picks"]
    )

    with tier_tab:
        order = ["Elite", "A+", "Strong", "A", "Good", "B+", "Playable", "B"]
        present = frame["_tier"].dropna().unique().tolist()
        tiers = [t for t in order if t in present] + sorted(t for t in present if t not in order)

        rows = []
        for name in tiers:
            result = _summary(frame.loc[frame["_tier"].eq(name)])
            rows.append({
                "Tier": name,
                "Record": result["Record"],
                "Hit Rate": round(result["Hit Rate"], 1),
                "Units": round(result["Units"], 2),
                "ROI": round(result["ROI"], 1),
                "Picks": result["Picks"],
            })

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Hit Rate": st.column_config.NumberColumn(format="%.1f%%"),
                "Units": st.column_config.NumberColumn(format="%+.2f"),
                "ROI": st.column_config.NumberColumn(format="%+.1f%%"),
            },
        )

    with calendar_tab:
        months = sorted(frame["_month"].unique(), reverse=True)
        selected = st.selectbox(
            "Select month",
            months,
            format_func=lambda x: pd.Period(x, freq="M").strftime("%B %Y"),
        )
        period = pd.Period(selected, freq="M")
        month_frame = frame.loc[frame["_month"].eq(selected)]
        result = _summary(month_frame)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Monthly Record", result["Record"])
        m2.metric("Hit Rate", f"{result['Hit Rate']:.1f}%")
        m3.metric("Units", f"{result['Units']:+.2f}u")
        m4.metric("ROI", f"{result['ROI']:+.1f}%")
        m5.metric("Picks", result["Picks"])

        st.markdown(
            _calendar_html(month_frame, period.year, period.month),
            unsafe_allow_html=True,
        )
        st.caption("Hover over a day to see its pick count and hit rate.")

    with monthly_tab:
        rows = []
        for month, group in frame.groupby("_month", sort=True):
            result = _summary(group)
            rows.append({
                "Month": pd.Period(month, freq="M").strftime("%B %Y"),
                "Record": result["Record"],
                "Hit Rate": round(result["Hit Rate"], 1),
                "Units": round(result["Units"], 2),
                "ROI": round(result["ROI"], 1),
                "Picks": result["Picks"],
            })

        st.dataframe(pd.DataFrame(rows).iloc[::-1], use_container_width=True, hide_index=True)

        units_by_month = frame.groupby("_month")["profit"].sum().rename("Units").to_frame()
        units_by_month.index = [
            pd.Period(x, freq="M").strftime("%b %Y") for x in units_by_month.index
        ]
        st.subheader("Units by Month")
        st.bar_chart(units_by_month)

        cumulative = frame.groupby("_date")["profit"].sum().cumsum().rename("Cumulative Units")
        st.subheader("Cumulative Verified Units")
        st.line_chart(cumulative)

    with streak_tab:
        decisions = frame.loc[frame["outcome"].isin({"WIN", "LOSS"})].sort_values("_date")
        current_type = ""
        current_count = 0
        longest_win = 0
        longest_loss = 0

        for outcome in decisions["outcome"]:
            if outcome == current_type:
                current_count += 1
            else:
                current_type = outcome
                current_count = 1
            if outcome == "WIN":
                longest_win = max(longest_win, current_count)
            else:
                longest_loss = max(longest_loss, current_count)

        daily = frame.groupby("_date")["profit"].sum()
        monthly = frame.groupby("_month")["profit"].sum()

        s1, s2, s3 = st.columns(3)
        s1.metric("Current Streak", f"{current_count} {current_type.title()}" if current_count else "N/A")
        s2.metric("Longest Win Streak", longest_win)
        s3.metric("Longest Loss Streak", longest_loss)

        s4, s5, s6 = st.columns(3)
        s4.metric("Best Day", f"{daily.max():+.2f}u")
        s5.metric("Best Month", f"{monthly.max():+.2f}u")
        s6.metric("Worst Month", f"{monthly.min():+.2f}u")

    with picks_tab:
        tier_options = ["All"] + sorted(frame["_tier"].dropna().unique().tolist())
        selected_tier = st.selectbox("Tier", tier_options, key="verified_tier_filter")
        picks = frame if selected_tier == "All" else frame.loc[frame["_tier"].eq(selected_tier)]

        columns = [
            "_date", "_tier", "platform", "player", "market", "direction",
            "line", "sportsbook_odds", "probability", "outcome", "profit",
        ]
        columns = [c for c in columns if c in picks.columns]
        display = picks[columns].sort_values("_date", ascending=False).copy()
        display = display.rename(columns={
            "_date": "Date", "_tier": "Tier", "platform": "Platform",
            "player": "Player", "market": "Market", "direction": "Direction",
            "line": "Line", "sportsbook_odds": "Odds",
            "probability": "Probability", "outcome": "Result", "profit": "Units",
        })
        if "Date" in display.columns:
            display["Date"] = pd.to_datetime(display["Date"]).dt.date
        if "Probability" in display.columns:
            display["Probability"] = pd.to_numeric(display["Probability"], errors="coerce") * 100

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Probability": st.column_config.NumberColumn(format="%.1f%%"),
                "Units": st.column_config.NumberColumn(format="%+.2f"),
            },
        )
