from pathlib import Path

import pandas as pd
import streamlit as st


PROP_PATH = Path("outputs/top_mlb_props.csv")
LOG_PATH = Path("data/model_results_log.csv")


st.set_page_config(
    page_title="Jude's Sports Model",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Jude's Sports Model")
st.caption(
    "MLB model projections, ranked props, and tracked performance."
)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, UnicodeDecodeError):
        return pd.DataFrame()


def format_market(value: str) -> str:
    return (
        str(value)
        .replace("_", " ")
        .title()
    )


props = load_csv(PROP_PATH)
results = load_csv(LOG_PATH)


# -----------------------------
# Performance summary
# -----------------------------
graded = pd.DataFrame()

if not results.empty and "win_loss" in results.columns:
    graded = results[
        results["win_loss"].isin(["WIN", "LOSS", "PUSH"])
    ].copy()

wins = int((graded["win_loss"] == "WIN").sum()) if not graded.empty else 0
losses = int((graded["win_loss"] == "LOSS").sum()) if not graded.empty else 0
pushes = int((graded["win_loss"] == "PUSH").sum()) if not graded.empty else 0

decisions = wins + losses
hit_rate = (wins / decisions * 100) if decisions else 0

if not graded.empty and "profit_units" in graded.columns:
    units = pd.to_numeric(
        graded["profit_units"],
        errors="coerce",
    ).fillna(0).sum()
else:
    units = 0.0


metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric("Tracked Picks", decisions + pushes)
metric_2.metric("Record", f"{wins}-{losses}-{pushes}")
metric_3.metric("Hit Rate", f"{hit_rate:.1f}%")
metric_4.metric("Flat Units", f"{units:+.1f}")


st.divider()


# -----------------------------
# Today's props
# -----------------------------
st.header("🔥 Today's Top MLB Props")

if props.empty:
    st.info(
        "No ranked props are currently available. "
        "Run the daily model and push the updated output to GitHub."
    )
else:
    props = props.copy()

    if "market" in props.columns:
        props["market_display"] = props["market"].apply(format_market)

    platforms = (
        sorted(props["platform"].dropna().unique())
        if "platform" in props.columns
        else []
    )

    markets = (
        sorted(props["market_display"].dropna().unique())
        if "market_display" in props.columns
        else []
    )

    filter_1, filter_2, filter_3 = st.columns(3)

    with filter_1:
        selected_platforms = st.multiselect(
            "Platform",
            platforms,
            default=platforms,
        )

    with filter_2:
        selected_markets = st.multiselect(
            "Market",
            markets,
            default=markets,
        )

    with filter_3:
        max_props = st.slider(
            "Maximum props",
            min_value=1,
            max_value=25,
            value=min(10, max(1, len(props))),
        )

    filtered = props.copy()

    if selected_platforms and "platform" in filtered.columns:
        filtered = filtered[
            filtered["platform"].isin(selected_platforms)
        ]

    if selected_markets and "market_display" in filtered.columns:
        filtered = filtered[
            filtered["market_display"].isin(selected_markets)
        ]

    if "edge" in filtered.columns:
        filtered["edge"] = pd.to_numeric(
            filtered["edge"],
            errors="coerce",
        )

        filtered["absolute_edge"] = filtered["edge"].abs()

        filtered = filtered.sort_values(
            "absolute_edge",
            ascending=False,
        )

    filtered = filtered.head(max_props)

    if filtered.empty:
        st.warning("No props match the selected filters.")
    else:
        for _, row in filtered.iterrows():
            tier = row.get("tier", "PROP")
            player = row.get("player", "Unknown Player")
            platform = row.get("platform", "Unknown Platform")
            market = row.get(
                "market_display",
                format_market(row.get("market", "")),
            )
            pick = row.get("pick", "")
            line = row.get("line", "")
            projection = row.get("projection", "")
            edge = row.get("edge", "")

            with st.container(border=True):
                left, middle, right = st.columns([2.2, 1.3, 1])

                with left:
                    st.subheader(f"{tier}: {player}")
                    st.write(f"**{platform} — {market}**")
                    st.write(f"Model pick: **{pick} {line}**")

                with middle:
                    st.metric(
                        "Model Projection",
                        f"{float(projection):.2f}"
                        if pd.notna(projection)
                        else "N/A",
                    )

                with right:
                    st.metric(
                        "Projection Edge",
                        f"{float(edge):+.2f}"
                        if pd.notna(edge)
                        else "N/A",
                    )


# -----------------------------
# Performance tables
# -----------------------------
st.divider()
st.header("📈 Model Performance")

if graded.empty:
    st.info(
        "There are no genuinely graded live picks yet. "
        "The earlier sample slate should not be treated as verified performance."
    )
else:
    tab_1, tab_2, tab_3 = st.tabs(
        ["By Market", "By Platform", "Pick History"]
    )

    def group_summary(data: pd.DataFrame, column: str) -> pd.DataFrame:
        rows = []

        for group_name, group in data.groupby(column, dropna=False):
            group_wins = int((group["win_loss"] == "WIN").sum())
            group_losses = int((group["win_loss"] == "LOSS").sum())
            group_pushes = int((group["win_loss"] == "PUSH").sum())
            group_decisions = group_wins + group_losses

            group_hit_rate = (
                group_wins / group_decisions * 100
                if group_decisions
                else 0
            )

            group_units = pd.to_numeric(
                group.get("profit_units"),
                errors="coerce",
            ).fillna(0).sum()

            rows.append({
                column: group_name,
                "Picks": len(group),
                "Wins": group_wins,
                "Losses": group_losses,
                "Pushes": group_pushes,
                "Hit Rate": round(group_hit_rate, 1),
                "Units": round(group_units, 2),
            })

        return pd.DataFrame(rows)

    with tab_1:
        if "market" in graded.columns:
            st.dataframe(
                group_summary(graded, "market"),
                use_container_width=True,
                hide_index=True,
            )

    with tab_2:
        if "platform" in graded.columns:
            st.dataframe(
                group_summary(graded, "platform"),
                use_container_width=True,
                hide_index=True,
            )

    with tab_3:
        history_columns = [
            "date",
            "platform",
            "player",
            "market",
            "line",
            "projection",
            "pick",
            "actual_result",
            "win_loss",
            "profit_units",
        ]

        history_columns = [
            column
            for column in history_columns
            if column in graded.columns
        ]

        st.dataframe(
            graded[history_columns].sort_values(
                "date",
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )


st.divider()
st.caption(
    "Model projections are experimental. Grades currently reflect "
    "projection differences and are not proven win probabilities."
)
