from datetime import datetime
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st


PROP_PATH = Path("outputs/mlb_daily_card.csv")
LOG_PATH = Path("data/model_results_log.csv")


st.set_page_config(
    page_title="Jude's Sports Model",
    page_icon="📊",
    layout="wide",
)


# --------------------------------------------------
# Styling
# --------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1450px;
        }

        .main-title {
            font-size: 2.6rem;
            font-weight: 800;
            margin-bottom: 0;
        }

        .subtitle {
            color: #6b7280;
            margin-top: 0.2rem;
            margin-bottom: 0.5rem;
        }

        .updated-time {
            color: #9ca3af;
            font-size: 0.85rem;
            margin-bottom: 1.5rem;
        }

        .prop-card {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 14px;
            padding: 18px 20px;
            margin-top: 8px;
            margin-bottom: 10px;
        }

        .best-bet {
            border-left: 7px solid #16a34a;
        }

        .strong-lean {
            border-left: 7px solid #2563eb;
        }

        .lean {
            border-left: 7px solid #eab308;
        }

        .tier-label {
            font-size: 0.8rem;
            font-weight: 800;
            letter-spacing: 0.08rem;
        }

        .player-name {
            font-size: 1.45rem;
            font-weight: 800;
            margin-top: 4px;
        }

        .prop-detail {
            color: #6b7280;
            font-size: 0.95rem;
        }

        .pick-more {
            font-size: 1.1rem;
            font-weight: 800;
            color: #16a34a;
        }

        .pick-less {
            font-size: 1.1rem;
            font-weight: 800;
            color: #dc2626;
        }

        .empty-board {
            border: 1px dashed rgba(128, 128, 128, 0.35);
            border-radius: 14px;
            padding: 30px;
            text-align: center;
            color: #6b7280;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------
@st.cache_data(ttl=60)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ):
        return pd.DataFrame()


def get_file_updated_time(path: Path) -> str:
    if not path.exists():
        return "Not available"

    updated_timestamp = path.stat().st_mtime
    updated_datetime = datetime.fromtimestamp(updated_timestamp)

    return updated_datetime.strftime("%B %d, %Y at %I:%M %p")


def format_market(value: str) -> str:
    return str(value).replace("_", " ").title()


def safe_number(value, decimals=2):
    number = pd.to_numeric(value, errors="coerce")

    if pd.isna(number):
        return "N/A"

    return f"{float(number):.{decimals}f}"


def safe_odds(value):
    odds = pd.to_numeric(value, errors="coerce")

    if pd.isna(odds):
        return "N/A"

    odds = int(round(float(odds)))

    if odds > 0:
        return f"+{odds}"

    return str(odds)


def normalize_tier(value: str) -> str:
    value = str(value).upper().strip()

    mapping = {
        "A+": "BEST BET",
        "A": "STRONG LEAN",
        "B": "LEAN",
    }

    return mapping.get(value, value)


def tier_class(tier: str) -> str:
    normalized = normalize_tier(tier)

    return {
        "BEST BET": "best-bet",
        "STRONG LEAN": "strong-lean",
        "LEAN": "lean",
    }.get(normalized, "lean")


def tier_rank(tier: str) -> int:
    normalized = normalize_tier(tier)

    return {
        "BEST BET": 1,
        "STRONG LEAN": 2,
        "LEAN": 3,
    }.get(normalized, 99)


def clean_text(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() == "nan":
        return ""

    return text


def build_matchup_text(row: pd.Series) -> str:
    team = clean_text(row.get("team"))
    opponent = clean_text(row.get("opponent"))

    if team and opponent:
        return f"{team} vs. {opponent}"

    if team:
        return team

    if opponent:
        return f"Opponent: {opponent}"

    return ""


def format_commence_time(value) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return ""

    parsed = pd.to_datetime(value, errors="coerce", utc=True)

    if pd.isna(parsed):
        return str(value)

    try:
        local_time = parsed.tz_convert("America/Chicago")
    except TypeError:
        return str(value)

    return local_time.strftime("%b %d at %I:%M %p CT")


def prepare_props(props: pd.DataFrame) -> pd.DataFrame:
    if props.empty:
        return props

    props = props.copy()

    required_columns = [
        "grade",
        "platform",
        "player",
        "market",
        "line",
        "projection",
        "edge",
        "pick",
    ]

    for column in required_columns:
        if column not in props.columns:
            props[column] = pd.NA

    numeric_columns = [
        "line",
        "projection",
        "edge",
        "absolute_edge",
        "sportsbook_odds",
    ]

    for column in numeric_columns:
        if column in props.columns:
            props[column] = pd.to_numeric(
                props[column],
                errors="coerce",
            )

    props = props[
        props["grade"].isin(["A+", "A", "B"])
    ].copy()

    props = props[
        props["projection"].notna()
        & props["line"].notna()
        & props["edge"].notna()
    ].copy()

    props["tier"] = props["grade"].apply(normalize_tier)
    props["tier_rank"] = props["tier"].apply(tier_rank)
    props["market_display"] = props["market"].apply(format_market)

    if "absolute_edge" not in props.columns:
        props["absolute_edge"] = props["edge"].abs()
    else:
        props["absolute_edge"] = props["absolute_edge"].fillna(
            props["edge"].abs()
        )

    props["player_key"] = (
        props["player"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    props = props.sort_values(
        ["tier_rank", "absolute_edge"],
        ascending=[True, False],
    )

    # Backup protection in case duplicate players reach the app.
    props = props.drop_duplicates(
        subset=["player_key"],
        keep="first",
    )

    return props.reset_index(drop=True)


def build_summary(group: pd.DataFrame) -> pd.Series:
    wins = int((group["win_loss"] == "WIN").sum())
    losses = int((group["win_loss"] == "LOSS").sum())
    pushes = int((group["win_loss"] == "PUSH").sum())

    decisions = wins + losses
    hit_rate = wins / decisions * 100 if decisions else 0

    if "profit_units" in group.columns:
        units = pd.to_numeric(
            group["profit_units"],
            errors="coerce",
        ).fillna(0).sum()
    else:
        units = 0

    return pd.Series(
        {
            "Picks": len(group),
            "Wins": wins,
            "Losses": losses,
            "Pushes": pushes,
            "Hit Rate": round(hit_rate, 1),
            "Units": round(units, 2),
        }
    )


# --------------------------------------------------
# Load data
# --------------------------------------------------
props = prepare_props(load_csv(PROP_PATH))
results = load_csv(LOG_PATH)


# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    '<div class="main-title">📊 Jude’s Sports Model</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Ranked MLB projections, model edges, sportsbook lines, "
    "and tracked performance."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="updated-time">'
    f"Card last updated: {escape(get_file_updated_time(PROP_PATH))}"
    "</div>",
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Performance metrics
# --------------------------------------------------
graded = pd.DataFrame()

if not results.empty and "win_loss" in results.columns:
    graded = results[
        results["win_loss"].isin(
            [
                "WIN",
                "LOSS",
                "PUSH",
            ]
        )
    ].copy()

wins = (
    int((graded["win_loss"] == "WIN").sum())
    if not graded.empty
    else 0
)

losses = (
    int((graded["win_loss"] == "LOSS").sum())
    if not graded.empty
    else 0
)

pushes = (
    int((graded["win_loss"] == "PUSH").sum())
    if not graded.empty
    else 0
)

decisions = wins + losses
hit_rate = wins / decisions * 100 if decisions else 0

units = 0.0

if not graded.empty and "profit_units" in graded.columns:
    units = pd.to_numeric(
        graded["profit_units"],
        errors="coerce",
    ).fillna(0).sum()

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Tracked Picks",
    len(graded),
)

metric_2.metric(
    "Record",
    f"{wins}-{losses}-{pushes}",
)

metric_3.metric(
    "Hit Rate",
    f"{hit_rate:.1f}%",
)

metric_4.metric(
    "Flat Units",
    f"{units:+.1f}",
)

st.caption(
    "Only recommendations logged before games begin should count "
    "toward the verified live record."
)

st.divider()


# --------------------------------------------------
# Today's board
# --------------------------------------------------
header_left, header_right = st.columns([4, 1])

with header_left:
    st.header("🔥 Today’s Top MLB Props")

with header_right:
    if st.button(
        "↻ Refresh Board",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.rerun()


if props.empty:
    st.markdown(
        """
        <div class="empty-board">
            No actionable MLB props are currently available.<br>
            Run the Daily MLB Model workflow and allow GitHub to
            commit the updated output files.
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    platforms = sorted(
        props["platform"]
        .dropna()
        .astype(str)
        .unique()
    )

    markets = sorted(
        props["market_display"]
        .dropna()
        .astype(str)
        .unique()
    )

    tiers = [
        tier
        for tier in [
            "BEST BET",
            "STRONG LEAN",
            "LEAN",
        ]
        if tier in props["tier"].unique()
    ]

    picks = [
        pick
        for pick in [
            "MORE/YES",
            "LESS/NO",
        ]
        if pick in props["pick"].astype(str).unique()
    ]

    filter_1, filter_2, filter_3, filter_4 = st.columns(4)

    with filter_1:
        selected_platforms = st.multiselect(
            "Sportsbook",
            options=platforms,
            default=platforms,
        )

    with filter_2:
        selected_markets = st.multiselect(
            "Market",
            options=markets,
            default=markets,
        )

    with filter_3:
        selected_tiers = st.multiselect(
            "Rating",
            options=tiers,
            default=tiers,
        )

    with filter_4:
        selected_picks = st.multiselect(
            "Pick Direction",
            options=picks,
            default=picks,
        )

    sort_choice = st.selectbox(
        "Sort by",
        [
            "Best Rating",
            "Largest Edge",
            "Player Name",
            "Sportsbook",
        ],
    )

    filtered = props.copy()

    if selected_platforms:
        filtered = filtered[
            filtered["platform"].isin(selected_platforms)
        ]
    else:
        filtered = filtered.iloc[0:0]

    if selected_markets:
        filtered = filtered[
            filtered["market_display"].isin(selected_markets)
        ]
    else:
        filtered = filtered.iloc[0:0]

    if selected_tiers:
        filtered = filtered[
            filtered["tier"].isin(selected_tiers)
        ]
    else:
        filtered = filtered.iloc[0:0]

    if selected_picks:
        filtered = filtered[
            filtered["pick"].isin(selected_picks)
        ]
    else:
        filtered = filtered.iloc[0:0]

    if sort_choice == "Largest Edge":
        filtered = filtered.sort_values(
            "absolute_edge",
            ascending=False,
        )

    elif sort_choice == "Player Name":
        filtered = filtered.sort_values(
            "player",
            ascending=True,
        )

    elif sort_choice == "Sportsbook":
        filtered = filtered.sort_values(
            [
                "platform",
                "tier_rank",
                "absolute_edge",
            ],
            ascending=[
                True,
                True,
                False,
            ],
        )

    else:
        filtered = filtered.sort_values(
            [
                "tier_rank",
                "absolute_edge",
            ],
            ascending=[
                True,
                False,
            ],
        )

    # Keep one best prop per player after all filters and sorting.
    filtered = filtered.drop_duplicates(
        subset=["player_key"],
        keep="first",
    ).reset_index(drop=True)

    available_props = len(filtered)

    if available_props == 0:
        st.warning(
            "No props match the selected filters."
        )
        max_props = 0

    elif available_props == 1:
        max_props = 1
        st.caption(
            "Showing 1 available prop."
        )

    else:
        max_props = st.slider(
            "Number of props to display",
            min_value=1,
            max_value=min(
                50,
                available_props,
            ),
            value=min(
                10,
                available_props,
            ),
        )

        st.caption(
            f"{available_props} unique players match "
            "the selected filters."
        )

    filtered = filtered.head(max_props)

    for rank, (_, row) in enumerate(
        filtered.iterrows(),
        start=1,
    ):
        tier = clean_text(row.get("tier")) or "LEAN"
        css_class = tier_class(tier)

        player = clean_text(row.get("player")) or "Unknown Player"
        platform = clean_text(row.get("platform")) or "Unknown Sportsbook"
        market = clean_text(row.get("market_display"))
        pick = clean_text(row.get("pick")).upper()

        line = safe_number(
            row.get("line"),
            decimals=1,
        )

        projection = safe_number(
            row.get("projection"),
            decimals=2,
        )

        edge = safe_number(
            row.get("edge"),
            decimals=2,
        )

        odds = safe_odds(
            row.get("sportsbook_odds")
        )

        matchup_text = build_matchup_text(row)
        game_time = format_commence_time(
            row.get("commence_time")
        )

        if pick == "MORE/YES":
            pick_display = "MORE"
            pick_class = "pick-more"
        elif pick == "LESS/NO":
            pick_display = "LESS"
            pick_class = "pick-less"
        else:
            pick_display = pick
            pick_class = "pick-less"

        detail_parts = [
            platform,
            market,
        ]

        if odds != "N/A":
            detail_parts.append(odds)

        detail_text = " · ".join(
            escape(part)
            for part in detail_parts
            if part
        )

        matchup_display = escape(matchup_text)

        if game_time:
            if matchup_display:
                matchup_display += f" · {escape(game_time)}"
            else:
                matchup_display = escape(game_time)

        st.markdown(
            f"""
            <div class="prop-card {css_class}">
                <div class="tier-label">
                    #{rank} · {escape(tier)}
                </div>
                <div class="player-name">
                    {escape(player)}
                </div>
                <div class="prop-detail">
                    {detail_text}
                </div>
                <div class="prop-detail">
                    {matchup_display}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_1, col_2, col_3, col_4 = st.columns(4)

        with col_1:
            st.markdown(
                f'<div class="{pick_class}">'
                f"{escape(pick_display)} {escape(line)}"
                "</div>",
                unsafe_allow_html=True,
            )

        with col_2:
            st.metric(
                "Model Projection",
                projection,
            )

        with col_3:
            st.metric(
                "Projection Edge",
                edge,
            )

        with col_4:
            if (
                "win_probability" in row.index
                and pd.notna(row.get("win_probability"))
            ):
                probability = pd.to_numeric(
                    row.get("win_probability"),
                    errors="coerce",
                )

                if pd.notna(probability):
                    if probability <= 1:
                        probability *= 100

                    st.metric(
                        "Win Probability",
                        f"{probability:.1f}%",
                    )
                else:
                    st.metric(
                        "Win Probability",
                        "Pending",
                    )
            else:
                st.metric(
                    "Win Probability",
                    "Pending calibration",
                )

        with st.expander("View model details"):
            detail_1, detail_2, detail_3 = st.columns(3)

            with detail_1:
                st.write(
                    f"**Sportsbook:** {platform}"
                )
                st.write(
                    f"**Sportsbook odds:** {odds}"
                )

            with detail_2:
                st.write(
                    f"**Market:** {market}"
                )
                st.write(
                    f"**Sportsbook line:** {line}"
                )

            with detail_3:
                st.write(
                    f"**Model projection:** {projection}"
                )
                st.write(
                    f"**Model edge:** {edge}"
                )

            st.write(
                f"**Recommended pick:** "
                f"{pick_display} {line}"
            )

            if matchup_text:
                st.write(
                    f"**Matchup:** {matchup_text}"
                )

            if game_time:
                st.write(
                    f"**Scheduled time:** {game_time}"
                )


# --------------------------------------------------
# Full card table
# --------------------------------------------------
if not props.empty:
    st.divider()
    st.header("📋 Full Actionable Card")

    table_columns = [
        "grade",
        "platform",
        "player",
        "market_display",
        "pick",
        "line",
        "sportsbook_odds",
        "projection",
        "edge",
        "team",
        "opponent",
    ]

    table_columns = [
        column
        for column in table_columns
        if column in props.columns
    ]

    display_table = props[table_columns].copy()

    display_table = display_table.rename(
        columns={
            "grade": "Grade",
            "platform": "Sportsbook",
            "player": "Player",
            "market_display": "Market",
            "pick": "Pick",
            "line": "Line",
            "sportsbook_odds": "Odds",
            "projection": "Projection",
            "edge": "Edge",
            "team": "Team",
            "opponent": "Opponent",
        }
    )

    for column in [
        "Line",
        "Projection",
        "Edge",
    ]:
        if column in display_table.columns:
            display_table[column] = pd.to_numeric(
                display_table[column],
                errors="coerce",
            ).round(2)

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------
# Performance section
# --------------------------------------------------
st.divider()
st.header("📈 Model Performance")

if graded.empty:
    st.info(
        "No graded live selections are available yet."
    )

else:
    tab_market, tab_platform, tab_history = st.tabs(
        [
            "By Market",
            "By Sportsbook",
            "Pick History",
        ]
    )

    with tab_market:
        market_summary_rows = []

        for market_name, group in graded.groupby(
            "market",
            dropna=False,
        ):
            summary = build_summary(group).to_dict()
            summary["Market"] = market_name
            market_summary_rows.append(summary)

        market_summary = pd.DataFrame(
            market_summary_rows
        )

        summary_columns = [
            "Market",
            "Picks",
            "Wins",
            "Losses",
            "Pushes",
            "Hit Rate",
            "Units",
        ]

        summary_columns = [
            column
            for column in summary_columns
            if column in market_summary.columns
        ]

        st.dataframe(
            market_summary[summary_columns],
            use_container_width=True,
            hide_index=True,
        )

    with tab_platform:
        platform_summary_rows = []

        for platform_name, group in graded.groupby(
            "platform",
            dropna=False,
        ):
            summary = build_summary(group).to_dict()
            summary["Sportsbook"] = platform_name
            platform_summary_rows.append(summary)

        platform_summary = pd.DataFrame(
            platform_summary_rows
        )

        summary_columns = [
            "Sportsbook",
            "Picks",
            "Wins",
            "Losses",
            "Pushes",
            "Hit Rate",
            "Units",
        ]

        summary_columns = [
            column
            for column in summary_columns
            if column in platform_summary.columns
        ]

        st.dataframe(
            platform_summary[summary_columns],
            use_container_width=True,
            hide_index=True,
        )

    with tab_history:
        history_columns = [
            "date",
            "platform",
            "player",
            "market",
            "line",
            "sportsbook_odds",
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

        history = graded[history_columns].copy()

        if "date" in history.columns:
            history["date"] = pd.to_datetime(
                history["date"],
                errors="coerce",
            )

            history = history.sort_values(
                "date",
                ascending=False,
            )

        st.dataframe(
            history,
            use_container_width=True,
            hide_index=True,
        )


st.divider()

st.caption(
    "Model projections are experimental and are not financial advice. "
    "Win probabilities, fair odds, and expected value should only be "
    "displayed after proper historical calibration."
)
