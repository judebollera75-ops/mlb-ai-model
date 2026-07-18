"""Streamlit dashboard for the production MLB prop model."""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent

PROP_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "mlb_daily_card.csv"
)

AUDIT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "mlb_daily_card_audit.csv"
)

HISTORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "history"
    / "mlb_bet_results.csv"
)

BACKTEST_SUMMARY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "backtesting"
    / "mlb_backtest_summary.csv"
)

BACKTEST_MARKET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "backtesting"
    / "mlb_backtest_by_market.csv"
)

BACKTEST_PLATFORM_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "backtesting"
    / "mlb_backtest_by_platform.csv"
)

BACKTEST_CONFIDENCE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "backtesting"
    / "mlb_backtest_by_confidence.csv"
)

BACKTEST_CALIBRATION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "backtesting"
    / "mlb_backtest_by_probability_bucket.csv"
)

BANKROLL_CURVE_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "backtesting"
    / "mlb_backtest_bankroll_curve.csv"
)


st.set_page_config(
    page_title="Jude's Sports Model",
    page_icon="📊",
    layout="wide",
)


st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1500px;
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

        .elite {
            border-left: 7px solid #16a34a;
        }

        .strong {
            border-left: 7px solid #2563eb;
        }

        .good {
            border-left: 7px solid #eab308;
        }

        .playable {
            border-left: 7px solid #f97316;
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

        .pick-over {
            font-size: 1.1rem;
            font-weight: 800;
            color: #16a34a;
        }

        .pick-under {
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


@st.cache_data(ttl=60)
def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV safely."""
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
    """Return a readable local file modification time."""
    if not path.exists():
        return "Not available"

    updated_datetime = datetime.fromtimestamp(
        path.stat().st_mtime
    )

    return updated_datetime.strftime(
        "%B %d, %Y at %I:%M %p"
    )


def clean_text(value: Any) -> str:
    """Return clean display text."""
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip()

    if text.casefold() == "nan":
        return ""

    return text


def safe_number(
    value: Any,
    decimals: int = 2,
) -> str:
    """Format a numeric value safely."""
    number = pd.to_numeric(
        value,
        errors="coerce",
    )

    if pd.isna(number):
        return "N/A"

    return f"{float(number):.{decimals}f}"


def safe_percentage(
    value: Any,
    decimals: int = 1,
) -> str:
    """Format decimal probability as a percentage."""
    number = pd.to_numeric(
        value,
        errors="coerce",
    )

    if pd.isna(number):
        return "N/A"

    number = float(number)

    if abs(number) <= 1.5:
        number *= 100.0

    return f"{number:.{decimals}f}%"


def safe_odds(value: Any) -> str:
    """Format American odds."""
    odds = pd.to_numeric(
        value,
        errors="coerce",
    )

    if pd.isna(odds):
        return "N/A"

    odds = int(round(float(odds)))

    return f"+{odds}" if odds > 0 else str(odds)


def format_market(value: Any) -> str:
    """Convert normalized market key to display text."""
    market_map = {
        "hitter_hits": "Hitter Hits",
        "hitter_total_bases": "Hitter Total Bases",
        "hitter_runs": "Hitter Runs",
        "hitter_rbis": "Hitter RBIs",
        "hitter_hits_runs_rbis": "Hits + Runs + RBIs",
        "hitter_fantasy_score": "Hitter Fantasy Score",
        "pitcher_strikeouts": "Pitcher Strikeouts",
        "pitcher_outs": "Pitcher Outs",
        "pitcher_fantasy_score": "Pitcher Fantasy Score",
    }

    cleaned = clean_text(value)

    return market_map.get(
        cleaned,
        cleaned.replace("_", " ").title(),
    )


def normalize_direction(value: Any) -> str:
    """Normalize pick direction for display."""
    cleaned = clean_text(value).casefold()

    if cleaned in {
        "over",
        "more",
        "yes",
        "higher",
        "more/yes",
    }:
        return "Over"

    if cleaned in {
        "under",
        "less",
        "no",
        "lower",
        "less/no",
    }:
        return "Under"

    return clean_text(value).title()


def confidence_rank(value: Any) -> int:
    """Return sort order for confidence tiers."""
    return {
        "Elite": 1,
        "Strong": 2,
        "Good": 3,
        "Playable": 4,
    }.get(clean_text(value).title(), 99)


def confidence_css(value: Any) -> str:
    """Return card CSS class."""
    return {
        "Elite": "elite",
        "Strong": "strong",
        "Good": "good",
        "Playable": "playable",
    }.get(
        clean_text(value).title(),
        "playable",
    )


def build_matchup_text(row: pd.Series) -> str:
    """Build readable matchup text."""
    team = clean_text(row.get("team"))
    opponent = clean_text(row.get("opponent"))

    if team and opponent:
        return f"{team} vs. {opponent}"

    home_team = clean_text(row.get("home_team"))
    away_team = clean_text(row.get("away_team"))

    if away_team and home_team:
        return f"{away_team} at {home_team}"

    return team or opponent


def format_commence_time(value: Any) -> str:
    """Convert UTC event time to Central Time."""
    if value is None or pd.isna(value):
        return ""

    parsed = pd.to_datetime(
        value,
        errors="coerce",
        utc=True,
    )

    if pd.isna(parsed):
        return clean_text(value)

    return parsed.tz_convert(
        "America/Chicago"
    ).strftime(
        "%b %d at %I:%M %p CT"
    )


def prepare_props(
    props: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize the production daily-card schema."""
    if props.empty:
        return props

    prepared = props.copy()

    required_defaults = {
        "grade": pd.NA,
        "confidence_tier": pd.NA,
        "platform": pd.NA,
        "player": pd.NA,
        "market": pd.NA,
        "direction": pd.NA,
        "line": pd.NA,
        "sportsbook_odds": pd.NA,
        "projection": pd.NA,
        "raw_projection_edge": pd.NA,
        "probability": pd.NA,
        "probability_edge": pd.NA,
        "expected_value": pd.NA,
        "recommended_bankroll_fraction": pd.NA,
    }

    for column, default_value in required_defaults.items():
        if column not in prepared.columns:
            prepared[column] = default_value

    numeric_columns = [
        "line",
        "sportsbook_odds",
        "projection",
        "raw_projection_edge",
        "probability",
        "sportsbook_implied_probability",
        "no_vig_implied_probability",
        "probability_edge",
        "expected_value",
        "fair_odds",
        "kelly_fraction",
        "recommended_bankroll_fraction",
        "validation_mae",
        "calibration_sample_size",
    ]

    for column in numeric_columns:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(
                prepared[column],
                errors="coerce",
            )

    prepared["direction"] = prepared[
        "direction"
    ].apply(normalize_direction)

    prepared["confidence_tier"] = prepared[
        "confidence_tier"
    ].astype("string").str.title()

    prepared["confidence_rank"] = prepared[
        "confidence_tier"
    ].apply(confidence_rank)

    prepared["market_display"] = prepared[
        "market"
    ].apply(format_market)

    prepared["player_key"] = (
        prepared["player"]
        .astype("string")
        .str.casefold()
        .str.strip()
    )

    prepared = prepared.dropna(
        subset=[
            "player",
            "market",
            "platform",
            "line",
            "projection",
            "probability",
        ]
    ).copy()

    prepared = prepared.loc[
        prepared["confidence_tier"].isin(
            {
                "Elite",
                "Strong",
                "Good",
                "Playable",
            }
        )
    ].copy()

    prepared = prepared.sort_values(
        [
            "confidence_rank",
            "expected_value",
            "probability_edge",
            "probability",
        ],
        ascending=[
            True,
            False,
            False,
            False,
        ],
    )

    return prepared.reset_index(drop=True)


def american_odds_profit(
    odds: Any,
    stake: float = 1.0,
) -> float:
    """Return unit profit for a winning American-odds wager."""
    numeric_odds = pd.to_numeric(odds, errors="coerce")
    if pd.isna(numeric_odds):
        return 0.0
    numeric_odds = float(numeric_odds)
    stake = float(stake)
    if numeric_odds > 0:
        return stake * numeric_odds / 100.0
    return stake * 100.0 / abs(numeric_odds)


def prepare_history(
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize, settle, and deduplicate history for dashboard metrics."""
    if history.empty:
        return history

    prepared = history.copy()
    if "outcome" not in prepared.columns:
        return pd.DataFrame()

    for column in [
        "outcome", "grade", "confidence_tier", "platform",
        "player", "market", "direction",
    ]:
        if column in prepared.columns:
            prepared[column] = prepared[column].astype("string").str.strip()

    prepared["outcome"] = prepared["outcome"].str.upper()
    prepared = prepared.loc[
        prepared["outcome"].isin({"WIN", "LOSS", "PUSH"})
    ].copy()

    for column in [
        "profit", "stake", "sportsbook_odds", "probability",
        "expected_value", "probability_edge", "line",
        "projection", "actual_result",
    ]:
        if column in prepared.columns:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    if "stake" not in prepared.columns:
        prepared["stake"] = 1.0
    else:
        prepared["stake"] = prepared["stake"].fillna(1.0)

    if "profit" not in prepared.columns:
        prepared["profit"] = pd.NA

    missing_profit = prepared["profit"].isna()
    win_mask = missing_profit & prepared["outcome"].eq("WIN")
    if "sportsbook_odds" in prepared.columns:
        prepared.loc[win_mask, "profit"] = prepared.loc[win_mask].apply(
            lambda row: american_odds_profit(
                row.get("sportsbook_odds"), row.get("stake", 1.0)
            ),
            axis=1,
        )

    loss_mask = missing_profit & prepared["outcome"].eq("LOSS")
    prepared.loc[loss_mask, "profit"] = -prepared.loc[loss_mask, "stake"]
    push_mask = missing_profit & prepared["outcome"].eq("PUSH")
    prepared.loc[push_mask, "profit"] = 0.0

    duplicate_columns = [
        c for c in ["event_date", "player", "market", "direction", "line"]
        if c in prepared.columns
    ]
    if duplicate_columns:
        sort_columns = [
            c for c in ["expected_value", "probability_edge", "probability"]
            if c in prepared.columns
        ]
        if sort_columns:
            prepared = prepared.sort_values(
                sort_columns, ascending=[False] * len(sort_columns),
                na_position="last",
            )
        prepared = prepared.drop_duplicates(
            subset=duplicate_columns, keep="first"
        )

    if "probability" in prepared.columns:
        prepared["Probability Bucket"] = pd.cut(
            prepared["probability"],
            bins=[0.00, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00],
            labels=[
                "Below 60%", "60–65%", "65–70%", "70–75%",
                "75–80%", "80–85%", "85–90%", "90%+",
            ],
            include_lowest=True,
        )

    return prepared.reset_index(drop=True)


def summarize_results(
    group: pd.DataFrame,
) -> pd.Series:
    """Calculate simple dashboard performance metrics."""
    wins = int(
        group["outcome"].eq("WIN").sum()
    )

    losses = int(
        group["outcome"].eq("LOSS").sum()
    )

    pushes = int(
        group["outcome"].eq("PUSH").sum()
    )

    settled = wins + losses

    hit_rate = (
        wins / settled * 100.0
        if settled
        else 0.0
    )

    total_profit = (
        pd.to_numeric(
            group.get(
                "profit",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )

    total_staked = (
        pd.to_numeric(
            group.get(
                "stake",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        )
        .fillna(1.0)
        .sum()
    )

    roi = (
        total_profit / total_staked * 100.0
        if total_staked > 0
        else 0.0
    )

    return pd.Series(
        {
            "Picks": len(group),
            "Wins": wins,
            "Losses": losses,
            "Pushes": pushes,
            "Hit Rate": round(hit_rate, 1),
            "Profit": round(float(total_profit), 2),
            "ROI": round(float(roi), 1),
        }
    )


props = prepare_props(
    load_csv(PROP_PATH)
)

history = prepare_history(
    load_csv(HISTORY_PATH)
)

backtest_summary = load_csv(
    BACKTEST_SUMMARY_PATH
)


st.markdown(
    '<div class="main-title">📊 Jude’s Sports Model</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Platform-specific MLB projections, calibrated probabilities, "
    "expected value, and verified historical performance."
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    f'<div class="updated-time">'
    f"Card last updated: "
    f"{escape(get_file_updated_time(PROP_PATH))}"
    "</div>",
    unsafe_allow_html=True,
)


wins = (
    int(history["outcome"].eq("WIN").sum())
    if not history.empty
    else 0
)

losses = (
    int(history["outcome"].eq("LOSS").sum())
    if not history.empty
    else 0
)

pushes = (
    int(history["outcome"].eq("PUSH").sum())
    if not history.empty
    else 0
)

settled_bets = wins + losses

hit_rate = (
    wins / settled_bets * 100.0
    if settled_bets
    else 0.0
)

total_profit = 0.0

if not history.empty and "profit" in history.columns:
    total_profit = float(
        pd.to_numeric(
            history["profit"],
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )

roi = 0.0

if not backtest_summary.empty and "roi" in backtest_summary.columns:
    roi_value = pd.to_numeric(
        backtest_summary.iloc[0].get("roi"),
        errors="coerce",
    )

    if pd.notna(roi_value):
        roi = float(roi_value) * 100.0

metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)

metric_1.metric(
    "Current Plays",
    len(props),
)

metric_2.metric(
    "Settled Record",
    f"{wins}-{losses}-{pushes}",
)

metric_3.metric(
    "Hit Rate",
    f"{hit_rate:.1f}%",
)

metric_4.metric(
    "Tracked Profit",
    f"{total_profit:+.2f} units",
)

metric_5.metric(
    "Backtest ROI",
    f"{roi:+.1f}%",
)

st.caption(
    "Only picks logged before game time and graded from official "
    "box scores should count toward the verified record."
)

st.divider()


header_left, header_right = st.columns(
    [4, 1]
)

with header_left:
    st.header("🔥 Today’s MLB Props")

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
            No props currently pass every probability, price,
            freshness, calibration, and expected-value filter.<br><br>
            This is normal when the model does not find a trustworthy edge.
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
        .tolist()
    )

    markets = sorted(
        props["market_display"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    confidence_options = [
        tier
        for tier in [
            "Elite",
            "Strong",
            "Good",
            "Playable",
        ]
        if tier in set(props["confidence_tier"])
    ]

    directions = [
        direction
        for direction in [
            "Over",
            "Under",
        ]
        if direction in set(props["direction"])
    ]

    filter_1, filter_2, filter_3, filter_4 = st.columns(4)

    with filter_1:
        selected_platforms = st.multiselect(
            "Platform",
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
        selected_confidence = st.multiselect(
            "Confidence",
            options=confidence_options,
            default=confidence_options,
        )

    with filter_4:
        selected_directions = st.multiselect(
            "Direction",
            options=directions,
            default=directions,
        )

    sort_choice = st.selectbox(
        "Sort by",
        [
            "Best Overall",
            "Highest Probability",
            "Highest Expected Value",
            "Largest Probability Edge",
            "Player Name",
            "Platform",
        ],
    )

    filtered = props.copy()

    filtered = filtered.loc[
        filtered["platform"].isin(
            selected_platforms
        )
        & filtered["market_display"].isin(
            selected_markets
        )
        & filtered["confidence_tier"].isin(
            selected_confidence
        )
        & filtered["direction"].isin(
            selected_directions
        )
    ].copy()

    if sort_choice == "Highest Probability":
        filtered = filtered.sort_values(
            "probability",
            ascending=False,
        )

    elif sort_choice == "Highest Expected Value":
        filtered = filtered.sort_values(
            "expected_value",
            ascending=False,
        )

    elif sort_choice == "Largest Probability Edge":
        filtered = filtered.sort_values(
            "probability_edge",
            ascending=False,
        )

    elif sort_choice == "Player Name":
        filtered = filtered.sort_values(
            "player",
            ascending=True,
        )

    elif sort_choice == "Platform":
        filtered = filtered.sort_values(
            [
                "platform",
                "confidence_rank",
                "expected_value",
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
                "confidence_rank",
                "expected_value",
                "probability_edge",
                "probability",
            ],
            ascending=[
                True,
                False,
                False,
                False,
            ],
        )

    available_props = len(filtered)
    slider_key = "props_slider"

    if available_props == 0:
        st.warning(
            "No props match the selected filters."
        )

        # Remove any remembered slider value so it cannot become invalid
        # when the user changes filters and matching props return later.
        st.session_state.pop(
            slider_key,
            None,
        )

        filtered = filtered.iloc[0:0]

    else:
        slider_max = min(
            50,
            available_props,
        )

        default_slider_value = min(
            10,
            slider_max,
        )

        remembered_slider_value = st.session_state.get(
            slider_key
        )

        # Streamlit preserves widget state between reruns. Reset the
        # remembered value whenever the new filtered result has a smaller
        # valid range than the previous result.
        if (
            remembered_slider_value is None
            or remembered_slider_value < 1
            or remembered_slider_value > slider_max
        ):
            st.session_state[
                slider_key
            ] = default_slider_value

        max_props = st.slider(
            "Number of props to display",
            min_value=1,
            max_value=slider_max,
            step=1,
            key=slider_key,
        )

        st.caption(
            f"{available_props} props match the selected filters."
        )

        filtered = filtered.head(max_props)

    for rank, (_, row) in enumerate(
        filtered.iterrows(),
        start=1,
    ):
        confidence = (
            clean_text(
                row.get("confidence_tier")
            ).title()
            or "Playable"
        )

        css_class = confidence_css(
            confidence
        )

        player = (
            clean_text(row.get("player"))
            or "Unknown Player"
        )

        platform = (
            clean_text(row.get("platform"))
            or "Unknown Platform"
        )

        market = clean_text(
            row.get("market_display")
        )

        direction = normalize_direction(
            row.get("direction")
        )

        line = safe_number(
            row.get("line"),
            1,
        )

        projection = safe_number(
            row.get("projection"),
            2,
        )

        probability = safe_percentage(
            row.get("probability"),
            1,
        )

        probability_edge = safe_percentage(
            row.get("probability_edge"),
            1,
        )

        expected_value = safe_percentage(
            row.get("expected_value"),
            1,
        )

        bankroll_fraction = safe_percentage(
            row.get(
                "recommended_bankroll_fraction"
            ),
            2,
        )

        odds = safe_odds(
            row.get("sportsbook_odds")
        )

        matchup_text = build_matchup_text(
            row
        )

        game_time = format_commence_time(
            row.get("commence_time")
        )

        pick_class = (
            "pick-over"
            if direction == "Over"
            else "pick-under"
        )

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

        matchup_display = escape(
            matchup_text
        )

        if game_time:
            matchup_display = (
                f"{matchup_display} · "
                f"{escape(game_time)}"
                if matchup_display
                else escape(game_time)
            )

        st.markdown(
            f"""
            <div class="prop-card {css_class}">
                <div class="tier-label">
                    #{rank} · {escape(confidence.upper())}
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

        col_1, col_2, col_3, col_4, col_5 = st.columns(5)

        with col_1:
            st.markdown(
                f'<div class="{pick_class}">'
                f"{escape(direction.upper())} "
                f"{escape(line)}"
                "</div>",
                unsafe_allow_html=True,
            )

        with col_2:
            st.metric(
                "Projection",
                projection,
            )

        with col_3:
            st.metric(
                "Win Probability",
                probability,
            )

        with col_4:
            st.metric(
                "Probability Edge",
                probability_edge,
            )

        with col_5:
            st.metric(
                "Expected Value",
                expected_value,
            )

        with st.expander(
            "View model and pricing details"
        ):
            detail_1, detail_2, detail_3, detail_4 = st.columns(4)

            with detail_1:
                st.write(
                    f"**Platform:** {platform}"
                )
                st.write(
                    f"**Odds:** {odds}"
                )
                st.write(
                    f"**Fair odds:** "
                    f"{safe_odds(row.get('fair_odds'))}"
                )

            with detail_2:
                st.write(
                    f"**Market:** {market}"
                )
                st.write(
                    f"**Line:** {line}"
                )
                st.write(
                    f"**Projection:** {projection}"
                )

            with detail_3:
                st.write(
                    f"**Win probability:** "
                    f"{probability}"
                )
                st.write(
                    f"**Probability edge:** "
                    f"{probability_edge}"
                )
                st.write(
                    f"**Expected value:** "
                    f"{expected_value}"
                )

            with detail_4:
                st.write(
                    f"**Suggested bankroll fraction:** "
                    f"{bankroll_fraction}"
                )
                st.write(
                    f"**Calibration samples:** "
                    f"{safe_number(row.get('calibration_sample_size'), 0)}"
                )
                st.write(
                    f"**Validation MAE:** "
                    f"{safe_number(row.get('validation_mae'), 3)}"
                )

            distribution_method = clean_text(
                row.get("distribution_method")
            )

            if distribution_method:
                st.write(
                    f"**Probability method:** "
                    f"{distribution_method}"
                )

            if matchup_text:
                st.write(
                    f"**Matchup:** {matchup_text}"
                )

            if game_time:
                st.write(
                    f"**Scheduled time:** {game_time}"
                )


if not props.empty:
    st.divider()
    st.header("📋 Full Actionable Card")

    table_columns = [
        "grade",
        "confidence_tier",
        "platform",
        "player",
        "market_display",
        "direction",
        "line",
        "sportsbook_odds",
        "projection",
        "probability",
        "probability_edge",
        "expected_value",
        "recommended_bankroll_fraction",
        "team",
        "opponent",
    ]

    table_columns = [
        column
        for column in table_columns
        if column in props.columns
    ]

    display_table = props[
        table_columns
    ].copy()

    display_table = display_table.rename(
        columns={
            "grade": "Grade",
            "confidence_tier": "Confidence",
            "platform": "Platform",
            "player": "Player",
            "market_display": "Market",
            "direction": "Direction",
            "line": "Line",
            "sportsbook_odds": "Odds",
            "projection": "Projection",
            "probability": "Win Probability",
            "probability_edge": "Probability Edge",
            "expected_value": "Expected Value",
            "recommended_bankroll_fraction": (
                "Suggested Bankroll Fraction"
            ),
            "team": "Team",
            "opponent": "Opponent",
        }
    )

    for column in [
        "Line",
        "Projection",
    ]:
        if column in display_table.columns:
            display_table[column] = pd.to_numeric(
                display_table[column],
                errors="coerce",
            ).round(2)

    for column in [
        "Win Probability",
        "Probability Edge",
        "Expected Value",
        "Suggested Bankroll Fraction",
    ]:
        if column in display_table.columns:
            display_table[column] = (
                pd.to_numeric(
                    display_table[column],
                    errors="coerce",
                )
                * 100.0
            ).round(2)

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
    )


st.divider()
st.header("📈 Verified Model Performance")

if history.empty:
    st.info(
        "No settled live recommendations are available yet."
    )

else:
    (
        tab_overall,
        tab_market,
        tab_platform,
        tab_confidence,
        tab_probability,
        tab_grade,
        tab_market_analysis,
        tab_optimizer,
        tab_history,
    ) = st.tabs(
        [
            "Overall",
            "By Market",
            "By Platform",
            "By Confidence",
            "By Probability",
            "📊 Grade Analysis",
            "📈 Market Analysis",
            "🧠 Threshold Optimizer",
            "Pick History",
        ]
    )

    with tab_overall:
        overall_summary = summarize_results(history)
        summary_1, summary_2, summary_3, summary_4 = st.columns(4)
        summary_1.metric("Verified Picks", int(overall_summary["Picks"]))
        summary_2.metric(
            "Record",
            f"{int(overall_summary['Wins'])}-{int(overall_summary['Losses'])}-{int(overall_summary['Pushes'])}",
        )
        summary_3.metric("Hit Rate", f"{float(overall_summary['Hit Rate']):.1f}%")
        summary_4.metric("ROI", f"{float(overall_summary['ROI']):+.1f}%")
        overall_table = pd.DataFrame([
            {"Category": "All Verified Picks", **overall_summary.to_dict()}
        ])
        st.dataframe(overall_table, use_container_width=True, hide_index=True)
        st.caption(
            "Repeated copies of the same player, market, direction, line, "
            "and event are counted once."
        )

    with tab_market:
        market_rows = []

        for market_name, group in history.groupby(
            "market",
            dropna=False,
        ):
            summary = summarize_results(
                group
            ).to_dict()

            summary["Market"] = format_market(
                market_name
            )

            market_rows.append(summary)

        market_summary = pd.DataFrame(
            market_rows
        )

        st.dataframe(
            market_summary,
            use_container_width=True,
            hide_index=True,
        )

    with tab_platform:
        platform_rows = []

        for platform_name, group in history.groupby(
            "platform",
            dropna=False,
        ):
            summary = summarize_results(
                group
            ).to_dict()

            summary["Platform"] = platform_name

            platform_rows.append(summary)

        platform_summary = pd.DataFrame(
            platform_rows
        )

        st.dataframe(
            platform_summary,
            use_container_width=True,
            hide_index=True,
        )

    with tab_confidence:
        confidence_rows = []

        if "confidence_tier" in history.columns:
            for confidence_name, group in history.groupby(
                "confidence_tier",
                dropna=False,
            ):
                summary = summarize_results(
                    group
                ).to_dict()

                summary["Confidence"] = confidence_name

                confidence_rows.append(summary)

        confidence_summary = pd.DataFrame(
            confidence_rows
        )

        if confidence_summary.empty:
            st.info(
                "No confidence-tier history is available yet."
            )
        else:
            st.dataframe(
                confidence_summary,
                use_container_width=True,
                hide_index=True,
            )

    with tab_probability:
        probability_rows = []
        if "Probability Bucket" in history.columns:
            for bucket_name, group in history.groupby(
                "Probability Bucket", dropna=False, observed=False
            ):
                if group.empty:
                    continue
                summary = summarize_results(group).to_dict()
                summary["Probability Bucket"] = clean_text(bucket_name) or "Unclassified"
                average_probability = pd.to_numeric(
                    group.get("probability", pd.Series(dtype=float)),
                    errors="coerce",
                ).mean()
                summary["Average Model Probability"] = (
                    round(float(average_probability) * 100.0, 1)
                    if pd.notna(average_probability) else pd.NA
                )
                probability_rows.append(summary)

        probability_summary = pd.DataFrame(probability_rows)
        if probability_summary.empty:
            st.info("No probability-bucket history is available yet.")
        else:
            probability_order = {
                "Below 60%": 1, "60–65%": 2, "65–70%": 3,
                "70–75%": 4, "75–80%": 5, "80–85%": 6,
                "85–90%": 7, "90%+": 8, "Unclassified": 9,
            }
            probability_summary["_sort"] = (
                probability_summary["Probability Bucket"]
                .map(probability_order)
                .fillna(99)
            )
            probability_summary = (
                probability_summary.sort_values("_sort").drop(columns="_sort")
            )
            st.dataframe(
                probability_summary, use_container_width=True, hide_index=True
            )
            if {"Average Model Probability", "Hit Rate"}.issubset(
                probability_summary.columns
            ):
                st.subheader("Predicted Probability vs. Actual Hit Rate")
                st.line_chart(
                    probability_summary[[
                        "Probability Bucket",
                        "Average Model Probability",
                        "Hit Rate",
                    ]].set_index("Probability Bucket")
                )
            st.caption(
                "A well-calibrated model should have actual hit rates that "
                "roughly follow its predicted probability buckets."
            )

    with tab_grade:
        if "grade" not in history.columns:
            st.info("No grade history is available yet.")
        else:
            grade_rows = []
            for grade_name, group in history.groupby("grade", dropna=False):
                if group.empty:
                    continue
                summary = summarize_results(group).to_dict()
                summary["Grade"] = clean_text(grade_name) or "Unclassified"

                for source, label in [
                    ("probability", "Average Probability"),
                    ("probability_edge", "Average Edge"),
                    ("expected_value", "Average EV"),
                ]:
                    values = pd.to_numeric(
                        group.get(source, pd.Series(dtype=float)),
                        errors="coerce",
                    )
                    average = values.mean()
                    summary[label] = (
                        round(float(average) * 100.0, 2)
                        if pd.notna(average)
                        else pd.NA
                    )

                grade_rows.append(summary)

            grade_summary = pd.DataFrame(grade_rows)
            if grade_summary.empty:
                st.info("No grade history is available yet.")
            else:
                grade_order = {"A+": 1, "A": 2, "B+": 3, "B": 4}
                grade_summary["_sort"] = (
                    grade_summary["Grade"].map(grade_order).fillna(99)
                )
                grade_summary = grade_summary.sort_values("_sort").drop(
                    columns="_sort"
                )
                st.dataframe(
                    grade_summary,
                    use_container_width=True,
                    hide_index=True,
                )
                st.subheader("Hit Rate by Grade")
                st.bar_chart(grade_summary.set_index("Grade")[["Hit Rate"]])

                grade_rates = grade_summary.set_index("Grade")["Hit Rate"]
                if "A+" in grade_rates.index and "A" in grade_rates.index:
                    a_plus = float(grade_rates.loc["A+"])
                    a_rate = float(grade_rates.loc["A"])
                    if a_plus < a_rate:
                        st.warning(
                            "A+ is currently underperforming A. "
                            f"A+: {a_plus:.1f}% vs. A: {a_rate:.1f}%. "
                            "Review the A+ probability, edge, and EV cutoffs."
                        )

    with tab_market_analysis:
        if "market" not in history.columns:
            st.info("No market history is available.")
        else:
            market_rows = []
            for market_name, group in history.groupby("market", dropna=False):
                summary = summarize_results(group).to_dict()
                summary["Market"] = format_market(market_name)

                for source, label in [
                    ("expected_value", "Average EV"),
                    ("probability", "Average Probability"),
                ]:
                    values = pd.to_numeric(
                        group.get(source, pd.Series(dtype=float)),
                        errors="coerce",
                    )
                    average = values.mean()
                    summary[label] = (
                        round(float(average) * 100.0, 2)
                        if pd.notna(average)
                        else pd.NA
                    )

                market_rows.append(summary)

            market_df = pd.DataFrame(market_rows)
            if market_df.empty:
                st.info("No market history is available.")
            else:
                market_df = market_df.sort_values("Profit", ascending=False)
                st.subheader("Market Performance")
                st.dataframe(
                    market_df,
                    use_container_width=True,
                    hide_index=True,
                )
                chart_df = market_df.set_index("Market")
                st.subheader("Profit by Market")
                st.bar_chart(chart_df[["Profit"]])
                st.subheader("Hit Rate by Market")
                st.bar_chart(chart_df[["Hit Rate"]])

    with tab_optimizer:
        st.subheader("Historical Threshold Optimizer")
        st.caption(
            "The optimizer now tunes cutoffs on older picks and checks them "
            "on newer, unseen picks when dates are available. This reduces "
            "the risk of choosing thresholds that only fit past noise."
        )

        required_optimizer_columns = {
            "probability",
            "probability_edge",
            "expected_value",
            "outcome",
            "profit",
            "stake",
        }
        missing_optimizer_columns = sorted(
            required_optimizer_columns - set(history.columns)
        )

        if missing_optimizer_columns:
            st.info(
                "The optimizer needs these history columns: "
                + ", ".join(missing_optimizer_columns)
            )
        else:
            optimizer_history = history.copy()
            for column in [
                "probability",
                "probability_edge",
                "expected_value",
                "profit",
                "stake",
            ]:
                optimizer_history[column] = pd.to_numeric(
                    optimizer_history[column], errors="coerce"
                )

            # Accept either decimal values (0.26) or percentages (26.0).
            for column in ["probability", "probability_edge", "expected_value"]:
                non_null = optimizer_history[column].dropna()
                if not non_null.empty and non_null.abs().median() > 1.5:
                    optimizer_history[column] = optimizer_history[column] / 100.0

            optimizer_history = optimizer_history.dropna(
                subset=[
                    "probability",
                    "probability_edge",
                    "expected_value",
                    "profit",
                    "stake",
                ]
            ).copy()

            available_markets = sorted(
                optimizer_history.get(
                    "market", pd.Series(dtype="string")
                ).dropna().astype(str).unique().tolist()
            )
            selected_optimizer_markets = st.multiselect(
                "Markets included in optimization",
                options=available_markets,
                default=available_markets,
                format_func=format_market,
                key="optimizer_markets",
            )

            if available_markets:
                optimizer_history = optimizer_history.loc[
                    optimizer_history["market"].isin(selected_optimizer_markets)
                ].copy()

            control_1, control_2, control_3, control_4 = st.columns(4)
            with control_1:
                minimum_sample_size = st.number_input(
                    "Minimum training picks",
                    min_value=10,
                    max_value=250,
                    value=30,
                    step=5,
                    key="optimizer_min_sample",
                )
            with control_2:
                minimum_validation_size = st.number_input(
                    "Minimum validation picks",
                    min_value=5,
                    max_value=100,
                    value=10,
                    step=5,
                    key="optimizer_min_validation",
                )
            with control_3:
                objective = st.selectbox(
                    "Primary objective",
                    ["Balanced", "Highest Hit Rate", "Highest ROI", "Highest Profit"],
                    key="optimizer_objective",
                )
            with control_4:
                show_top_n = st.slider(
                    "Results to display",
                    min_value=5,
                    max_value=40,
                    value=15,
                    step=5,
                    key="optimizer_top_n",
                )

            if optimizer_history.empty:
                st.info("No settled picks remain for the selected markets.")
            else:
                # Use chronological holdout when possible.
                has_dates = "event_date" in optimizer_history.columns
                if has_dates:
                    optimizer_history["event_date"] = pd.to_datetime(
                        optimizer_history["event_date"], errors="coerce"
                    )
                    optimizer_history = optimizer_history.sort_values(
                        "event_date", na_position="last"
                    ).reset_index(drop=True)

                split_index = max(
                    minimum_sample_size,
                    int(len(optimizer_history) * 0.70),
                )
                split_index = min(split_index, len(optimizer_history))

                if (
                    has_dates
                    and len(optimizer_history) - split_index >= minimum_validation_size
                ):
                    training_history = optimizer_history.iloc[:split_index].copy()
                    validation_history = optimizer_history.iloc[split_index:].copy()
                    validation_mode = True
                else:
                    training_history = optimizer_history.copy()
                    validation_history = pd.DataFrame()
                    validation_mode = False

                range_1, range_2, range_3 = st.columns(3)
                range_1.metric(
                    "Observed Probability Range",
                    f"{training_history['probability'].min() * 100:.1f}%–"
                    f"{training_history['probability'].max() * 100:.1f}%",
                )
                range_2.metric(
                    "Observed Edge Range",
                    f"{training_history['probability_edge'].min() * 100:.1f}%–"
                    f"{training_history['probability_edge'].max() * 100:.1f}%",
                )
                range_3.metric(
                    "Observed EV Range",
                    f"{training_history['expected_value'].min() * 100:.1f}%–"
                    f"{training_history['expected_value'].max() * 100:.1f}%",
                )

                probability_thresholds = [value / 100.0 for value in range(50, 91, 2)]
                edge_thresholds = [value / 100.0 for value in range(0, 41, 2)]
                ev_thresholds = [value / 100.0 for value in range(0, 61, 5)]

                optimizer_rows = []
                for probability_cutoff in probability_thresholds:
                    probability_frame = training_history.loc[
                        training_history["probability"] >= probability_cutoff
                    ]
                    if len(probability_frame) < minimum_sample_size:
                        continue

                    for edge_cutoff in edge_thresholds:
                        edge_frame = probability_frame.loc[
                            probability_frame["probability_edge"] >= edge_cutoff
                        ]
                        if len(edge_frame) < minimum_sample_size:
                            continue

                        for ev_cutoff in ev_thresholds:
                            filtered_training = edge_frame.loc[
                                edge_frame["expected_value"] >= ev_cutoff
                            ]
                            if len(filtered_training) < minimum_sample_size:
                                continue

                            training_summary = summarize_results(
                                filtered_training
                            ).to_dict()
                            settled = int(
                                training_summary["Wins"] + training_summary["Losses"]
                            )
                            if settled < minimum_sample_size:
                                continue

                            row = {
                                "Minimum Probability": probability_cutoff * 100.0,
                                "Minimum Edge": edge_cutoff * 100.0,
                                "Minimum EV": ev_cutoff * 100.0,
                                "Training Picks": training_summary["Picks"],
                                "Training Hit Rate": training_summary["Hit Rate"],
                                "Training Profit": training_summary["Profit"],
                                "Training ROI": training_summary["ROI"],
                                "Training Wins": training_summary["Wins"],
                                "Training Losses": training_summary["Losses"],
                                "Training Pushes": training_summary["Pushes"],
                            }

                            if validation_mode:
                                filtered_validation = validation_history.loc[
                                    (validation_history["probability"] >= probability_cutoff)
                                    & (validation_history["probability_edge"] >= edge_cutoff)
                                    & (validation_history["expected_value"] >= ev_cutoff)
                                ]
                                validation_summary = summarize_results(
                                    filtered_validation
                                ).to_dict()
                                row.update(
                                    {
                                        "Validation Picks": validation_summary["Picks"],
                                        "Validation Hit Rate": validation_summary["Hit Rate"],
                                        "Validation Profit": validation_summary["Profit"],
                                        "Validation ROI": validation_summary["ROI"],
                                        "Validation Wins": validation_summary["Wins"],
                                        "Validation Losses": validation_summary["Losses"],
                                        "Validation Pushes": validation_summary["Pushes"],
                                    }
                                )
                            optimizer_rows.append(row)

                optimizer_results = pd.DataFrame(optimizer_rows)
                if optimizer_results.empty:
                    st.info(
                        "No threshold combination meets the selected minimum "
                        "sample size. Lower the minimums or include more markets."
                    )
                else:
                    # Remove threshold rows that produce exactly the same pick set/results.
                    dedupe_columns = [
                        "Training Picks",
                        "Training Wins",
                        "Training Losses",
                        "Training Pushes",
                        "Training Profit",
                    ]
                    if validation_mode:
                        dedupe_columns += [
                            "Validation Picks",
                            "Validation Wins",
                            "Validation Losses",
                            "Validation Pushes",
                            "Validation Profit",
                        ]
                    optimizer_results = optimizer_results.drop_duplicates(
                        subset=dedupe_columns, keep="first"
                    ).copy()

                    score_hit_rate = (
                        optimizer_results["Validation Hit Rate"]
                        if validation_mode
                        else optimizer_results["Training Hit Rate"]
                    )
                    score_roi = (
                        optimizer_results["Validation ROI"]
                        if validation_mode
                        else optimizer_results["Training ROI"]
                    )
                    score_profit = (
                        optimizer_results["Validation Profit"]
                        if validation_mode
                        else optimizer_results["Training Profit"]
                    )
                    score_picks = (
                        optimizer_results["Validation Picks"]
                        if validation_mode
                        else optimizer_results["Training Picks"]
                    )

                    eligible = score_picks >= (
                        minimum_validation_size if validation_mode else minimum_sample_size
                    )
                    scored_results = optimizer_results.loc[eligible].copy()

                    if scored_results.empty:
                        st.info(
                            "Thresholds were found in training, but none produced "
                            "enough unseen validation picks. Lower the validation "
                            "minimum or collect more settled history."
                        )
                    else:
                        score_hit_rate = (
                            scored_results["Validation Hit Rate"]
                            if validation_mode
                            else scored_results["Training Hit Rate"]
                        )
                        score_roi = (
                            scored_results["Validation ROI"]
                            if validation_mode
                            else scored_results["Training ROI"]
                        )
                        score_profit = (
                            scored_results["Validation Profit"]
                            if validation_mode
                            else scored_results["Training Profit"]
                        )
                        score_picks = (
                            scored_results["Validation Picks"]
                            if validation_mode
                            else scored_results["Training Picks"]
                        )

                        max_profit = max(float(score_profit.max()), 1.0)
                        max_picks = max(float(score_picks.max()), 1.0)
                        scored_results["Balanced Score"] = (
                            score_hit_rate
                            + 0.25 * score_roi.clip(lower=-100, upper=100)
                            + 10.0 * score_profit / max_profit
                            + 5.0 * score_picks / max_picks
                        )

                        sort_map = {
                            "Balanced": ["Balanced Score", "Training Picks"],
                            "Highest Hit Rate": [
                                "Validation Hit Rate" if validation_mode else "Training Hit Rate",
                                "Validation Picks" if validation_mode else "Training Picks",
                            ],
                            "Highest ROI": [
                                "Validation ROI" if validation_mode else "Training ROI",
                                "Validation Picks" if validation_mode else "Training Picks",
                            ],
                            "Highest Profit": [
                                "Validation Profit" if validation_mode else "Training Profit",
                                "Validation Picks" if validation_mode else "Training Picks",
                            ],
                        }
                        scored_results = scored_results.sort_values(
                            sort_map[objective], ascending=[False, False]
                        ).reset_index(drop=True)

                        best = scored_results.iloc[0]
                        metric_a, metric_b, metric_c, metric_d = st.columns(4)
                        metric_a.metric(
                            "Recommended Probability",
                            f"{best['Minimum Probability']:.0f}%+",
                        )
                        metric_b.metric(
                            "Recommended Edge",
                            f"{best['Minimum Edge']:.0f}%+",
                        )
                        metric_c.metric(
                            "Recommended EV",
                            f"{best['Minimum EV']:.0f}%+",
                        )
                        metric_d.metric(
                            "Validation Mode",
                            "Chronological holdout" if validation_mode else "In-sample only",
                        )

                        if validation_mode:
                            result_a, result_b, result_c, result_d = st.columns(4)
                            result_a.metric(
                                "Unseen Hit Rate",
                                f"{best['Validation Hit Rate']:.1f}%",
                            )
                            result_b.metric(
                                "Unseen ROI",
                                f"{best['Validation ROI']:+.1f}%",
                            )
                            result_c.metric(
                                "Unseen Profit",
                                f"{best['Validation Profit']:+.2f}u",
                            )
                            result_d.metric(
                                "Unseen Record",
                                f"{int(best['Validation Wins'])}-"
                                f"{int(best['Validation Losses'])}-"
                                f"{int(best['Validation Pushes'])}",
                            )
                            st.caption(
                                f"Thresholds were selected using {len(training_history)} older "
                                f"picks and evaluated on {len(validation_history)} newer picks."
                            )
                        else:
                            result_a, result_b, result_c, result_d = st.columns(4)
                            result_a.metric(
                                "Historical Hit Rate",
                                f"{best['Training Hit Rate']:.1f}%",
                            )
                            result_b.metric(
                                "Historical ROI",
                                f"{best['Training ROI']:+.1f}%",
                            )
                            result_c.metric(
                                "Historical Profit",
                                f"{best['Training Profit']:+.2f}u",
                            )
                            result_d.metric(
                                "Historical Record",
                                f"{int(best['Training Wins'])}-"
                                f"{int(best['Training Losses'])}-"
                                f"{int(best['Training Pushes'])}",
                            )

                        display_optimizer = scored_results.head(show_top_n).copy()
                        display_optimizer = display_optimizer.drop(
                            columns=["Balanced Score"], errors="ignore"
                        )
                        for column in display_optimizer.columns:
                            if column not in {
                                "Training Wins", "Training Losses", "Training Pushes",
                                "Validation Wins", "Validation Losses", "Validation Pushes",
                            }:
                                display_optimizer[column] = pd.to_numeric(
                                    display_optimizer[column], errors="coerce"
                                )

                        st.subheader("Top Distinct Threshold Combinations")
                        st.dataframe(
                            display_optimizer,
                            use_container_width=True,
                            hide_index=True,
                        )

                        # Market-by-market diagnostic using a conservative sample floor.
                        if "market" in optimizer_history.columns:
                            market_recommendations = []
                            for market_name, market_group in optimizer_history.groupby("market"):
                                if len(market_group) < 15:
                                    continue
                                market_best = None
                                for probability_cutoff in probability_thresholds:
                                    for edge_cutoff in edge_thresholds:
                                        filtered = market_group.loc[
                                            (market_group["probability"] >= probability_cutoff)
                                            & (market_group["probability_edge"] >= edge_cutoff)
                                        ]
                                        if len(filtered) < 10:
                                            continue
                                        summary = summarize_results(filtered).to_dict()
                                        candidate = {
                                            "Market": format_market(market_name),
                                            "Minimum Probability": probability_cutoff * 100.0,
                                            "Minimum Edge": edge_cutoff * 100.0,
                                            **summary,
                                        }
                                        if market_best is None or (
                                            candidate["Hit Rate"], candidate["ROI"], candidate["Picks"]
                                        ) > (
                                            market_best["Hit Rate"], market_best["ROI"], market_best["Picks"]
                                        ):
                                            market_best = candidate
                                if market_best is not None:
                                    market_recommendations.append(market_best)

                            if market_recommendations:
                                st.subheader("Exploratory Market-Specific Cutoffs")
                                st.dataframe(
                                    pd.DataFrame(market_recommendations).sort_values(
                                        ["Hit Rate", "ROI"], ascending=False
                                    ),
                                    use_container_width=True,
                                    hide_index=True,
                                )

                        st.warning(
                            "Do not automatically replace production thresholds from "
                            "this screen alone. Even chronological validation can be noisy "
                            "with small samples. Promote a cutoff only after it remains "
                            "profitable across additional unseen picks."
                        )

    with tab_history:
        history_columns = [
            "event_date",
            "platform",
            "player",
            "market",
            "direction",
            "line",
            "sportsbook_odds",
            "projection",
            "probability",
            "actual_result",
            "outcome",
            "profit",
        ]

        history_columns = [
            column
            for column in history_columns
            if column in history.columns
        ]

        history_table = history[
            history_columns
        ].copy()

        if "event_date" in history_table.columns:
            history_table["event_date"] = pd.to_datetime(
                history_table["event_date"],
                errors="coerce",
            )

            history_table = history_table.sort_values(
                "event_date",
                ascending=False,
            )

        if "market" in history_table.columns:
            history_table["market"] = history_table[
                "market"
            ].apply(format_market)

        st.dataframe(
            history_table,
            use_container_width=True,
            hide_index=True,
        )


st.divider()
st.header("🧪 Backtest and Calibration Reports")

backtest_market = load_csv(
    BACKTEST_MARKET_PATH
)

backtest_platform = load_csv(
    BACKTEST_PLATFORM_PATH
)

backtest_confidence = load_csv(
    BACKTEST_CONFIDENCE_PATH
)

backtest_calibration = load_csv(
    BACKTEST_CALIBRATION_PATH
)

bankroll_curve = load_csv(
    BANKROLL_CURVE_PATH
)

report_tab_1, report_tab_2, report_tab_3, report_tab_4 = st.tabs(
    [
        "Backtest Summary",
        "Calibration",
        "Bankroll Curve",
        "Diagnostics",
    ]
)

with report_tab_1:
    if backtest_summary.empty:
        st.info(
            "Backtest reports will appear after settled picks exist."
        )
    else:
        st.dataframe(
            backtest_summary,
            use_container_width=True,
            hide_index=True,
        )

        if not backtest_market.empty:
            st.subheader(
                "Performance by Market"
            )

            st.dataframe(
                backtest_market,
                use_container_width=True,
                hide_index=True,
            )

        if not backtest_platform.empty:
            st.subheader(
                "Performance by Platform"
            )

            st.dataframe(
                backtest_platform,
                use_container_width=True,
                hide_index=True,
            )

        if not backtest_confidence.empty:
            st.subheader(
                "Performance by Confidence"
            )

            st.dataframe(
                backtest_confidence,
                use_container_width=True,
                hide_index=True,
            )

with report_tab_2:
    if backtest_calibration.empty:
        st.info(
            "Calibration results will appear after enough settled picks."
        )
    else:
        st.dataframe(
            backtest_calibration,
            use_container_width=True,
            hide_index=True,
        )

        if {
            "average_probability",
            "hit_rate",
        }.issubset(backtest_calibration.columns):
            calibration_chart = (
                backtest_calibration[
                    [
                        "average_probability",
                        "hit_rate",
                    ]
                ]
                .apply(
                    pd.to_numeric,
                    errors="coerce",
                )
                .dropna()
            )

            if not calibration_chart.empty:
                st.line_chart(
                    calibration_chart
                )

with report_tab_3:
    if bankroll_curve.empty:
        st.info(
            "The bankroll curve will appear after settled picks."
        )
    else:
        chart_columns = [
            column
            for column in [
                "cumulative_profit",
                "bankroll",
            ]
            if column in bankroll_curve.columns
        ]

        if chart_columns:
            st.line_chart(
                bankroll_curve[
                    chart_columns
                ].apply(
                    pd.to_numeric,
                    errors="coerce",
                )
            )

        st.dataframe(
            bankroll_curve.tail(100),
            use_container_width=True,
            hide_index=True,
        )

with report_tab_4:
    audit = load_csv(
        AUDIT_PATH
    )

    if audit.empty:
        st.info(
            "No current daily-card audit is available."
        )
    else:
        if "rejection_reason" in audit.columns:
            rejection_summary = (
                audit["rejection_reason"]
                .value_counts()
                .rename_axis("Reason")
                .reset_index(name="Rows")
            )

            st.subheader(
                "Current Rejection Reasons"
            )

            st.dataframe(
                rejection_summary,
                use_container_width=True,
                hide_index=True,
            )

        st.subheader(
            "Full Daily-Card Audit"
        )

        st.dataframe(
            audit,
            use_container_width=True,
            hide_index=True,
        )


st.divider()

st.caption(
    "Model outputs are experimental and are not financial advice. "
    "A high modeled probability does not guarantee a winning result. "
    "Use verified calibration, expected value, and bankroll discipline."
)
