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
            margin-bottom: 1.5rem;
        }

        .prop-card {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 14px;
            padding: 18px 20px;
            margin-bottom: 14px;
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

        .fade {
            border-left: 7px solid #dc2626;
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
    except (pd.errors.EmptyDataError, UnicodeDecodeError):
        return pd.DataFrame()


def format_market(value: str) -> str:
    return str(value).replace("_", " ").title()


def safe_number(value, decimals=2):
    number = pd.to_numeric(value, errors="coerce")

    if pd.isna(number):
        return "N/A"

    return f"{float(number):.{decimals}f}"


def normalize_tier(value: str) -> str:
    value = str(value).upper().strip()

    mapping = {
        "A+": "BEST BET",
        "A": "STRONG LEAN",
        "B": "LEAN",
    }

    return mapping.get(value, value)


def tier_class(tier: str) -> str:
    tier = normalize_tier(tier)

    return {
        "BEST BET": "best-bet",
        "STRONG LEAN": "strong-lean",
        "LEAN": "lean",
        "FADE": "fade",
    }.get(tier, "lean")


def tier_rank(tier: str) -> int:
    tier = normalize_tier(tier)

    return {
        "BEST BET": 1,
        "STRONG LEAN": 2,
        "LEAN": 3,
        "FADE": 4,
    }.get(tier, 5)


def build_summary(group: pd.DataFrame) -> pd.Series:
    wins = int((group["win_loss"] == "WIN").sum())
    losses = int((group["win_loss"] == "LOSS").sum())
    pushes = int((group["win_loss"] == "PUSH").sum())

    decisions = wins + losses
    hit_rate = wins / decisions * 100 if decisions else 0

    units = pd.to_numeric(
        group.get("profit_units"),
        errors="coerce",
    ).fillna(0).sum()

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
props = load_csv(PROP_PATH)
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
    "Ranked MLB projections, model edges, and tracked performance."
    "</div>",
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Performance metrics
# --------------------------------------------------
graded = pd.DataFrame()

if not results.empty and "win_loss" in results.columns:
    graded = results[
        results["win_loss"].isin(["WIN", "LOSS", "PUSH"])
    ].copy()

wins = int((graded["win_loss"] == "WIN").sum()) if not graded.empty else 0
losses = int((graded["win_loss"] == "LOSS").sum()) if not graded.empty else 0
pushes = int((graded["win_loss"] == "PUSH").sum()) if not graded.empty else 0

decisions = wins + losses
hit_rate = wins / decisions * 100 if decisions else 0

units = 0.0

if not graded.empty and "profit_units" in graded.columns:
    units = pd.to_numeric(
        graded["profit_units"],
        errors="coerce",
    ).fillna(0).sum()

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric("Tracked Picks", len(graded))
metric_2.metric("Record", f"{wins}-{losses}-{pushes}")
metric_3.metric("Hit Rate", f"{hit_rate:.1f}%")
metric_4.metric("Flat Units", f"{units:+.1f}")

st.caption(
    "The current 4–0 record came from the original sample test slate. "
    "Your verified live record begins with lines logged before games start."
)

st.divider()


# --------------------------------------------------
# Today's board
# --------------------------------------------------
header_left, header_right = st.columns([4, 1])

with header_left:
    st.header("🔥 Today’s Top MLB Props")

with header_right:
    if st.button("↻ Refresh Board", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

if props.empty:
    st.info(
        "No actionable props are available. Run the daily model and "
        "push the updated CSV files to GitHub."
    )

else:
    props = props.copy()

    for column in ["line", "projection", "edge"]:
        if column in props.columns:
            props[column] = pd.to_numeric(
                props[column],
                errors="coerce",
            )

    if "tier" not in props.columns:
        props["tier"] = props.get("grade", "LEAN")

    props["tier"] = props["tier"].apply(normalize_tier)
    props["tier_rank"] = props["tier"].apply(tier_rank)
    props["market_display"] = props["market"].apply(format_market)
    props["absolute_edge"] = props["edge"].abs()

    platforms = sorted(
        props["platform"].dropna().astype(str).unique()
    )

    markets = sorted(
        props["market_display"].dropna().astype(str).unique()
    )

    tiers = [
        tier
        for tier in [
            "BEST BET",
            "STRONG LEAN",
            "LEAN",
            "FADE",
        ]
        if tier in props["tier"].unique()
    ]

    filter_1, filter_2, filter_3, filter_4 = st.columns(4)

    with filter_1:
        selected_platforms = st.multiselect(
            "Sportsbook",
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
        selected_tiers = st.multiselect(
            "Rating",
            tiers,
            default=tiers,
        )

    with filter_4:
        sort_choice = st.selectbox(
            "Sort by",
            [
                "Best Rating",
                "Largest Edge",
                "Player Name",
            ],
        )

    filtered = props[
        props["platform"].isin(selected_platforms)
        & props["market_display"].isin(selected_markets)
        & props["tier"].isin(selected_tiers)
    ].copy()

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

    else:
        filtered = filtered.sort_values(
            ["tier_rank", "absolute_edge"],
            ascending=[True, False],
        )

    available_props = len(filtered)

    if available_props <= 1:
        max_props = available_props
        st.caption(f"Showing {available_props} available prop.")
    else:
        max_props = st.slider(
            "Number of props to display",
            min_value=1,
            max_value=min(25, available_props),
            value=min(10, available_props),
        )

    filtered = filtered.head(max_props)

    if filtered.empty:
        st.warning("No props match the selected filters.")

    for rank, (_, row) in enumerate(
        filtered.iterrows(),
        start=1,
    ):
        tier = row.get("tier", "LEAN")
        css_class = tier_class(tier)

        player = row.get("player", "Unknown Player")
        platform = row.get("platform", "Unknown")
        market = row.get("market_display", "")
        pick = str(row.get("pick", "")).upper()
        line = safe_number(row.get("line"))
        projection = safe_number(row.get("projection"))
        edge = safe_number(row.get("edge"))

        team = row.get("team", "")
        opponent = row.get("opponent", "")

        pick_class = (
            "pick-over"
            if "OVER" in pick
            else "pick-under"
        )

        team_text = ""

        if pd.notna(team) and str(team) not in ["", "nan"]:
            team_text = str(team)

        if pd.notna(opponent) and str(opponent) not in ["", "nan"]:
            team_text += f" vs. {opponent}"

        st.markdown(
            f"""<div class="prop-card {css_class}">
<div class="tier-label">#{rank} · {tier}</div>
<div class="player-name">{player}</div>
<div class="prop-detail">{platform} · {market}</div>
<div class="prop-detail">{team_text}</div>
</div>""",
            unsafe_allow_html=True,
        )

        col_1, col_2, col_3, col_4 = st.columns(4)

        with col_1:
            st.markdown(
                f'<div class="{pick_class}">'
                f"{pick} {line}"
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
            if "win_probability" in row.index and pd.notna(
                row.get("win_probability")
            ):
                st.metric(
                    "Model Win Probability",
                    f"{float(row['win_probability']):.1f}%",
                )
            else:
                st.metric(
                    "Model Win Probability",
                    "Pending calibration",
                )

        with st.expander("View model details"):
            detail_1, detail_2, detail_3 = st.columns(3)

            detail_1.write(f"**Sportsbook:** {platform}")
            detail_1.write(f"**Market:** {market}")

            detail_2.write(f"**Line:** {line}")
            detail_2.write(f"**Projection:** {projection}")

            detail_3.write(f"**Pick:** {pick}")
            detail_3.write(f"**Raw edge:** {edge}")

            if "fair_odds" in row.index and pd.notna(
                row.get("fair_odds")
            ):
                st.write(
                    f"**Model fair odds:** {row['fair_odds']}"
                )

            if "sportsbook_odds" in row.index and pd.notna(
                row.get("sportsbook_odds")
            ):
                st.write(
                    f"**Sportsbook odds:** {row['sportsbook_odds']}"
                )


# --------------------------------------------------
# Performance section
# --------------------------------------------------
st.divider()
st.header("📈 Model Performance")

if graded.empty:
    st.info("No graded live selections are available yet.")

else:
    tab_market, tab_platform, tab_history = st.tabs(
        [
            "By Market",
            "By Sportsbook",
            "Pick History",
        ]
    )

    with tab_market:
        market_summary = (
            graded.groupby("market", dropna=False)
            .apply(build_summary, include_groups=False)
            .reset_index()
        )

        st.dataframe(
            market_summary,
            use_container_width=True,
            hide_index=True,
        )

    with tab_platform:
        platform_summary = (
            graded.groupby("platform", dropna=False)
            .apply(build_summary, include_groups=False)
            .reset_index()
        )

        st.dataframe(
            platform_summary,
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
    "Model projections are experimental. Win probabilities, fair odds, "
    "and expected value will appear only after proper historical calibration."
)
