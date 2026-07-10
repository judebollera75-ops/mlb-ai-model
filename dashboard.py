import os
import pandas as pd


LOG_PATH = "data/model_results_log.csv"
OUTPUT_PATH = "outputs/performance_summary.csv"


def percentage(numerator, denominator):
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 1)


def summarize(group):
    wins = (group["win_loss"] == "WIN").sum()
    losses = (group["win_loss"] == "LOSS").sum()
    pushes = (group["win_loss"] == "PUSH").sum()

    decisions = wins + losses

    units = pd.to_numeric(
        group["profit_units"],
        errors="coerce"
    ).fillna(0).sum()

    return pd.Series({
        "picks": len(group),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate": percentage(wins, decisions),
        "profit_units": round(units, 2),
        "roi_percent": percentage(units, decisions),
    })


def print_section(title, table):
    print(f"\n{title}")
    print("-" * len(title))

    if table.empty:
        print("No graded picks yet.")
    else:
        print(table.to_string())


def build_dashboard():
    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(
            f"Missing results log: {LOG_PATH}"
        )

    log = pd.read_csv(LOG_PATH)

    required_columns = [
        "date",
        "platform",
        "player",
        "market",
        "grade",
        "win_loss",
        "profit_units",
    ]

    missing = [
        column
        for column in required_columns
        if column not in log.columns
    ]

    if missing:
        raise ValueError(
            f"Results log is missing columns: {missing}"
        )

    log["date"] = pd.to_datetime(
        log["date"],
        errors="coerce"
    )

    graded = log[
        log["win_loss"].isin(["WIN", "LOSS", "PUSH"])
    ].copy()

    if graded.empty:
        print("No graded picks are available yet.")
        return

    lifetime = summarize(graded)

    print("\nMLB MODEL PERFORMANCE DASHBOARD")
    print("=" * 31)

    print("\nLIFETIME")
    print("--------")
    print("Total graded picks:", int(lifetime["picks"]))
    print("Wins:", int(lifetime["wins"]))
    print("Losses:", int(lifetime["losses"]))
    print("Pushes:", int(lifetime["pushes"]))
    print("Hit rate:", f'{lifetime["hit_rate"]}%')
    print("Profit units:", lifetime["profit_units"])
    print("Flat-unit ROI:", f'{lifetime["roi_percent"]}%')

    by_grade = (
        graded.groupby("grade", dropna=False)
        .apply(summarize, include_groups=False)
        .sort_values("profit_units", ascending=False)
    )

    by_market = (
        graded.groupby("market", dropna=False)
        .apply(summarize, include_groups=False)
        .sort_values("profit_units", ascending=False)
    )

    by_platform = (
        graded.groupby("platform", dropna=False)
        .apply(summarize, include_groups=False)
        .sort_values("profit_units", ascending=False)
    )

    print_section("BY GRADE", by_grade)
    print_section("BY MARKET", by_market)
    print_section("BY PLATFORM", by_platform)

    latest_date = graded["date"].max()

    last_7_days = graded[
        graded["date"] >= latest_date - pd.Timedelta(days=6)
    ]

    last_30_days = graded[
        graded["date"] >= latest_date - pd.Timedelta(days=29)
    ]

    recent = pd.DataFrame({
        "last_7_days": summarize(last_7_days),
        "last_30_days": summarize(last_30_days),
        "lifetime": lifetime,
    }).T

    print_section("RECENT PERFORMANCE", recent)

    summary_rows = []

    for category, table in [
        ("grade", by_grade),
        ("market", by_market),
        ("platform", by_platform),
    ]:
        exported = table.reset_index()
        exported.insert(0, "category", category)
        exported = exported.rename(
            columns={exported.columns[1]: "group"}
        )
        summary_rows.append(exported)

    summary = pd.concat(
        summary_rows,
        ignore_index=True
    )

    os.makedirs("outputs", exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved summary to {OUTPUT_PATH}")


if __name__ == "__main__":
    build_dashboard()
