import os
import pandas as pd


INPUT_PATH = "outputs/mlb_daily_card.csv"
OUTPUT_PATH = "outputs/top_mlb_props.csv"

MAX_PROPS = 15


def label_prop(row):
    grade = str(row.get("grade", "")).upper()
    edge = abs(pd.to_numeric(row.get("edge"), errors="coerce"))

    if pd.isna(edge):
        return "PASS"

    if grade == "A+":
        return "BEST BET"

    if grade == "A":
        return "STRONG LEAN"

    if grade == "B":
        return "LEAN"

    return "PASS"


def clean_pick(value):
    value = str(value).upper().strip()

    if value == "MORE/YES":
        return "OVER"

    if value == "LESS/NO":
        return "UNDER"

    return value


def build_clean_prop_board():
    os.makedirs("outputs", exist_ok=True)

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Missing daily card: {INPUT_PATH}"
        )

    card = pd.read_csv(INPUT_PATH)

    card["projection"] = pd.to_numeric(
        card["projection"],
        errors="coerce"
    )

    card["line"] = pd.to_numeric(
        card["line"],
        errors="coerce"
    )

    card["edge"] = pd.to_numeric(
        card["edge"],
        errors="coerce"
    )

    # Remove stale lines and rows without a model projection
    clean = card[
        card["projection"].notna()
        & card["line"].notna()
        & card["edge"].notna()
        & ~card["pick"].astype(str).str.contains(
            "NO PROJECTION",
            case=False,
            na=False,
        )
    ].copy()

    if clean.empty:
        print("No matched props are available yet.")
        print(
            "The platform lines do not currently match "
            "today's model projections."
        )

        pd.DataFrame().to_csv(
            OUTPUT_PATH,
            index=False
        )
        return pd.DataFrame()

    clean["pick"] = clean["pick"].apply(clean_pick)
    clean["tier"] = clean.apply(label_prop, axis=1)
    clean["absolute_edge"] = clean["edge"].abs()

    # Hide very weak or unmatched plays
    clean = clean[
        clean["tier"] != "PASS"
    ].copy()

    tier_order = {
        "BEST BET": 1,
        "STRONG LEAN": 2,
        "LEAN": 3,
    }

    clean["tier_order"] = clean["tier"].map(tier_order)

    clean = clean.sort_values(
        ["tier_order", "absolute_edge"],
        ascending=[True, False]
    )

    clean = clean.head(MAX_PROPS)

    columns = [
        "tier",
        "platform",
        "player",
        "market",
        "line",
        "projection",
        "edge",
        "pick",
        "team",
        "opponent",
    ]

    columns = [
        column
        for column in columns
        if column in clean.columns
    ]

    output = clean[columns].copy()

    output["projection"] = output[
        "projection"
    ].round(2)

    output["edge"] = output[
        "edge"
    ].round(2)

    output.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nTODAY'S TOP MLB PROPS")
    print("=" * 70)

    for _, row in output.iterrows():
        print(
            f"\n{row['tier']} — "
            f"{row['player']}"
        )

        print(
            f"{row['platform']} | "
            f"{row['market']}"
        )

        print(
            f"Line: {row['line']} | "
            f"Model: {row['projection']} | "
            f"Pick: {row['pick']}"
        )

        print(
            f"Projection edge: "
            f"{row['edge']:+.2f}"
        )

    print(
        f"\nSaved {len(output)} ranked props "
        f"to {OUTPUT_PATH}"
    )

    return output


if __name__ == "__main__":
    build_clean_prop_board()
