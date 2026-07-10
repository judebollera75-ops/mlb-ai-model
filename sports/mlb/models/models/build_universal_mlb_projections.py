import os
import pandas as pd


OUTPUT_PATH = "outputs/mlb_universal_projections.csv"


def add_market_rows(
    rows,
    dataframe,
    player_column,
    projection_column,
    market_name,
    extra_columns=None,
):
    extra_columns = extra_columns or []

    for _, row in dataframe.iterrows():
        projection = pd.to_numeric(
            row.get(projection_column),
            errors="coerce"
        )

        if pd.isna(projection):
            continue

        output_row = {
            "sport": "MLB",
            "player": row.get(player_column),
            "market": market_name,
            "projection": round(float(projection), 3),
            "team": row.get("team"),
            "opponent": row.get("opponent"),
        }

        for column in extra_columns:
            output_row[column] = row.get(column)

        rows.append(output_row)


def build_universal_mlb_projections():
    os.makedirs("outputs", exist_ok=True)

    rows = []

    # Hitter projections
    hitter_path = "outputs/hitters/today_hitter_projections.csv"

    if os.path.exists(hitter_path):
        hitters = pd.read_csv(hitter_path)

        add_market_rows(
            rows,
            hitters,
            player_column="player_name",
            projection_column="projected_hits",
            market_name="hitter_hits",
            extra_columns=["game_id", "batting_order", "position"],
        )

        add_market_rows(
            rows,
            hitters,
            player_column="player_name",
            projection_column="projected_total_bases",
            market_name="hitter_total_bases",
            extra_columns=["game_id", "batting_order", "position"],
        )
    else:
        print(f"Missing hitter file: {hitter_path}")

    # Pitcher strikeout projections
    strikeout_path = "outputs/calibrated_strikeout_projections.csv"

    if os.path.exists(strikeout_path):
        strikeouts = pd.read_csv(strikeout_path)

        add_market_rows(
            rows,
            strikeouts,
            player_column="pitcher_name",
            projection_column="calibrated_projected_ks",
            market_name="pitcher_strikeouts",
            extra_columns=["game_id"],
        )
    else:
        print(f"Missing strikeout file: {strikeout_path}")

    # Pitcher outs projections
    outs_path = "outputs/pitcher_outs_projections.csv"

    if os.path.exists(outs_path):
        outs = pd.read_csv(outs_path)

        add_market_rows(
            rows,
            outs,
            player_column="pitcher_name",
            projection_column="projected_outs",
            market_name="pitcher_outs",
            extra_columns=["game_id"],
        )
    else:
        print(f"Missing outs file: {outs_path}")

    # Pitcher fantasy projections
    fantasy_path = "outputs/pitcher_fantasy_projections.csv"

    if os.path.exists(fantasy_path):
        fantasy = pd.read_csv(fantasy_path)

        fantasy_projection_column = None

        for candidate in [
            "draftkings_pitcher_points",
            "projected_fantasy_score",
            "fantasy_projection",
        ]:
            if candidate in fantasy.columns:
                fantasy_projection_column = candidate
                break

        if fantasy_projection_column:
            add_market_rows(
                rows,
                fantasy,
                player_column="pitcher_name",
                projection_column=fantasy_projection_column,
                market_name="pitcher_fantasy_score",
                extra_columns=["game_id"],
            )
        else:
            print("Pitcher fantasy projection column not found.")
    else:
        print(f"Missing fantasy file: {fantasy_path}")

    universal = pd.DataFrame(rows)

    if universal.empty:
        raise ValueError("No projection rows were created.")

    universal = universal.dropna(
        subset=["player", "market", "projection"]
    )

    universal = universal.drop_duplicates(
        subset=["player", "market"],
        keep="first"
    )

    universal = universal.sort_values(
        ["market", "projection"],
        ascending=[True, False]
    )

    universal.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(universal)} projections to {OUTPUT_PATH}")
    print()
    print(universal.groupby("market").size())
    print()
    print(universal.head(50).to_string(index=False))

    return universal


if __name__ == "__main__":
    build_universal_mlb_projections()
