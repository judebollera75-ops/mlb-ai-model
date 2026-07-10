import os
import joblib
import pandas as pd


TARGET_DATE = "2026-07-10"

HITTERS_PATH = f"data/hitters/{TARGET_DATE}.csv"
LOGS_PATH = "data/hitter_game_logs/2026.csv"

HITS_MODEL_PATH = "models/hitters/hits_model.pkl"
TOTAL_BASES_MODEL_PATH = "models/hitters/total_bases_model.pkl"

OUTPUT_PATH = "outputs/hitters/today_hitter_projections.csv"

ROLLING_STATS = [
    "plate_appearances",
    "at_bats",
    "hits",
    "total_bases",
    "home_runs",
    "runs",
    "rbi",
    "walks",
    "strikeouts",
    "stolen_bases",
]


def load_model_bundle(path):
    bundle = joblib.load(path)

    required_keys = ["model", "features", "medians"]

    for key in required_keys:
        if key not in bundle:
            raise ValueError(f"{path} is missing required key: {key}")

    return bundle


def build_today_features():
    hitters = pd.read_csv(HITTERS_PATH)
    logs = pd.read_csv(LOGS_PATH)

    hitters["player_id"] = pd.to_numeric(
        hitters["player_id"],
        errors="coerce"
    )

    logs["player_id"] = pd.to_numeric(
        logs["player_id"],
        errors="coerce"
    )

    logs["date"] = pd.to_datetime(
        logs["date"],
        errors="coerce"
    )

    target_date = pd.to_datetime(TARGET_DATE)

    logs = logs[
        logs["date"] < target_date
    ].copy()

    logs = logs.sort_values(
        ["player_id", "date", "game_id"]
    )

    for stat in ROLLING_STATS:
        if stat in logs.columns:
            logs[stat] = pd.to_numeric(
                logs[stat],
                errors="coerce"
            ).fillna(0)

    feature_rows = []

    for player_id, player_logs in logs.groupby("player_id"):
        player_logs = player_logs.sort_values(
            ["date", "game_id"]
        )

        row = {
            "player_id": player_id,
            "prior_games": len(player_logs),
        }

        if len(player_logs) > 0:
            last_game_date = player_logs["date"].max()

            row["days_rest"] = max(
                0,
                min(
                    14,
                    (target_date - last_game_date).days
                )
            )
        else:
            row["days_rest"] = None

        for window in [3, 5, 10]:
            recent = player_logs.tail(window)

            for stat in ROLLING_STATS:
                if stat in recent.columns:
                    row[f"last{window}_avg_{stat}"] = (
                        recent[stat].mean()
                        if len(recent) > 0
                        else None
                    )

        if len(player_logs) > 0:
            previous_game = player_logs.iloc[-1]

            for stat in [
                "hits",
                "total_bases",
                "home_runs",
                "runs",
                "rbi",
                "walks",
                "strikeouts",
            ]:
                row[f"previous_game_{stat}"] = (
                    previous_game.get(stat)
                )

        feature_rows.append(row)

    features = pd.DataFrame(feature_rows)

    today = hitters.merge(
        features,
        on="player_id",
        how="left"
    )

    # A box score can contain duplicate player rows.
    today = today.drop_duplicates(
        subset=["game_id", "player_id"]
    )

    return today


def make_predictions(df, bundle, output_column):
    feature_cols = bundle["features"]
    medians = bundle["medians"]
    model = bundle["model"]

    for column in feature_cols:
        if column not in df.columns:
            df[column] = None

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    X = df[feature_cols].copy()

    for column in feature_cols:
        X[column] = X[column].fillna(
            medians.get(column, 0)
        )

    predictions = model.predict(X)
    df[output_column] = predictions.clip(min=0)

    return df


def project_today_hitters():
    os.makedirs("outputs/hitters", exist_ok=True)

    today = build_today_features()

    hits_bundle = load_model_bundle(
        HITS_MODEL_PATH
    )

    total_bases_bundle = load_model_bundle(
        TOTAL_BASES_MODEL_PATH
    )

    today = make_predictions(
        today,
        hits_bundle,
        "projected_hits"
    )

    today = make_predictions(
        today,
        total_bases_bundle,
        "projected_total_bases"
    )

    today["projected_hits"] = today[
        "projected_hits"
    ].round(3)

    today["projected_total_bases"] = today[
        "projected_total_bases"
    ].round(3)

    output_columns = [
        "date",
        "game_id",
        "player_id",
        "player_name",
        "team",
        "opponent",
        "side",
        "batting_order",
        "position",
        "prior_games",
        "days_rest",
        "projected_hits",
        "projected_total_bases",
    ]

    output_columns = [
        column
        for column in output_columns
        if column in today.columns
    ]

    output = today[output_columns].copy()

    output = output.sort_values(
        [
            "projected_total_bases",
            "projected_hits",
        ],
        ascending=False
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"Saved {len(output)} projections to {OUTPUT_PATH}")
    print()
    print(
        output.head(40).to_string(index=False)
    )

    return output


if __name__ == "__main__":
    project_today_hitters()
