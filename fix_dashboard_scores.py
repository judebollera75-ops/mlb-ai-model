#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

TARGET = Path("sports/mlb/betting/build_daily_card.py")

SCORE_FUNCTION = """
def add_dashboard_quality_scores(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    def numeric(name: str, default: float = 0.0) -> pd.Series:
        if name not in result.columns:
            return pd.Series(default, index=result.index, dtype=float)
        return pd.to_numeric(
            result[name],
            errors="coerce",
        ).fillna(default)

    probability = numeric("probability").clip(0.0, 1.0)
    probability_edge = numeric("probability_edge").clip(-0.25, 0.25)
    expected_value = numeric("expected_value").clip(-0.50, 1.00)
    sample_size = numeric("calibration_sample_size").clip(lower=0.0)
    validation_mae = pd.to_numeric(
        result.get(
            "validation_mae",
            pd.Series(np.nan, index=result.index),
        ),
        errors="coerce",
    )

    probability_component = (
        (probability - 0.50) / 0.35
    ).clip(0.0, 1.0) * 100.0

    edge_component = (
        probability_edge / 0.15
    ).clip(0.0, 1.0) * 100.0

    ev_component = (
        expected_value / 0.25
    ).clip(0.0, 1.0) * 100.0

    sample_component = (
        sample_size / 1000.0
    ).clip(0.0, 1.0) * 100.0

    line_scale = numeric("line").abs().clip(lower=1.0)
    relative_mae = validation_mae / line_scale

    mae_component = (
        100.0
        - relative_mae.fillna(0.50).clip(0.0, 1.0) * 100.0
    ).clip(0.0, 100.0)

    no_vig_series = result.get(
        "no_vig_implied_probability",
        pd.Series(np.nan, index=result.index),
    )
    has_no_vig = pd.to_numeric(
        no_vig_series,
        errors="coerce",
    ).notna().astype(float)

    price_quality = (
        100.0
        - (
            numeric("sportsbook_implied_probability", 0.50)
            - probability
        ).abs().clip(0.0, 0.35)
        / 0.35
        * 100.0
    ).clip(0.0, 100.0)

    result["projection_quality_score"] = (
        0.55 * mae_component
        + 0.45 * sample_component
    ).clip(0.0, 100.0)

    result["market_quality_score"] = (
        0.55 * price_quality
        + 0.30 * sample_component
        + 0.15 * has_no_vig * 100.0
    ).clip(0.0, 100.0)

    result["confidence_score"] = (
        0.45 * probability_component
        + 0.20 * edge_component
        + 0.15 * ev_component
        + 0.10 * result["projection_quality_score"]
        + 0.10 * result["market_quality_score"]
    ).clip(0.0, 100.0)

    result["risk_score"] = (
        100.0
        - (
            0.55 * result["confidence_score"]
            + 0.25 * result["projection_quality_score"]
            + 0.20 * result["market_quality_score"]
        )
    ).clip(0.0, 100.0)

    result["ranking_score"] = (
        0.50 * result["confidence_score"]
        + 0.20 * (100.0 - result["risk_score"])
        + 0.15 * result["market_quality_score"]
        + 0.15 * result["projection_quality_score"]
    ).clip(0.0, 100.0)

    result["line_type"] = "standard"

    for column in [
        "confidence_score",
        "risk_score",
        "market_quality_score",
        "projection_quality_score",
        "ranking_score",
    ]:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        ).round(1)

    return result


"""

def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Could not patch {label}: expected exactly one marker, found {count}."
        )
    return text.replace(old, new, 1)

def main() -> None:
    if not TARGET.exists():
        raise FileNotFoundError(
            f"Could not find {TARGET}. Run this script from the repository root."
        )

    text = TARGET.read_text(encoding="utf-8")

    if (
        '"confidence_score"' in text
        and "def add_dashboard_quality_scores" in text
        and "probabilities = add_dashboard_quality_scores" in text
    ):
        print("Dashboard score patch is already installed.")
        return

    backup = TARGET.with_suffix(".py.bak")
    backup.write_text(text, encoding="utf-8")

    output_marker = '    "validation_mae",\n    "elite_score",\n'
    output_replacement = (
        '    "validation_mae",\n'
        '    "confidence_score",\n'
        '    "risk_score",\n'
        '    "market_quality_score",\n'
        '    "projection_quality_score",\n'
        '    "ranking_score",\n'
        '    "line_type",\n'
        '    "elite_score",\n'
    )
    text = replace_once(
        text, output_marker, output_replacement, "OUTPUT_COLUMNS"
    )

    function_marker = "\ndef load_probability_table() -> pd.DataFrame:\n"
    text = replace_once(
        text,
        function_marker,
        "\n" + SCORE_FUNCTION + "def load_probability_table() -> pd.DataFrame:\n",
        "score function",
    )

    call_marker = (
        '    probabilities["line_is_fresh"] = probabilities[\n'
        '        "fetched_at"\n'
        '    ].apply(line_is_fresh)\n'
    )
    call_replacement = (
        '    probabilities = add_dashboard_quality_scores(\n'
        '        probabilities\n'
        '    )\n\n'
        + call_marker
    )
    text = replace_once(
        text, call_marker, call_replacement, "score calculation call"
    )

    history_marker = '        "validation_mae",\n        "elite_score",\n'
    history_replacement = (
        '        "validation_mae",\n'
        '        "confidence_score",\n'
        '        "risk_score",\n'
        '        "market_quality_score",\n'
        '        "projection_quality_score",\n'
        '        "ranking_score",\n'
        '        "line_type",\n'
        '        "elite_score",\n'
    )
    text = replace_once(
        text, history_marker, history_replacement, "history columns"
    )

    numeric_marker = '        "validation_mae",\n        "elite_score",\n'
    numeric_replacement = (
        '        "validation_mae",\n'
        '        "confidence_score",\n'
        '        "risk_score",\n'
        '        "market_quality_score",\n'
        '        "projection_quality_score",\n'
        '        "ranking_score",\n'
        '        "elite_score",\n'
    )
    text = replace_once(
        text, numeric_marker, numeric_replacement, "numeric conversion"
    )

    TARGET.write_text(text, encoding="utf-8")
    compile(text, str(TARGET), "exec")

    print(f"Patched: {TARGET}")
    print(f"Backup:  {backup}")
    print(
        "Run the normal MLB workflow next. The regenerated "
        "outputs/mlb_daily_card.csv will contain real score values."
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
