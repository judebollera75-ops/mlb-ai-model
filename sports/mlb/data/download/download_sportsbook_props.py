def download_sportsbook_props() -> pd.DataFrame:
    """Temporarily print sample ParlayAPI rows for parser debugging."""
    if not API_KEY:
        raise RuntimeError(
            "PARLAY_API_KEY is missing from the environment."
        )

    target_date = get_target_date()

    fetched_at = datetime.now(
        timezone.utc
    ).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    existing_props = load_existing_props()

    print("=" * 72)
    print("DOWNLOADING CURRENT MLB PLATFORM PROPS")
    print("=" * 72)
    print("Provider: ParlayAPI")
    print(
        f"Slate date: "
        f"{target_date.isoformat()}"
    )
    print(
        f"Endpoint: {PROPS_URL}"
    )
    print(
        f"Requested markets: "
        f"{len(API_MARKETS)}"
    )
    print("=" * 72)

    session = build_session()

    raw_rows = fetch_props(
        session
    )

    print(
        f"Raw ParlayAPI rows: "
        f"{len(raw_rows):,}"
    )

    # Temporary debugging output so we can inspect ParlayAPI's
    # real response field names and structure.
    import json

    print("\n" + "=" * 72)
    print("PARLAYAPI RAW RESPONSE SAMPLE")
    print("=" * 72)

    print(
        json.dumps(
            raw_rows[:2],
            indent=2,
            default=str,
        )
    )

    print("=" * 72)
    print("END RAW RESPONSE SAMPLE")
    print("=" * 72)

    # Stop before overwriting platform_lines.csv with an empty file.
    raise SystemExit(
        "Temporary debug run completed successfully."
    )

    normalized = normalize_props(
        raw_rows=raw_rows,
        target_date=target_date,
        fetched_at=fetched_at,
    )

    props = clean_props(
        normalized,
        target_date,
    )

    print(
        f"Normalized side rows: "
        f"{len(normalized):,}"
    )

    print(
        f"Validated current rows: "
        f"{len(props):,}"
    )

    if not props.empty:
        save_props(
            props
        )

        return props

    if existing_props_are_current(
        existing_props,
        target_date,
    ):
        print(
            "\nWARNING: ParlayAPI returned no usable current props. "
            "Preserving the existing same-slate file because it was "
            f"fetched within {MAX_EVENT_AGE_MINUTES} minutes."
        )

        return existing_props

    print(
        "\nWARNING: No valid current platform props were available. "
        "Writing an empty file so stale recommendations cannot remain "
        "visible in the app."
    )

    empty = empty_props_frame()

    save_props(
        empty
    )

    return empty
