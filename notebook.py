import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import altair as alt
    import polars as pl

    return Path, alt, pl


@app.cell
def _(Path):
    MISSION1 = Path("../data/ESA_Mission1.parquet")
    META = Path("../data/ESA-Mission1")
    return META, MISSION1


@app.cell
def _(MISSION1, pl):
    lf = pl.scan_parquet(MISSION1)
    return (lf,)


@app.cell
def _(META, pl):
    channels_meta = pl.read_csv(META / "channels.csv")
    channels_meta
    return (channels_meta,)


@app.cell
def _(channels_meta, lf, pl):
    def _key(name):
        stem, num = name.rsplit("_", 1)
        return stem, int(num)

    # collect_schema() reads the footer only, no data.
    _names = lf.collect_schema().names()
    CHANNELS = sorted((c for c in _names if c.startswith("channel_")), key=_key)
    TELECOMMANDS = sorted((c for c in _names if c.startswith("telecommand_")), key=_key)

    # Only the 58 Target=YES channels were ever annotated. The other 18 channels
    # and all 11 telecommands have all-zero label columns, which means
    # *unlabelled*, not nominal - scoring against them would hand you ~23% of the
    # evaluation surface as free true-negatives. Widen to CHANNELS only if you
    # have a reason to.
    LABELED_CHANNELS = sorted(
        channels_meta.filter(pl.col("Target") == "YES")["Channel"].to_list(),
        key=_key,
    )

    len(CHANNELS), len(TELECOMMANDS), len(LABELED_CHANNELS)
    return CHANNELS, LABELED_CHANNELS, TELECOMMANDS


@app.cell
def _(pl):
    # first 3 months of data, to make it easier to work with
    WINDOW = pl.col("timestamp").is_between(
        pl.lit("2006-01-01").str.to_datetime(),
        pl.lit("2012-01-01").str.to_datetime(),
    )
    return (WINDOW,)


@app.cell
def _(CHANNELS, TELECOMMANDS, WINDOW, lf):
    features = lf.filter(WINDOW).select("timestamp", *CHANNELS, *TELECOMMANDS).collect()
    features
    return (features,)


@app.cell
def _(LABELED_CHANNELS, WINDOW, lf, pl):
    #
    # is_anomaly collapses the per-channel labels by precedence, not by sum -
    # summing 0/1/2/3 across columns lets three rare-event channels outrank one
    # real anomaly. -1 marks rows that should train nothing and score nothing:
    # rare events are expected behaviour, and gaps are stretches where telemetry
    # was never received and the resampler forward-filled a flat line.
    # the labels follow:
    #    NOMINAL = 0
    #    ANOMALY = 1
    #    RARE_EVENT = 2
    #    GAP = 3
    #    INVALID = 4
    #

    ANOMALY_LABELS = [1, 2]
    # map 1,2 -> anomaly
    # also did serch for 'invalid' rows, none found tho
    # gaps are considered normal
    _cols = [f"is_anomaly_{c}" for c in LABELED_CHANNELS]
    labels = (
        lf.filter(WINDOW)
        .select("timestamp", *_cols)
        .with_columns(
            is_anomaly=pl.when(pl.any_horizontal(pl.col(_cols).is_in(ANOMALY_LABELS)))
            .then(1)
            .when(pl.any_horizontal(pl.col(_cols) == 4))
            .then(-1)
            .otherwise(0)
            .cast(pl.Int8),
        )
        .select("timestamp", "is_anomaly", pl.exclude("timestamp", "is_anomaly"))
        .collect()
    )
    labels
    return ANOMALY_LABELS, labels


@app.cell
def _(labels, pl) -> None:
    labels.select(
        rows=pl.len(),
        anomaly=(pl.col("is_anomaly") == 1).sum(),
        ignore=(pl.col("is_anomaly") == -1).sum(),
        nominal=(pl.col("is_anomaly") == 0).sum(),
    )


@app.cell
def _(ANOMALY_LABELS, LABELED_CHANNELS, labels, pl):
    # "Most anomalies" ranked by timestep count, which favours long events. Rank
    # by spans.height instead to surface channels with many *short* events - a
    # different, and usually harder, detection problem.
    _counts = labels.select(
        pl.col(f"is_anomaly_{c}").is_in(ANOMALY_LABELS).sum().alias(c)
        for c in LABELED_CHANNELS
    ).row(0, named=True)
    # TOP_CHANNEL = max(_counts, key=_counts.get)
    # TOP_CHANNEL, _counts[TOP_CHANNEL]
    TOP_CHANNEL = "channel_44"
    return (TOP_CHANNEL,)


@app.cell
def _(ANOMALY_LABELS, TOP_CHANNEL, labels, pl):
    spans = (
        labels.select(
            "timestamp",
            flag=pl.col(f"is_anomaly_{TOP_CHANNEL}").is_in(ANOMALY_LABELS),
        )
        .with_columns(
            run=(pl.col("flag") != pl.col("flag").shift(1, fill_value=False)).cum_sum(),
        )
        .filter("flag")
        .group_by("run")
        .agg(
            start=pl.col("timestamp").min(),
            end=pl.col("timestamp").max(),
        )
        .sort("start")
        .drop("run")
    )
    spans
    return (spans,)


@app.cell
def _(TOP_CHANNEL, features, pl):
    series = (
        features.select("timestamp", TOP_CHANNEL)
        .sort("timestamp")
        .group_by_dynamic("timestamp", every="4h")
        .agg(
            lo=pl.col(TOP_CHANNEL).min(),
            hi=pl.col(TOP_CHANNEL).max(),
            mid=pl.col(TOP_CHANNEL).mean(),
        )
    )
    return (series,)


@app.cell
def _(TOP_CHANNEL, alt, series, spans) -> None:
    # mark_rect with x/x2 and no y encoding spans the full plot height.
    # Spans that touch a window edge are *truncated* events, not events that
    # ended there - handle that explicitly once you move to event-level metrics.
    _y = alt.Scale(
        zero=False,
    )  # channel sits in [0.66, 0.80]; zero wastes 85% of the plot
    _bg = (
        alt.Chart(spans)
        .mark_rect(color="red", opacity=0.15)
        .encode(
            x=alt.X("start:T", title=None),
            x2="end:T",
            tooltip=["start:T", "end:T"],
        )
    )
    _band = (
        alt.Chart(series)
        .mark_area(opacity=0.25)
        .encode(
            x=alt.X("timestamp:T", title=None),
            y=alt.Y("lo:Q", title=TOP_CHANNEL, scale=_y),
            y2="hi:Q",
        )
    )
    _line = (
        alt.Chart(series)
        .mark_line(strokeWidth=1)
        .encode(x="timestamp:T", y=alt.Y("mid:Q", scale=_y), tooltip=["mid:Q"])
    )
    (_bg + _band + _line).properties(
        width=900,
        height=260,
        title=f"{TOP_CHANNEL} — {spans.height} anomalous spans",
    )


if __name__ == "__main__":
    app.run()
