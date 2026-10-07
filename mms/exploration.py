"""Notebook summaries, explicit clock alignment and scratch-only exports."""

from __future__ import annotations

import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.signal import lombscargle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import fixation, hrv, io, paths, stats

SESSIONS = (None, 1, 2, 3)
EXPECTED_ITEMS = {"HADS": 14, "STAI-S": 20, "STAI-T": 20, "BFI": 10, "FQ": 24}
GROUP_LABELS = {
    "HRV_SDNN": "Historical interval-sample SD (ms)",
    "Pupil_Dilation_STD": "Historical pupil SD (source units)",
    "Psychometric_Test_Duration_STD": "Response-duration SD (s)",
}


def data_root():
    return os.environ.get("MMS_DATA_ROOT") or None


def session_label(session):
    return "Baseline" if session is None else f"Session {session:02d}"


def recordings(kind, *, source="case-study"):
    """Read all four recordings without merging their independent clocks."""
    loaders = {
        "hr": io.load_hr,
        "ibi": io.load_ibi,
        "eye": lambda s, **kw: io.load_raw("sed", s, **kw),
    }
    if kind not in loaders:
        raise ValueError("kind must be hr, ibi or eye")
    return {
        session_label(s): loaders[kind](s, source=source, data_root=data_root())
        for s in SESSIONS
    }


def _real(series, name):
    if pd.api.types.is_datetime64_any_dtype(
        series
    ) or pd.api.types.is_timedelta64_dtype(series):
        raise ValueError(f"{name} must contain real measurements")
    if any(
        isinstance(v, (bool, np.bool_, complex, np.complexfloating)) for v in series
    ):
        raise ValueError(f"{name} must contain real measurements")
    try:
        return pd.to_numeric(series, errors="raise")
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must contain real measurements") from error


def _channels(frame):
    if "iSensor" not in frame:
        return [("unrecorded", frame)]
    if frame["iSensor"].isna().any():
        raise ValueError("channel identities must not be missing")
    return frame.groupby("iSensor", sort=False)


def _valid_hr(frame):
    values = _real(frame["heart_rate"], "heart_rate")
    if "confidence" not in frame:
        raise ValueError("HR confidence is required")
    confidence = _real(frame["confidence"], "confidence")
    return values.where(
        np.isfinite(values) & (values > 0) & np.isfinite(confidence) & confidence.eq(1)
    )


def hr_summary(streams):
    rows = []
    for label, frame in streams.items():
        for channel, part in _channels(frame):
            sample = _valid_hr(part).dropna()
            rows.append(
                {
                    "Recording": label,
                    "Channel": channel,
                    "n_samples": len(sample),
                    "Mean HR (bpm)": sample.mean(),
                    "HR SD (bpm)": sample.std(ddof=1),
                }
            )
    return pd.DataFrame(rows)


def interval_summary(streams):
    rows = []
    for label, frame in streams.items():
        # Shared rolling validation checks original order and real-valued times.
        hrv.hrv_rolling(frame, window_beats=30)
        for channel, part in _channels(frame):
            sample = _real(part["ibi"], "ibi").where(
                lambda x: np.isfinite(x) & x.between(300, 2000)
            )
            rows.append(
                {
                    "Recording": label,
                    "Channel": channel,
                    "n_samples": int(sample.notna().sum()),
                    "n_pairs": int(sample.diff().notna().sum()),
                    "Sample SD (ms)": hrv.sdnn(part["ibi"]),
                    "Adjacent-sample RMS difference (ms)": hrv.rmssd(part["ibi"]),
                }
            )
    return pd.DataFrame(rows)


def low_movement_events(frame):
    """Legacy source-vector step <0.01; quality >=0.5; gaps >0.1s split runs."""
    clean = frame.drop(columns=["fixation"], errors="ignore")
    return fixation.fixation_events(
        clean, quality_min=0.5, displacement_threshold=0.01, max_gap_s=0.1
    )


def closure_candidates(frame):
    """Exploratory openness <=1, quality >=0.5, duration 0.05–0.5s, gap <=0.1s."""
    values = [
        _real(frame[name], name)
        for name in ("leftEyeOpen", "rightEyeOpen", "leftEyeOpenQ", "rightEyeOpenQ")
    ]
    left, right, lq, rq = values
    valid = np.isfinite(left) & np.isfinite(right) & np.isfinite(lq) & np.isfinite(rq)
    valid &= (
        left.between(0, 10)
        & right.between(0, 10)
        & lq.between(0.5, 1)
        & rq.between(0.5, 1)
    )
    closed = (left <= 1) & (right <= 1)
    events = fixation.blink_events(
        frame["reltime"],
        closed,
        valid=valid,
        min_duration_s=0.05,
        max_duration_s=0.5,
        max_gap_s=0.1,
    )
    time = _real(frame["reltime"], "reltime")
    delta = time.diff()
    observed = float(
        delta.where(valid & valid.shift(fill_value=False) & delta.le(0.1)).sum()
    )
    return events, observed


def eye_summary(streams):
    rows = []
    for label, frame in streams.items():
        pupil = fixation.pupil_samples(frame, quality_min=0.5)
        events = low_movement_events(frame)
        complete = events.loc[~events["left_censored"] & ~events["right_censored"]]
        closures, observed = closure_candidates(frame)
        rows.append(
            {
                "Recording": label,
                "n_valid_pupil": int(pupil.notna().sum()),
                "Mean pupil (source units)": pupil.mean(),
                "Pupil SD (source units)": pupil.std(ddof=1),
                "Complete low-movement runs": len(complete),
                "Mean run duration (ms)": complete["duration_s"].mean() * 1000,
                "Closure candidates": len(closures),
                "Valid eye observation (s)": observed,
                "Closure candidates/min": len(closures) * 60 / observed
                if observed > 0
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def raw_answer_summary(frame):
    """Complete raw answer codes, without assuming scale scoring or reversal keys."""
    io.validate_psychometric(frame)
    if frame["Type"].isna().any() or set(frame["Type"]) != set(EXPECTED_ITEMS):
        raise ValueError("expected the five declared instruments")
    ids = frame["Test"].astype(str).str.extract(r"^\s*(\d+)\.")[0]
    if ids.isna().any():
        raise ValueError("item identities must begin with a numbered item")
    check = frame.assign(item_id=ids.astype(int))
    answers = _real(frame["Answer"], "Answer")
    if not np.isfinite(answers).all():
        raise ValueError("raw answer codes must be finite")
    for kind, count in EXPECTED_ITEMS.items():
        part = check.loc[check["Type"] == kind]
        if (
            len(part) != count
            or part["item_id"].duplicated().any()
            or set(part["item_id"]) != set(range(1, count + 1))
        ):
            raise ValueError(
                f"{kind} responses must contain each of {count} items exactly once"
            )
    return (
        frame.assign(raw_code=answers)
        .groupby("Type")["raw_code"]
        .agg(
            n_items="count",
            mean_raw_code="mean",
            min_raw_code="min",
            max_raw_code="max",
        )
        .reset_index()
    )


def duration_summary(frame):
    io.validate_psychometric(frame)
    return (
        frame.assign(response_s=_real(frame["Time(s)"], "Time(s)"))
        .groupby("Type")["response_s"]
        .agg(
            n_items="count",
            total_response_s="sum",
            mean_response_s="mean",
            response_sd_s="std",
        )
        .reset_index()
    )


def assign_questions(sample_times, questions, *, sensor_timezone=None):
    """Assign samples to [start, answer) intervals; unknown sensor zones fail closed."""
    io.validate_psychometric(questions)
    time = io.parse_datetime(sample_times, naive_timezone=sensor_timezone)
    start = io.parse_datetime(questions["Question Start Time"])
    end = io.parse_datetime(questions["Question Answer Time"])
    if time.dt.tz is None or start.dt.tz is None or end.dt.tz is None:
        raise ValueError(
            "question alignment requires explicit timezone metadata for every clock"
        )
    start, end, time = (
        start.dt.tz_convert("UTC"),
        end.dt.tz_convert("UTC"),
        time.dt.tz_convert("UTC"),
    )
    order = np.argsort(start.to_numpy(), kind="stable")
    starts, ends = start.iloc[order], end.iloc[order]
    if len(starts) > 1 and np.any(
        starts.iloc[1:].to_numpy() < ends.iloc[:-1].to_numpy()
    ):
        raise ValueError("question intervals overlap")
    result = pd.Series(
        pd.NA, index=sample_times.index, dtype="Int64", name="question_pos"
    )
    # Values are row positions, so duplicate external indexes cannot confuse joins.
    for position in order:
        mask = (time >= start.iloc[position]) & (time < end.iloc[position])
        result.iloc[np.flatnonzero(mask.to_numpy())] = int(position)
    return result


def question_features(session, *, source="case-study", sensor_timezone=None):
    """Per-question/channel descriptive features and coverage; no scale score."""
    questions = io.load_psychometric(session, source=source, data_root=data_root())
    raw_answer_summary(questions)
    hr = io.load_hr(session, source=source, data_root=data_root())
    ibi = io.load_ibi(session, source=source, data_root=data_root())
    eye = io.load_raw("sed", session, source=source, data_root=data_root())
    selected = {
        name: assign_questions(
            frame["datetime"], questions, sensor_timezone=sensor_timezone
        )
        for name, frame in (("hr", hr), ("ibi", ibi), ("eye", eye))
    }
    rows = []
    for pos in range(len(questions)):
        item = questions.iloc[pos]
        eye_part = eye.loc[selected["eye"].eq(pos).fillna(False)]
        pupils = fixation.pupil_samples(eye_part, quality_min=0.5)
        events = low_movement_events(eye_part)
        complete = events.loc[~events["left_censored"] & ~events["right_censored"]]
        closures, observed = closure_candidates(eye_part)
        channels = set(ibi["iSensor"]) | set(hr["iSensor"])
        for channel in sorted(channels):
            h = hr.loc[selected["hr"].eq(pos).fillna(False) & hr["iSensor"].eq(channel)]
            b = ibi.loc[
                selected["ibi"].eq(pos).fillna(False) & ibi["iSensor"].eq(channel)
            ]
            hv = _valid_hr(h)
            bv = _real(b["ibi"], "ibi").where(
                lambda x: np.isfinite(x) & x.between(300, 2000)
            )
            rows.append(
                {
                    "Session": session,
                    "Type": item["Type"],
                    "Item": int(str(item["Test"]).split(".")[0]),
                    "Channel": channel,
                    "raw_answer_code": item["Answer"],
                    "response_s": item["Time(s)"],
                    "n_hr": int(hv.notna().sum()),
                    "mean_hr_bpm": hv.mean(),
                    "n_interval_samples": int(bv.notna().sum()),
                    "n_interval_pairs": int(bv.diff().notna().sum()),
                    "interval_sd_ms": hrv.sdnn(b["ibi"]),
                    "interval_rmsdiff_ms": hrv.rmssd(b["ibi"]),
                    "n_pupil": int(pupils.notna().sum()),
                    "n_complete_low_movement_runs": len(complete),
                    "mean_complete_run_ms": complete["duration_s"].mean() * 1000,
                    "n_closure_candidates": len(closures),
                    "valid_eye_observation_s": observed,
                    "closure_candidates_per_min": len(closures) * 60 / observed
                    if observed > 0
                    else np.nan,
                    "mean_pupil_source_units": pupils.mean(),
                }
            )
    return pd.DataFrame(rows)


def aligned_questions(*, source="case-study"):
    """Opt in through documented acquisition metadata, never a timezone guess."""
    zone = os.environ.get("MMS_SENSOR_TIMEZONE") or None
    if zone is None:
        print(
            "Question/sensor alignment skipped: sensor timezone is undocumented. Set MMS_SENSOR_TIMEZONE only from acquisition metadata."
        )
        return pd.DataFrame()
    frames = [
        question_features(s, source=source, sensor_timezone=zone) for s in (1, 2, 3)
    ]
    return pd.concat(frames, ignore_index=True)


def interval_spectra(streams):
    """Descriptive Lomb-Scargle sample spectra on recorded elapsed time, not NN HRV."""
    result = {}
    frequencies = np.linspace(0.01, 0.5, 500)
    for label, frame in streams.items():
        hrv.hrv_rolling(frame, window_beats=30)
        for channel, part in _channels(frame):
            sample = _real(part["ibi"], "ibi")
            valid = np.isfinite(sample) & sample.between(300, 2000)
            time = _real(part["reltime"], "reltime")[valid].to_numpy(float)
            values = sample[valid].to_numpy(float)
            if len(values) < 10 or len(np.unique(time)) < 10 or np.ptp(values) == 0:
                continue
            time = time - time[0]
            power = lombscargle(
                time, values - values.mean(), 2 * np.pi * frequencies, normalize=True
            )
            result[f"{label}, channel {channel}"] = pd.DataFrame(
                {"Frequency (Hz)": frequencies, "Normalized sample power": power}
            )
    return result


def group_frames(metrics=None):
    return {
        metric: io.load_group_summary(metric, data_root=data_root()).set_index(
            "Participant"
        )
        for metric in (paths.GROUP_METRICS if metrics is None else metrics)
    }


def group_correlations(metrics=None):
    combined = pd.concat(
        [
            frame.add_suffix(f"_{metric}")
            for metric, frame in group_frames(metrics).items()
        ],
        axis=1,
    )
    return stats.corr_matrix_fdr(combined, method="spearman")


def paired_group_tests(frames):
    """Participant-paired changes; Holm families: all omnibus and all pair tests."""
    if not frames:
        raise ValueError("at least one repeated-session metric is required")
    pair_rows, omnibus = [], []
    for metric, frame in frames.items():
        values = (
            frame.apply(_real, name="group measurements")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if values.shape[1] != 3:
            raise ValueError("three repeated session columns are required")
        if len(values) >= 3 and np.any(np.ptp(values.to_numpy(), axis=1) > 0):
            statistic, p = scipy_stats.friedmanchisquare(
                *(values.iloc[:, i] for i in range(3))
            )
        else:
            statistic, p = np.nan, np.nan
        omnibus.append(
            {
                "Metric": metric,
                "n_participants": len(values),
                "Friedman statistic": statistic,
                "p_raw": p,
            }
        )
        for i, j in combinations(range(3), 2):
            difference = values.iloc[:, j] - values.iloc[:, i]
            if len(difference) < 2:
                p = np.nan
            elif np.all(difference == 0):
                p = 1.0
            else:
                p = scipy_stats.wilcoxon(difference).pvalue
            sd = difference.std(ddof=1)
            mean = difference.mean()
            width = (
                scipy_stats.t.ppf(0.975, len(difference) - 1)
                * sd
                / np.sqrt(len(difference))
                if len(difference) > 1
                else np.nan
            )
            pair_rows.append(
                {
                    "Metric": metric,
                    "Pair": f"{values.columns[i]} to {values.columns[j]}",
                    "n_participants": len(values),
                    "Mean paired change": mean,
                    "CI low": mean - width,
                    "CI high": mean + width,
                    "Paired d_z": mean / sd if sd > 0 else np.nan,
                    "p_raw": p,
                }
            )
    pair, omni = pd.DataFrame(pair_rows), pd.DataFrame(omnibus)
    for table in (pair, omni):
        table["p_holm"] = stats.holm(table["p_raw"].to_numpy())
    return omni, pair


def cluster_features(frame, *, n_clusters=3):
    """Exploratory feature-space clusters; PCA is only a display projection."""
    if (
        isinstance(n_clusters, (bool, np.bool_))
        or not isinstance(n_clusters, (int, np.integer))
        or n_clusters < 2
    ):
        raise ValueError("n_clusters must be an integer >=2")
    values = (
        frame.apply(_real, name="cluster features")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if (
        values.shape[1] < 2
        or len(values) <= n_clusters
        or len(values.drop_duplicates()) <= n_clusters
    ):
        raise ValueError(
            "clustering needs >=2 features and more distinct complete rows than clusters"
        )
    scaled = StandardScaler().fit_transform(values)
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(
        scaled
    )
    projection = PCA(n_components=2).fit_transform(scaled)
    return pd.DataFrame(projection, index=values.index, columns=["PC1", "PC2"]).assign(
        Cluster=labels
    ), silhouette_score(scaled, labels)


def synchronized_features(session, *, source="case-study"):
    """Native timestamp joins only after acquisition clocks are explicitly confirmed."""
    zone = os.environ.get("MMS_SENSOR_TIMEZONE")
    if not zone or os.environ.get("MMS_CLOCKS_SYNCHRONIZED") != "1":
        print(
            "Cross-sensor clustering skipped: declare sensor timezone and confirmed clock synchronization."
        )
        return pd.DataFrame()
    hr = io.load_hr(session, source=source, data_root=data_root())
    ibi = io.load_ibi(session, source=source, data_root=data_root())
    eye = io.load_raw("sed", session, source=source, data_root=data_root())
    rows = []
    for channel, part in _channels(ibi):
        h = hr.loc[hr["iSensor"].eq(channel)].copy()
        b = part.copy()
        e = eye.copy()
        for f in (h, b, e):
            f["clock"] = io.parse_datetime(
                f["datetime"], naive_timezone=zone
            ).dt.tz_convert("UTC")
        if any(f["clock"].duplicated().any() for f in (h, b, e)):
            raise ValueError(
                "synchronized feature timestamps must be unique within each stream/channel"
            )
        e["pupil"] = fixation.pupil_samples(e, quality_min=0.5)
        h["heart_rate"] = _valid_hr(h)
        b["ibi"] = _real(b["ibi"], "ibi").where(
            lambda x: np.isfinite(x) & x.between(300, 2000)
        )
        base = b[["clock", "ibi"]].merge(
            h[["clock", "heart_rate"]], on="clock", validate="one_to_one"
        )
        # Backward containment avoids assigning a future eye sample.
        base = pd.merge_asof(
            base.sort_values("clock"),
            e[["clock", "pupil"]].sort_values("clock"),
            on="clock",
            direction="backward",
            tolerance=pd.Timedelta(100, unit="ms"),
        )
        base = base.loc[
            np.isfinite(base["ibi"])
            & base["ibi"].between(300, 2000)
            & np.isfinite(base["heart_rate"])
            & (base["heart_rate"] > 0)
        ]
        base["Channel"] = channel
        rows.append(base)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def export_frame(frame, relative):
    root = os.environ.get("MMS_OUTPUT_ROOT")
    if not root:
        print("Export skipped: set MMS_OUTPUT_ROOT to a separate scratch directory.")
        return None
    path = paths.output_path(root, relative, data_root=data_root())
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def export_figure(figure, relative):
    root = os.environ.get("MMS_OUTPUT_ROOT")
    if not root:
        print(
            "Figure export skipped: set MMS_OUTPUT_ROOT to a separate scratch directory."
        )
        return None
    path = paths.output_path(root, relative, data_root=data_root())
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    return path


def convert_raw(sensor, session, *, source):
    frame = io.load_raw(sensor, session, source=source, data_root=data_root())
    suffix = "" if session is None else f"_{session:02d}"
    return export_frame(frame, f"{source}/processed/{sensor}{suffix}.csv")


def plot_summary(table, value, *, group="Recording", title=None):
    fig, ax = plt.subplots(figsize=(9, 4))
    if not table.empty:
        if "Channel" in table:
            for channel, part in table.groupby("Channel", sort=False):
                ax.plot(part[group], part[value], "o-", label=f"Channel {channel}")
            ax.legend()
        else:
            ax.bar(table[group], table[value])
    ax.set_ylabel(value)
    ax.set_title(title or value)
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_traces(streams, kind):
    fig, axes = plt.subplots(
        len(streams), 1, figsize=(10, 3 * len(streams)), squeeze=False
    )
    for ax, (label, frame) in zip(axes.flat, streams.items()):
        if kind == "eye":
            ax.plot(
                frame["reltime"],
                fixation.pupil_samples(frame, quality_min=0.5),
                linewidth=0.5,
            )
            ax.set_ylabel("Pupil (source units)")
        else:
            for channel, part in _channels(frame):
                if kind == "hr":
                    value = _valid_hr(part)
                    ax.plot(
                        part["reltime"],
                        value,
                        linewidth=0.6,
                        label=f"Channel {channel}",
                    )
                    ax.set_ylabel("HR (bpm)")
                elif kind == "ibi":
                    rolled = hrv.hrv_rolling(part, window_beats=30)
                    ax.plot(
                        rolled["reltime"],
                        rolled["rmssd"],
                        linewidth=0.6,
                        label=f"Channel {channel}",
                    )
                    ax.set_ylabel("30-sample RMS difference (ms)")
                else:
                    raise ValueError("unknown trace kind")
            ax.legend()
        ax.set_title(label)
        ax.set_xlabel("Seconds from this recording's origin")
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_events(streams, *, histogram=False):
    fig, ax = plt.subplots(figsize=(9, 4))
    tables = []
    for label, frame in streams.items():
        events = low_movement_events(frame)
        complete = events.loc[~events["left_censored"] & ~events["right_censored"]]
        durations = complete["duration_s"] * 1000
        if histogram:
            ax.hist(durations, bins=30, alpha=0.4, label=label)
        else:
            ax.scatter(np.repeat(label, len(durations)), durations, s=8, alpha=0.4)
        tables.append(
            {
                "Recording": label,
                "Complete events": len(complete),
                "Censored events": len(events) - len(complete),
                "Mean duration (ms)": durations.mean(),
            }
        )
    ax.set_ylabel("Event count" if histogram else "Run duration (ms)")
    ax.set_xlabel("Run duration (ms)" if histogram else "Recording")
    ax.set_title("Legacy low-movement runs, one observation per event")
    if histogram:
        ax.legend()
    fig.tight_layout()
    plt.show()
    plt.close(fig)
    return pd.DataFrame(tables)


def plot_group(frames, *, distribution=False):
    fig, axes = plt.subplots(
        1, len(frames), figsize=(6 * len(frames), 4), squeeze=False
    )
    for ax, (metric, frame) in zip(axes.flat, frames.items()):
        if distribution:
            ax.boxplot([frame[col] for col in frame])
            ax.set_xticks(range(1, len(frame.columns) + 1), frame.columns)
        else:
            for _, row in frame.iterrows():
                ax.plot(frame.columns, row, "o-", alpha=0.4)
            ax.plot(
                frame.columns, frame.mean(), "s--", color="black", label="Group mean"
            )
            ax.legend()
        ax.set_ylabel(GROUP_LABELS[metric])
        ax.set_title(GROUP_LABELS[metric])
        ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    plt.show()
    plt.close(fig)


def plot_correlations(result):
    r, adjusted = result["r"], result["p_fdr"]
    annot = np.empty(r.shape, dtype=object)
    for i in range(len(r)):
        for j in range(len(r)):
            p = adjusted.iloc[i, j]
            mark = "*" if pd.notna(p) and p < 0.05 else ""
            annot[i, j] = f"{r.iloc[i, j]:.2f}{mark}"
    fig, ax = plt.subplots(figsize=(12, 9))
    image = ax.imshow(r.to_numpy(float), cmap="coolwarm", vmin=-1, vmax=1)
    fig.colorbar(
        image,
        ax=ax,
        label="Spearman coefficient"
        if result["method"] == "spearman"
        else "Correlation coefficient",
    )
    ax.set_xticks(np.arange(len(r.columns)), r.columns, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(r.index)), r.index)
    for i in range(len(r.index)):
        for j in range(len(r.columns)):
            coefficient = r.iloc[i, j]
            label = annot[i, j] if pd.notna(coefficient) else "NA"
            color = (
                "white"
                if pd.notna(coefficient) and abs(coefficient) > 0.65
                else "black"
            )
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=8)
    method = result["method"].capitalize()
    adjustment = result["adjustment"]
    if adjustment == "none":
        title = f"Descriptive {method} coefficients; repeated items, p values omitted"
    else:
        title = f"{method} correlations, {result['n_tests']} tests; * {adjustment}-adjusted p<0.05"
    ax.set_title(title)
    fig.tight_layout()
    plt.show()
    plt.close(fig)
    return fig


def gaze_rate_summary(streams):
    """Source-vector displacement per elapsed second; not angular velocity."""
    rows = []
    for label, frame in streams.items():
        low_movement_events(frame)  # Shared chronological/quality validation.
        columns = [f"gazeDir.{axis}" for axis in "xyz"]
        xyz = frame[columns].apply(_real, name="gaze vector")
        quality = _real(frame["gazeQ"], "gazeQ")
        valid = (
            np.isfinite(xyz).all(axis=1)
            & np.isfinite(quality)
            & quality.between(0.5, 1)
        )
        delta = _real(frame["reltime"], "reltime").diff()
        step = np.sqrt((xyz.diff() ** 2).sum(axis=1, min_count=3))
        rate = (step / delta).where(
            valid & valid.shift(fill_value=False) & delta.gt(0) & delta.le(0.1)
        )
        rows.append(
            {
                "Recording": label,
                "n_valid_steps": int(rate.notna().sum()),
                "Mean source-vector rate (1/s)": rate.mean(),
                "Fraction of valid steps >0.1/s": rate.gt(0.1).sum()
                / rate.notna().sum()
                if rate.notna().any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def response_tables(*, source="individual"):
    durations, codes = [], []
    for session in (1, 2, 3):
        frame = io.load_psychometric(session, source=source, data_root=data_root())
        durations.append(duration_summary(frame).assign(Session=session_label(session)))
        codes.append(raw_answer_summary(frame).assign(Session=session_label(session)))
    return pd.concat(durations, ignore_index=True), pd.concat(codes, ignore_index=True)


def descriptive_correlations(frame):
    """Spearman coefficients/counts only for repeated within-person observations."""
    columns = list(frame.columns)
    if not frame.columns.is_unique:
        raise ValueError("descriptive correlation columns must be unique")
    values = [_real(frame[column], str(column)).to_numpy(float) for column in columns]
    size = len(columns)
    correlation = np.full((size, size), np.nan)
    counts = np.zeros((size, size), dtype=int)
    for i, first in enumerate(values):
        for j in range(i, size):
            second = values[j]
            valid = np.isfinite(first) & np.isfinite(second)
            left, right = first[valid], second[valid]
            counts[i, j] = counts[j, i] = len(left)
            if len(left) >= 2 and np.any(left != left[0]) and np.any(right != right[0]):
                coefficient = pd.Series(left).rank().corr(pd.Series(right).rank())
                correlation[i, j] = correlation[j, i] = coefficient

    def table(array):
        return pd.DataFrame(array, index=columns, columns=columns)

    unrun = table(np.full((size, size), np.nan))
    return {
        "r": table(correlation),
        "n_pairs": table(counts),
        "p_raw": unrun,
        "p_fdr": unrun.copy(),
        "n_tests": 0,
        "method": "spearman",
        "adjustment": "none",
        "family": "within-person repeated items; inferential tests omitted",
    }


def question_correlations(features):
    """Separate instrument/channel descriptions without pseudoreplicated p values."""
    results = {}
    columns = [
        "raw_answer_code",
        "response_s",
        "mean_hr_bpm",
        "interval_rmsdiff_ms",
        "mean_pupil_source_units",
    ]
    if not features.empty:
        for identity, part in features.groupby(["Type", "Channel"], sort=False):
            results[identity] = descriptive_correlations(part[columns])
    return results


def within_instrument_fits(features):
    """In-sample least squares of raw answer codes; no diagnostic or predictive claim."""
    rows = []
    predictors = ["mean_hr_bpm", "interval_rmsdiff_ms", "mean_pupil_source_units"]
    if features.empty:
        return pd.DataFrame()
    for (kind, channel), part in features.groupby(["Type", "Channel"], sort=False):
        valid = (
            part[predictors + ["raw_answer_code"]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(valid) <= len(predictors) + 1:
            continue
        design = np.column_stack(
            [np.ones(len(valid)), valid[predictors].to_numpy(float)]
        )
        if np.linalg.matrix_rank(design) != design.shape[1]:
            continue
        outcome = valid["raw_answer_code"].to_numpy(float)
        coefficients = np.linalg.lstsq(design, outcome, rcond=None)[0]
        residual = outcome - design @ coefficients
        total = np.sum((outcome - outcome.mean()) ** 2)
        rows.append(
            {
                "Type": kind,
                "Channel": channel,
                "n_items": len(valid),
                "In-sample R2 of raw codes": 1 - np.sum(residual**2) / total
                if total > 0
                else np.nan,
                **dict(zip(["Intercept", *predictors], coefficients)),
            }
        )
    return pd.DataFrame(rows)


def baseline_changes(table, values):
    """One recorded baseline; differences retain each feature's original units."""
    keys = ["Channel"] if "Channel" in table else []
    baseline = table.loc[table["Recording"] == "Baseline", keys + list(values)]
    if keys:
        joined = table.merge(
            baseline, on=keys, suffixes=("", "_baseline"), validate="many_to_one"
        )
    else:
        if len(baseline) != 1:
            raise ValueError("exactly one recorded baseline is required")
        joined = table.copy()
        for value in values:
            joined[f"{value}_baseline"] = baseline.iloc[0][value]
    for value in values:
        joined[f"Change in {value}"] = joined[value] - joined[f"{value}_baseline"]
    return joined


def derive_eye(frame):
    """Retain every source row; event durations stay in a separate event table."""
    events = low_movement_events(frame)
    samples = frame.copy()
    samples["legacy_low_movement"] = False
    samples["event_id"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    for event in events.itertuples(index=False):
        positions = np.arange(event.start_pos, event.end_pos + 1)
        samples.iloc[positions, samples.columns.get_loc("legacy_low_movement")] = True
        samples.iloc[positions, samples.columns.get_loc("event_id")] = event.event_id
    return samples, events
