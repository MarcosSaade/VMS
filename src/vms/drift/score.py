"""Offline scoring of drift detectors over stored drift runs (task 9.3).

`python -m vms.drift score --scenario bench/scenarios/drift-detect-cpu.yaml`:

1. **Series.** For every repetition of every listed loadgen run (bench/results, with its
   recorded cameras and drift onsets), the camera's stored frames and detections are read
   back from Postgres and replayed exactly as the live monitor sees them: a reference from
   the camera's first `reference_s`, then a window of `window_s` ending every
   `update_every_s`, compared with the same functions (vms.drift.signals). Each camera
   becomes a stream of channel values (vms.drift.detect), cached under `cache_dir`.
2. **Folds.** Repetitions are the unit of independence (each has its own seed and onsets).
   Leave-one-repetition-out: for each fold, the clean streams of the other repetitions
   calibrate every channel (mean and std on clean data) and choose each detector family's
   operating point; the held-out repetition is scored with it. Every repetition is held out
   once, so every reported operating-point number is out of sample.
3. **Sweep.** Each detector configuration (fixed threshold h; CUSUM allowance k and decision
   interval h) is scored on all repetitions (in sample, calibrated on all clean streams) for
   the delay against false-alarm curve.

Scoring a stream: the detector runs from the first window after the reference. An alarm
before the stream's onset (or anywhere on a clean stream) is a false alarm, and resets the
camera's detector (an operator acknowledging it). The first alarm at or after the onset is
the detection; its delay is alarm time minus onset (the window end, in capture time; the
live monitor adds `settle_s` and the scrape interval). No alarm after the onset is a miss.
False alarms per camera-hour = false alarms / clean monitored hours (clean streams and the
pre-onset part of drifted ones).

Operating point rule (fixed before scoring): the configuration with no false alarm on the
training repetitions that misses the fewest drifted streams, then has the lowest median
delay. With zero false alarms in T clean hours, the 95% upper bound of the rate is 3 / T
("rule of three"), reported with every zero.
"""

from __future__ import annotations

import bisect
import itertools
import json
import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from vms.bench.results import RunResult, git_info, hardware, package_versions, write_result
from vms.common.ids import new_ulid
from vms.common.logs import get_logger
from vms.drift.detect import (
    CHANNELS,
    Calibration,
    CameraDetector,
    DetectorSettings,
    channel_values,
)
from vms.drift.settings import MonitorSettings
from vms.drift.signals import (
    DetectionRow,
    FrameRow,
    Sample,
    SignalSettings,
    compare,
    coverage,
)
from vms.drift.store import DriftStore

log = get_logger(__name__)


THRESHOLD_H = tuple(float(h) for h in (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 70, 100))
CUSUM_H = tuple(float(h) for h in (2, 3, 5, 8, 10, 15, 20, 30, 50, 80, 120, 200, 300))


class SweepSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    threshold_h: list[float] = Field(default_factory=lambda: list(THRESHOLD_H))
    cusum_k: list[float] = Field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0])
    cusum_h: list[float] = Field(default_factory=lambda: list(CUSUM_H))


class ScoreScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    kind: str = "drift-score"
    runs: list[Path] = Field(min_length=1)  # loadgen results JSONs of the drift scenarios
    monitor: MonitorSettings = Field(default_factory=MonitorSettings)
    signals: SignalSettings = Field(default_factory=SignalSettings)
    # Floors of the clean std per channel (vms.drift.detect.Calibration.fit).
    min_std: dict[str, float] = Field(
        default_factory=lambda: {
            **{c: 0.1 for c in CHANNELS if c.startswith("shift:")},
            "confidence_ks": 0.005,
            "confidence_wasserstein": 0.002,
            "class_psi": 0.002,
        }
    )
    sweep: SweepSettings = Field(default_factory=SweepSettings)
    cache_dir: Path = Path("data/drift/series")
    plot: Path | None = Path("docs/images/drift-delay-vs-false-alarms.png")


def load_scenario(path: Path) -> ScoreScenario:
    return ScoreScenario.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass
class Stream:
    scenario: str
    rep: int
    seed: int
    camera_id: str
    clip: str
    kind: str | None  # drift kind; None for a clean camera
    ramp_s: float
    onset_s: float | None  # after the repetition start
    times_s: list[float] = field(default_factory=list)  # window ends after the start
    values: list[dict[str, float]] = field(default_factory=list)

    @property
    def mode(self) -> str:
        if self.kind is None:
            return "clean"
        return "abrupt" if self.ramp_s == 0 else "gradual"

    @property
    def clean_s(self) -> float:
        """Monitored time before the onset (all of it on a clean stream)."""
        if not self.times_s:
            return 0.0
        end = self.times_s[-1] if self.onset_s is None else min(self.onset_s, self.times_s[-1])
        return max(0.0, end - self.times_s[0])


# --- 1. Series ------------------------------------------------------------------------------


def _slice(
    frames: Sequence[FrameRow], keys: Sequence[datetime], a: datetime, b: datetime
) -> Sequence[FrameRow]:
    return frames[bisect.bisect_left(keys, a) : bisect.bisect_left(keys, b)]


def camera_series(
    frames: Sequence[FrameRow],
    detections: Sequence[DetectionRow],
    start: datetime,
    monitor: MonitorSettings,
    signals: SignalSettings,
) -> tuple[list[float], list[dict[str, float]]]:
    """Window ends (seconds after `start`) and channel values, replaying the live monitor."""
    if not frames:
        return [], []
    keys = [f.ts for f in frames]
    by_frame: dict[str, list[DetectionRow]] = {}
    for d in detections:
        by_frame.setdefault(d.frame_id, []).append(d)

    def sample(a: datetime, b: datetime) -> Sample:
        rows = _slice(frames, keys, a, b)
        dets = [d for f in rows for d in by_frame.get(f.frame_id, ())]
        return Sample.build(rows, dets, signals)

    ref_start = frames[0].ts
    ref_end = ref_start + timedelta(seconds=monitor.reference_s)
    reference = sample(ref_start, ref_end)
    times, values = [], []
    t = ref_end + timedelta(seconds=monitor.window_s)
    last = frames[-1].ts
    while t <= last + timedelta(seconds=1e-6):
        a = t - timedelta(seconds=monitor.window_s)
        stats = compare(sample(a, t), reference, signals)
        v = channel_values(stats)
        if coverage(_slice(frames, keys, a, t), monitor.window_s) < monitor.detect_min_coverage:
            v = dict.fromkeys(v, math.nan)  # as the live monitor: not fed to the detector
        times.append((t - start).total_seconds())
        values.append(v)
        t += timedelta(seconds=monitor.update_every_s)
    return times, values


async def extract(
    run_paths: Sequence[Path], store: DriftStore, scenario: ScoreScenario
) -> list[Stream]:
    """Every camera stream of every repetition, from the cache or the database."""
    streams: list[Stream] = []
    for path in run_paths:
        run = json.loads(path.read_text(encoding="utf-8"))
        name = run["name"]
        for rep_index, rep in enumerate(run["metrics"]["repetitions"]):
            start = datetime.fromisoformat(rep["started_at"]).astimezone(UTC)
            end = start + timedelta(seconds=rep["elapsed_s"] + 1)
            events = {e["camera_id"]: e for e in (rep.get("drift") or {}).get("events", [])}
            cameras = rep.get("cameras") or {
                f"{run['config']['camera_prefix']}{i + 1:03d}": "?"
                for i in range(run["config"]["cameras"])
            }
            cache = scenario.cache_dir / f"{run['run_id']}-r{rep_index}.json"
            cached = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else None
            for camera_id, clip in sorted(cameras.items()):
                e = events.get(camera_id)
                stream = Stream(
                    scenario=name,
                    rep=rep_index,
                    seed=rep["seed"],
                    camera_id=camera_id,
                    clip=clip,
                    kind=e["kind"] if e else None,
                    ramp_s=float(e["ramp_s"]) if e else 0.0,
                    onset_s=float(e["onset_s"]) if e else None,
                )
                if cached is not None and camera_id in cached:
                    stream.times_s = cached[camera_id]["times_s"]
                    stream.values = cached[camera_id]["values"]
                else:
                    frames = await store.frames([camera_id], start, end)
                    dets = await store.detections(
                        [camera_id], start, end, scenario.signals.min_conf
                    )
                    stream.times_s, stream.values = camera_series(
                        frames, dets, start, scenario.monitor, scenario.signals
                    )
                    if not stream.times_s:
                        # Not "no drift": its frames were never processed (or were deleted).
                        raise ValueError(
                            f"{name} rep {rep_index}: no processed frames of {camera_id} "
                            f"between {start.isoformat()} and {end.isoformat()}"
                        )
                streams.append(stream)
            if cached is None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    s.camera_id: {"times_s": s.times_s, "values": s.values}
                    for s in streams
                    if s.scenario == name and s.rep == rep_index
                }
                cache.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
            log.info("series ready", extra={"run": name, "rep": rep_index})
    return streams


# --- 2. Scoring -----------------------------------------------------------------------------


@dataclass
class StreamScore:
    false_alarms: int
    clean_s: float
    delay_s: float | None  # None: clean stream, or missed
    missed: bool
    channels: list[str]  # channels that fired at the detection


def score_stream(stream: Stream, settings: DetectorSettings, cal: Calibration) -> StreamScore:
    det = CameraDetector(settings, cal)
    false_alarms = 0
    for t, values in zip(stream.times_s, stream.values, strict=True):
        fired = det.update(values)
        if not fired:
            continue
        if stream.onset_s is None or t < stream.onset_s:
            false_alarms += 1
            det.reset()
            continue
        return StreamScore(false_alarms, stream.clean_s, t - stream.onset_s, False, fired)
    drifted = stream.onset_s is not None
    return StreamScore(false_alarms, stream.clean_s, None, drifted, [])


@dataclass
class Summary:
    false_alarms: int
    clean_hours: float
    drifted: int
    missed: int
    delays_s: list[float]

    @property
    def fa_per_hour(self) -> float:
        return self.false_alarms / self.clean_hours if self.clean_hours else math.nan

    @property
    def fa_upper_95(self) -> float:
        """Poisson 95% upper bound; the rule of three when there is no false alarm."""
        if not self.clean_hours:
            return math.nan
        return _poisson_upper(self.false_alarms) / self.clean_hours

    @property
    def median_delay_s(self) -> float | None:
        return statistics.median(self.delays_s) if self.delays_s else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "false_alarms": self.false_alarms,
            "clean_hours": round(self.clean_hours, 4),
            "fa_per_hour": self.fa_per_hour,
            "fa_per_hour_upper_95": self.fa_upper_95,
            "drifted": self.drifted,
            "missed": self.missed,
            "median_delay_s": self.median_delay_s,
            "delay_s_p90": _quantile(self.delays_s, 0.9),
            "max_delay_s": max(self.delays_s) if self.delays_s else None,
        }


def _poisson_upper(n: int) -> float:
    # Exact one-sided 95% upper limits for a Poisson count (Garwood), n = 0..10.
    table = [3.00, 4.74, 6.30, 7.75, 9.15, 10.51, 11.84, 13.15, 14.43, 15.71, 16.96]
    return table[n] if n < len(table) else n + 1.645 * math.sqrt(n) + 1.0


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, math.ceil(q * len(s)) - 1)]


def summarize(scores: Iterable[StreamScore]) -> Summary:
    scores = list(scores)
    return Summary(
        false_alarms=sum(s.false_alarms for s in scores),
        clean_hours=sum(s.clean_s for s in scores) / 3600,
        drifted=sum(1 for s in scores if s.missed or s.delay_s is not None),
        missed=sum(1 for s in scores if s.missed),
        delays_s=[s.delay_s for s in scores if s.delay_s is not None],
    )


def configurations(sweep: SweepSettings) -> list[DetectorSettings]:
    out = [DetectorSettings(kind="threshold", h=h) for h in sweep.threshold_h]
    out += [
        DetectorSettings(kind="cusum", k=k, h=h)
        for k, h in itertools.product(sweep.cusum_k, sweep.cusum_h)
    ]
    return out


def calibrate(streams: Iterable[Stream], min_std: dict[str, float]) -> Calibration:
    """Channel means and stds over the clean streams (cameras without drift)."""
    samples: dict[str, list[float]] = {c: [] for c in CHANNELS}
    for s in streams:
        if s.kind is not None:
            continue
        for values in s.values:
            for c in CHANNELS:
                samples[c].append(values.get(c, math.nan))
    return Calibration.fit(samples, min_std)


def choose(
    streams: Sequence[Stream], configs: Sequence[DetectorSettings], cal: Calibration
) -> DetectorSettings | None:
    """The operating point rule: no false alarm, fewest misses, then lowest median delay."""
    best: tuple[int, float, DetectorSettings] | None = None
    for cfg in configs:
        s = summarize(score_stream(st, cfg, cal) for st in streams)
        if s.false_alarms:
            continue
        key = (s.missed, s.median_delay_s if s.median_delay_s is not None else math.inf, cfg)
        if best is None or key[:2] < best[:2]:
            best = key
    return best[2] if best else None


def _label(cfg: DetectorSettings) -> str:
    return f"threshold h={cfg.h:g}" if cfg.kind == "threshold" else f"cusum k={cfg.k:g} h={cfg.h:g}"


def _by_drift(streams: Sequence[Stream], scores: Sequence[StreamScore]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    groups: dict[str, list[tuple[Stream, StreamScore]]] = {}
    for st, sc in zip(streams, scores, strict=True):
        if st.kind is None:
            continue
        groups.setdefault(f"{st.kind}/{st.mode}", []).append((st, sc))
    for key, items in sorted(groups.items()):
        firing: dict[str, int] = {}
        for _, sc in items:
            for c in sc.channels:
                firing[c] = firing.get(c, 0) + 1
        out[key] = {
            "streams": len(items),
            "missed": sum(1 for _, sc in items if sc.missed),
            "delays_s": [
                round(sc.delay_s, 1) if sc.delay_s is not None else None for _, sc in items
            ],
            "seeds": [st.seed for st, _ in items],
            "median_delay_s": statistics.median(d)
            if (d := [sc.delay_s for _, sc in items if sc.delay_s is not None])
            else None,
            "first_channels": dict(sorted(firing.items(), key=lambda kv: -kv[1])),
        }
    return out


def score(streams: list[Stream], scenario: ScoreScenario) -> dict[str, Any]:
    configs = configurations(scenario.sweep)
    reps = sorted({s.rep for s in streams})
    # In sample: the curve.
    cal_all = calibrate(streams, scenario.min_std)
    sweep = []
    for cfg in configs:
        scores = [score_stream(st, cfg, cal_all) for st in streams]
        s = summarize(scores)
        per_seed = {}
        for rep in reps:
            sub = [sc for st, sc in zip(streams, scores, strict=True) if st.rep == rep]
            per_seed[str(rep)] = summarize(sub).as_dict()
        sweep.append(
            {
                "detector": cfg.model_dump(exclude={"channels"}),
                "label": _label(cfg),
                **s.as_dict(),
                "per_rep": per_seed,
            }
        )
    # Out of sample: leave one repetition out, per detector family.
    folds = []
    held: dict[str, list[tuple[Stream, StreamScore]]] = {"threshold": [], "cusum": []}
    for rep in reps:
        train = [s for s in streams if s.rep != rep]
        test = [s for s in streams if s.rep == rep]
        cal = calibrate(train, scenario.min_std)
        fold: dict[str, Any] = {"held_out_rep": rep, "calibration": cal.model_dump()}
        for family in ("threshold", "cusum"):
            chosen = choose(train, [c for c in configs if c.kind == family], cal)
            if chosen is None:
                fold[family] = None
                continue
            scores = [score_stream(st, chosen, cal) for st in test]
            held[family] += list(zip(test, scores, strict=True))
            fold[family] = {
                "detector": chosen.model_dump(exclude={"channels"}),
                "label": _label(chosen),
                "train": summarize(score_stream(st, chosen, cal) for st in train).as_dict(),
                "test": summarize(scores).as_dict(),
            }
        folds.append(fold)
    out_of_sample = {
        family: {
            **summarize(sc for _, sc in items).as_dict(),
            "by_drift": _by_drift([st for st, _ in items], [sc for _, sc in items]),
        }
        for family, items in held.items()
        if items
    }
    return {
        "streams": {
            "total": len(streams),
            "drifted": sum(1 for s in streams if s.kind is not None),
            "clean_hours": round(sum(s.clean_s for s in streams) / 3600, 4),
            "updates": sum(len(s.times_s) for s in streams),
            "repetitions": reps,
        },
        "calibration_all": cal_all.model_dump(),
        "sweep": sweep,
        "folds": folds,
        "out_of_sample": out_of_sample,
    }


# --- 3. Plot and results --------------------------------------------------------------------


def _front(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """The configurations no other one beats on both false alarms and `key` (a delay)."""
    out: list[dict[str, Any]] = []
    for r in sorted(rows, key=lambda r: (r["fa_per_hour"], r[key])):
        if not out or r[key] < out[-1][key]:
            out.append(r)
    return out


def plot(metrics: dict[str, Any], path: Path) -> None:
    """Delay against false alarms, one Pareto front per detector family: median delay (spread:
    the per-repetition medians) and p90 delay (dominated by gradual drift)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = [("threshold", None, "Fixed threshold", "#d1495b")] + [
        ("cusum", k, f"CUSUM k={k:g}", c)
        for k, c in zip(
            sorted(
                {r["detector"]["k"] for r in metrics["sweep"] if r["detector"]["kind"] == "cusum"}
            ),
            ["#00798c", "#30638e", "#003d5b", "#66a182"],
            strict=False,
        )
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), dpi=110, sharex=True)
    for ax, key, title in [
        (axes[0], "median_delay_s", "Median delay (bars: range of per-repetition medians)"),
        (axes[1], "delay_s_p90", "p90 delay (mostly gradual drift)"),
    ]:
        for kind, k, label, color in families:
            rows = [
                r
                for r in metrics["sweep"]
                if r["detector"]["kind"] == kind
                and (k is None or r["detector"]["k"] == k)
                and r[key] is not None
                and r["missed"] == 0
            ]
            rows = _front(rows, key)
            if not rows:
                continue
            if rows[0]["false_alarms"] == 0:  # name the setting at the zero false-alarm end
                setting = rows[0]["label"].split(" ", 1)[1]
                label = f"{label} ({setting.split()[-1]} at 0 FA)"
            xs = [r["fa_per_hour"] for r in rows]
            ys = [r[key] for r in rows]
            if key == "median_delay_s":
                lo, hi = [], []
                for r in rows:
                    meds = [p[key] for p in r["per_rep"].values() if p[key] is not None] or [r[key]]
                    lo.append(r[key] - min(meds))
                    hi.append(max(meds) - r[key])
                ax.errorbar(
                    xs,
                    ys,
                    yerr=[lo, hi],
                    marker="o",
                    ms=4,
                    lw=1.5,
                    capsize=2,
                    label=label,
                    color=color,
                    drawstyle="steps-post",
                )
            else:
                ax.plot(
                    xs,
                    ys,
                    marker="o",
                    ms=4,
                    lw=1.5,
                    label=label,
                    color=color,
                    drawstyle="steps-post",
                )
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_xlabel("False alarms per camera-hour (clean streams)")
        ax.set_ylabel("Delay after onset (s)")
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.3)
    for ax in axes:
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Drift detection: delay vs false alarms (Pareto front per detector, no missed drift)",
        fontsize=10,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


async def run_score(
    scenario: ScoreScenario, store: DriftStore, root: Path, results_dir: Path
) -> Path:
    started = datetime.now(UTC)
    streams = await extract([root / p for p in scenario.runs], store, scenario)
    metrics = score(streams, scenario)
    if scenario.plot is not None:
        plot(metrics, root / scenario.plot)
        metrics["plot"] = str(scenario.plot)
    metrics["stream_list"] = [
        {k: v for k, v in asdict(s).items() if k not in ("times_s", "values")}
        | {"updates": len(s.times_s)}
        for s in streams
    ]
    result = RunResult(
        run_id=new_ulid(),
        name=scenario.name,
        kind="drift.score",
        started_at=started,
        ended_at=datetime.now(UTC),
        seed=0,  # nothing random: the series and the sweep are deterministic
        git=git_info(root),
        hardware=hardware(),
        packages=package_versions(("numpy", "matplotlib")),
        config=scenario.model_dump(mode="json"),
        metrics=metrics,
        notes="Detection delay is window end minus onset in capture time (the live monitor "
        "adds settle_s and the scrape interval). Operating points are chosen on the other "
        "repetitions (leave one repetition out); the sweep is in sample. Clips loop every 17 "
        "to 35 s, so clean windows are more alike than on a real camera and false-alarm rates "
        "are optimistic.",
    )
    path = write_result(root / results_dir, result)
    log.info("drift score written", extra={"path": str(path)})
    return path
