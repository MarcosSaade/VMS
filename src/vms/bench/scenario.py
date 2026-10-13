"""End-to-end benchmark scenarios (`bench/scenarios/*.yaml` with `kind: e2e`).

A scenario is a list of steps run in order, repeated `repeats` times (each repetition is a full
pass over the steps, so slow drift of the host shows up as variance rather than as a trend
across steps). A step is one point of the load space: `cameras` virtual cameras at `fps` from
loadgen (0: no ingest load) and `concurrency` closed-loop query clients (0: no query load).
The steps are every combination of `ingest.cameras` and `query.concurrency`, cameras outermost.

Every step runs `warmup_s + hold_s`. Only the last `hold_s` (the steady-state window) is
measured: loadgen's own summary, the query samples and the Prometheus window all exclude the
warmup.

Example (an ingest ramp with a small query load at each step):

    name: ramp-cpu
    kind: e2e
    repeats: 3
    warmup_s: 10
    hold_s: 60
    ingest:
      cameras: [1, 2, 4, 8]
      fps: 5
      loadgen: {clips: {ids: [mot17-09-mini]}}   # overrides configs/loadgen.yaml
    query:
      concurrency: [1]
      questions: configs/demo/canned_plans.v1.jsonl
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

KIND = "e2e"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IngestLoad(_Section):
    cameras: list[int] = Field(min_length=1)
    fps: float = Field(default=5.0, gt=0, le=120)
    # Overrides for configs/loadgen.yaml (clips, frame, http, ...), as in a loadgen scenario.
    loadgen: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _positive(self) -> IngestLoad:
        if any(c < 1 for c in self.cameras):
            raise ValueError("ingest.cameras must be positive; leave out `ingest` for none")
        reserved = {"cameras", "fps", "duration_s", "warmup_s", "target_url", "mode", "seed"}
        if clash := reserved & self.loadgen.keys():
            raise ValueError(f"ingest.loadgen must not set {sorted(clash)}: the runner sets them")
        return self


class QueryLoad(_Section):
    concurrency: list[int] = Field(min_length=1)
    # JSONL with a "question" field per line (other fields are ignored).
    questions: Path = Path("configs/demo/canned_plans.v1.jsonl")
    # Pause per client between a reply and its next question; 0 is a closed loop at full speed.
    think_time_s: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _positive(self) -> QueryLoad:
        if any(c < 1 for c in self.concurrency):
            raise ValueError("query.concurrency must be positive; leave out `query` for none")
        return self


class Step(_Section):
    index: int
    cameras: int  # 0: no ingest load
    concurrency: int  # 0: no query load


class BenchScenario(_Section):
    name: str
    description: str = ""
    kind: Literal["e2e"] = "e2e"
    repeats: int = Field(default=3, ge=1)
    seed: int | None = None  # None: the settings' seed
    warmup_s: float = Field(default=10.0, ge=0)
    hold_s: float = Field(default=60.0, gt=0)
    ingest: IngestLoad | None = None
    query: QueryLoad | None = None

    @model_validator(mode="after")
    def _some_load(self) -> BenchScenario:
        if self.ingest is None and self.query is None:
            raise ValueError("a scenario needs `ingest`, `query` or both")
        return self

    def steps(self) -> list[Step]:
        cameras = self.ingest.cameras if self.ingest else [0]
        concurrency = self.query.concurrency if self.query else [0]
        return [
            Step(index=i, cameras=c, concurrency=q)
            for i, (c, q) in enumerate(itertools.product(cameras, concurrency))
        ]

    @property
    def step_s(self) -> float:
        return self.warmup_s + self.hold_s


def read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: a scenario must be a mapping")
    return data


def load_questions(path: Path) -> list[str]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            questions.append(str(json.loads(line)["question"]))
    if not questions:
        raise ValueError(f"{path}: no questions")
    return questions
