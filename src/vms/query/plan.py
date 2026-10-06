"""`QueryPlan`: the typed intermediate representation the LLM planner emits instead of SQL.

Draft (`schema_version` 0.1, task 0.6); task 1.7 finalizes it together with the compiler.

Why a typed plan: the output space is bounded, so validation is exhaustive and user values are
always bound as SQL parameters; planner accuracy does not depend on the SQL dialect; and
training targets are short and checkable (ARCHITECTURE 4.7, D11). Questions the schema cannot
express must abstain with reason `unsupported`.

This module checks only what is decidable from the plan alone (structure and field
combinations). Checks against live data (known cameras, zones and classes, time range sanity,
limit cap from config) are the semantic validation step in task 1.7.

Times: every time in a plan is camera-local wall-clock time. The compiler resolves relative and
absolute ranges in each camera's IANA timezone with the injected clock, never the LLM.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION: Final = "0.1"
PLAN_LIMIT_MAX = 1000  # hard schema bound; the configured row cap may be lower
MAX_FILTER_VALUES = 20
MAX_VISUAL_QUERY_CHARS = 200


class Intent(StrEnum):
    COUNT = "count"  # one number (or one per group)
    AGGREGATE = "aggregate"  # avg/min/max/sum of a measure (or per group)
    LIST = "list"  # rows of tracks, events or detections
    TREND = "trend"  # a count per time bucket
    TOP_K = "top_k"  # the k groups with the highest count
    DWELL = "dwell"  # time spent by tracks on camera, or inside zones
    FRAMES = "frames"  # matching frames; with visual_query, ranked by CLIP (hybrid)
    ABSTAIN = "abstain"


class AbstainReason(StrEnum):
    AMBIGUOUS = "ambiguous"  # several readings with different answers; ask the user
    UNANSWERABLE = "unanswerable"  # the data does not contain the answer
    UNSUPPORTED = "unsupported"  # answerable with SQL, but outside this plan schema
    DISALLOWED = "disallowed"  # a request to modify data, read secrets, or similar


class Target(StrEnum):
    """What the plan counts or lists."""

    TRACKS = "tracks"  # distinct tracked objects ("how many people")
    DETECTIONS = "detections"  # per-frame boxes
    EVENTS = "events"  # zone_enter, zone_exit, loiter
    FRAMES = "frames"


class EventType(StrEnum):
    ZONE_ENTER = "zone_enter"
    ZONE_EXIT = "zone_exit"
    LOITER = "loiter"


class GroupKey(StrEnum):
    CAMERA = "camera"
    ZONE = "zone"
    CLASS_NAME = "class_name"
    EVENT_TYPE = "event_type"


class TimeBucket(StrEnum):
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"


class MeasureOp(StrEnum):
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    SUM = "sum"


class MeasureField(StrEnum):
    CONFIDENCE = "confidence"  # detections
    N_FRAMES = "n_frames"  # tracks
    DWELL_SECONDS = "dwell_seconds"  # dwell intent only


class OrderBy(StrEnum):
    COUNT = "count"
    VALUE = "value"  # the measure
    TIME = "time"


class Direction(StrEnum):
    ASC = "asc"
    DESC = "desc"


class TimeUnit(StrEnum):
    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"


class NamedPeriod(StrEnum):
    """Calendar periods in camera-local time. Boundaries are defined by the compiler (1.7):
    morning 06-12, afternoon 12-18, evening 18-24, last_night yesterday 18:00 to today 06:00,
    weeks start on Monday."""

    TODAY = "today"
    YESTERDAY = "yesterday"
    THIS_MORNING = "this_morning"
    THIS_AFTERNOON = "this_afternoon"
    THIS_EVENING = "this_evening"
    LAST_NIGHT = "last_night"
    THIS_WEEK = "this_week"
    LAST_WEEK = "last_week"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, use_enum_values=False)


class AbsoluteTimeRange(_Model):
    """Half-open [start, end) in camera-local wall-clock time (no UTC offset)."""

    kind: Literal["absolute"] = "absolute"
    start: dt.datetime
    end: dt.datetime

    @model_validator(mode="after")
    def _check(self) -> AbsoluteTimeRange:
        if self.start.tzinfo is not None or self.end.tzinfo is not None:
            raise ValueError("absolute times are camera-local: omit the UTC offset")
        if self.start >= self.end:
            raise ValueError("time_range.start must be before time_range.end")
        return self


class LastTimeRange(_Model):
    """The trailing window ending now, e.g. the last 2 hours."""

    kind: Literal["last"] = "last"
    amount: int = Field(ge=1, le=10_000)
    unit: TimeUnit


class NamedTimeRange(_Model):
    kind: Literal["named"] = "named"
    name: NamedPeriod


TimeRange = Annotated[
    AbsoluteTimeRange | LastTimeRange | NamedTimeRange, Field(discriminator="kind")
]


class Filters(_Model):
    """All filters are ANDed; values inside one list are ORed. Names, not IDs."""

    cameras: list[str] = Field(default_factory=list, max_length=MAX_FILTER_VALUES)
    zones: list[str] = Field(default_factory=list, max_length=MAX_FILTER_VALUES)
    classes: list[str] = Field(default_factory=list, max_length=MAX_FILTER_VALUES)
    event_types: list[EventType] = Field(default_factory=list, max_length=len(EventType))
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    time_range: TimeRange | None = None

    def is_empty(self) -> bool:
        return self == Filters()


class Measure(_Model):
    op: MeasureOp
    field: MeasureField


class Order(_Model):
    by: OrderBy
    direction: Direction = Direction.DESC


# Which targets each intent accepts.
_INTENT_TARGETS: dict[Intent, frozenset[Target]] = {
    Intent.COUNT: frozenset(Target),
    Intent.AGGREGATE: frozenset({Target.DETECTIONS, Target.TRACKS}),
    Intent.LIST: frozenset({Target.TRACKS, Target.EVENTS, Target.DETECTIONS}),
    Intent.TREND: frozenset(Target),
    Intent.TOP_K: frozenset({Target.TRACKS, Target.EVENTS, Target.DETECTIONS}),
    Intent.DWELL: frozenset({Target.TRACKS}),
    Intent.FRAMES: frozenset({Target.FRAMES}),
}
_MEASURE_FIELD_TARGET = {
    MeasureField.CONFIDENCE: Target.DETECTIONS,
    MeasureField.N_FRAMES: Target.TRACKS,
}
_ORDER_INTENTS: dict[OrderBy, frozenset[Intent]] = {
    OrderBy.COUNT: frozenset({Intent.COUNT, Intent.TOP_K, Intent.TREND}),
    OrderBy.VALUE: frozenset({Intent.AGGREGATE, Intent.DWELL}),
    OrderBy.TIME: frozenset({Intent.LIST, Intent.FRAMES, Intent.TREND}),
}
# Trend may group too: one series per group (hourly people count per camera).
_GROUPABLE_INTENTS = frozenset(
    {Intent.COUNT, Intent.AGGREGATE, Intent.TOP_K, Intent.DWELL, Intent.TREND}
)


class QueryPlan(_Model):
    """A structured question. Every error found is reported at once, so a single repair
    attempt can fix all of them."""

    schema_version: Literal["0.1"] = SCHEMA_VERSION
    intent: Intent
    target: Target | None = None
    filters: Filters = Field(default_factory=Filters)
    measure: Measure | None = None
    group_by: list[GroupKey] = Field(default_factory=list, max_length=2)
    bucket: TimeBucket | None = None
    order: Order | None = None
    limit: int | None = Field(default=None, ge=1, le=PLAN_LIMIT_MAX)
    visual_query: str | None = Field(default=None, min_length=1, max_length=MAX_VISUAL_QUERY_CHARS)
    abstain_reason: AbstainReason | None = None

    @model_validator(mode="after")
    def _check_combinations(self) -> QueryPlan:
        errors = self._abstain_errors() if self.intent is Intent.ABSTAIN else self._plan_errors()
        if errors:
            raise ValueError("; ".join(errors))
        return self

    def _abstain_errors(self) -> list[str]:
        errors = []
        if self.abstain_reason is None:
            errors.append("intent 'abstain' requires abstain_reason")
        extra = [
            name
            for name in ("target", "measure", "bucket", "order", "limit", "visual_query")
            if getattr(self, name) is not None
        ]
        if self.group_by:
            extra.append("group_by")
        if not self.filters.is_empty():
            extra.append("filters")
        if extra:
            errors.append(f"intent 'abstain' takes no other fields (got {', '.join(extra)})")
        return errors

    def _plan_errors(self) -> list[str]:
        errors: list[str] = []
        intent, target, f = self.intent, self.target, self.filters

        if self.abstain_reason is not None:
            errors.append("abstain_reason is only allowed with intent 'abstain'")
        if target is None:
            errors.append(f"intent '{intent}' requires target")
        elif target not in _INTENT_TARGETS[intent]:
            allowed = ", ".join(sorted(_INTENT_TARGETS[intent]))
            errors.append(f"intent '{intent}' takes target {allowed}, not '{target}'")

        # Measure: aggregate and dwell only, with a field that matches the target.
        if intent in (Intent.AGGREGATE, Intent.DWELL):
            if self.measure is None:
                errors.append(f"intent '{intent}' requires measure")
            elif intent is Intent.DWELL and self.measure.field is not MeasureField.DWELL_SECONDS:
                errors.append("intent 'dwell' requires measure.field 'dwell_seconds'")
            elif intent is Intent.AGGREGATE:
                needed = _MEASURE_FIELD_TARGET.get(self.measure.field)
                if needed is None:
                    errors.append("use intent 'dwell' for dwell_seconds")
                elif target is not None and target is not needed:
                    errors.append(
                        f"measure.field '{self.measure.field}' requires target '{needed}'"
                    )
        elif self.measure is not None:
            errors.append(
                f"measure is only allowed with intents aggregate and dwell, not '{intent}'"
            )

        if intent is Intent.TREND and self.bucket is None:
            errors.append("intent 'trend' requires bucket")
        if intent is not Intent.TREND and self.bucket is not None:
            errors.append("bucket is only allowed with intent 'trend'")

        if intent is Intent.TOP_K:
            if len(self.group_by) != 1:
                errors.append("intent 'top_k' requires exactly one group_by key")
            if self.limit is None:
                errors.append("intent 'top_k' requires limit (the k)")
        if self.group_by and intent not in _GROUPABLE_INTENTS:
            errors.append(f"group_by is not allowed with intent '{intent}'")
        if len(set(self.group_by)) != len(self.group_by):
            errors.append("group_by keys must be unique")

        if self.visual_query is not None and intent is not Intent.FRAMES:
            errors.append("visual_query is only allowed with intent 'frames'")
        if self.order is not None and intent not in _ORDER_INTENTS[self.order.by]:
            errors.append(f"order.by '{self.order.by}' is not valid for intent '{intent}'")

        if target is not None:
            errors.extend(self._target_filter_errors(target, f))
        return errors

    def _target_filter_errors(self, target: Target, f: Filters) -> list[str]:
        errors = []
        if f.event_types and target is not Target.EVENTS:
            errors.append("filters.event_types requires target 'events'")
        if f.zones and target not in (Target.EVENTS, Target.TRACKS):
            errors.append("filters.zones requires target 'events' or 'tracks'")
        if f.min_confidence is not None and target not in (Target.DETECTIONS, Target.FRAMES):
            errors.append("filters.min_confidence requires target 'detections' or 'frames'")
        for key in self.group_by:
            if key is GroupKey.EVENT_TYPE and target is not Target.EVENTS:
                errors.append("group_by 'event_type' requires target 'events'")
            if key is GroupKey.ZONE and target not in (Target.EVENTS, Target.TRACKS):
                errors.append("group_by 'zone' requires target 'events' or 'tracks'")
            if key is GroupKey.CLASS_NAME and target is Target.FRAMES:
                errors.append("group_by 'class_name' is not valid for target 'frames'")
        return errors


class ResultKind(StrEnum):
    """How a result is presented: drives the answer template and the frontend view."""

    SCALAR = "scalar"
    TABLE = "table"
    TIME_SERIES = "time_series"
    FRAMES = "frames"
    ABSTENTION = "abstention"


def result_kind(plan: QueryPlan) -> ResultKind:
    match plan.intent:
        case Intent.ABSTAIN:
            return ResultKind.ABSTENTION
        case Intent.TREND:
            return ResultKind.TIME_SERIES
        case Intent.FRAMES:
            return ResultKind.FRAMES
        case Intent.COUNT | Intent.AGGREGATE | Intent.DWELL if not plan.group_by:
            return ResultKind.SCALAR
        case _:
            return ResultKind.TABLE


def plan_json_schema() -> dict[str, Any]:
    """JSON schema for constrained decoding (vLLM structured outputs, provider JSON mode)."""
    return QueryPlan.model_json_schema()
