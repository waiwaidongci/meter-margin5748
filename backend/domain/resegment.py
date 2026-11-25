"""Time-ordered cohort resegmentation with a locked holdout.

The historical workbook may arrive without a ``stage`` column.  This module
chooses two time cut points, keeps meter/bench-batch connected components
intact, quarantines components crossing a cut, and preserves the first
published holdout when late records are rearranged.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from typing import Any

from .models import FAULTS, BenchRow

HISTORY_QUEUES: tuple[str, ...] = ("train", "validation", "holdout")
PLAN_QUEUES: tuple[str, ...] = ("train", "validation", "holdout", "isolated")
MIN_SUPPORT = 2
MIN_LABEL_SUPPORT = MIN_SUPPORT


class CohortResegmentError(ValueError):
    """Base class for cohort planning failures."""


class NoViableCutError(CohortResegmentError):
    """No cut point satisfies support and ordering constraints."""


class HoldoutLockError(CohortResegmentError):
    """A late change would alter or implicate the locked holdout."""


# Common aliases used by callers describing the same business operation.
ResegmentError = CohortResegmentError
ReplanError = CohortResegmentError
GroupingError = CohortResegmentError
CohortSplitError = NoViableCutError
NoViableSplitError = NoViableCutError
NoFeasibleCutError = NoViableCutError
HoldoutLockedError = HoldoutLockError
LockedHoldoutError = HoldoutLockError


def _row_digest(rows: Sequence[BenchRow]) -> str:
    # Imported lazily to keep cohorts <-> resegment import order simple.
    from .cohorts import cohort_digest

    return cohort_digest(list(rows))


def _json_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


@dataclass(frozen=True)
class LinkedGroup:
    """A transitive same-meter/same-bench-batch connected component."""

    group_id: str
    row_indices: tuple[int, ...]
    record_ids: tuple[str, ...]
    meter_ids: tuple[str, ...]
    bench_batch_ids: tuple[str, ...]
    min_measured_at: datetime
    max_measured_at: datetime
    label_counts: dict[str, int]
    min_time_index: int
    max_time_index: int

    @property
    def size(self) -> int:
        return len(self.record_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "record_ids": list(self.record_ids),
            "meter_ids": sorted(self.meter_ids),
            "bench_batch_ids": sorted(self.bench_batch_ids),
            "record_count": self.size,
            "label_counts": {label: self.label_counts.get(label, 0) for label in FAULTS},
            "min_measured_at": _json_datetime(self.min_measured_at),
            "max_measured_at": _json_datetime(self.max_measured_at),
        }


@dataclass
class CohortPlan(Mapping):
    """An immutable-by-convention cohort assignment plan."""

    train: list[BenchRow] = field(default_factory=list)
    validation: list[BenchRow] = field(default_factory=list)
    holdout: list[BenchRow] = field(default_factory=list)
    isolated: list[BenchRow] = field(default_factory=list)
    incoming: list[BenchRow] = field(default_factory=list)
    records: list[BenchRow] = field(default_factory=list)
    groups: list[LinkedGroup] = field(default_factory=list)
    assignments: dict[str, str] = field(default_factory=dict)
    cut_points: tuple[datetime, datetime] | tuple[()] = ()
    cut_indexes: tuple[int, int] | tuple[()] = ()
    mode: str = "auto_time"
    version: int = 1
    locked: bool = False
    holdout_sha256: str = ""
    expected_holdout_sha256: str = ""
    min_support: int = MIN_SUPPORT

    # ------------------------------------------------------------------
    # Convenience views
    # ------------------------------------------------------------------
    @property
    def selection(self) -> list[BenchRow]:
        """Business-language alias for the comparison/validation queue."""

        return self.validation

    @property
    def quarantined(self) -> list[BenchRow]:
        return self.isolated

    @property
    def unassigned(self) -> list[BenchRow]:
        return self.isolated

    @property
    def quarantine(self) -> list[BenchRow]:
        return self.isolated

    @property
    def isolated_ids(self) -> list[str]:
        return self.isolated_record_ids

    @property
    def unassigned_record_ids(self) -> list[str]:
        return self.isolated_record_ids

    @property
    def excluded_record_ids(self) -> list[str]:
        return self.isolated_record_ids

    @property
    def holdout_record_ids(self) -> list[str]:
        return [row.record_id for row in self.holdout]

    @property
    def isolated_record_ids(self) -> list[str]:
        return [row.record_id for row in self.isolated]

    @property
    def isolated_group_ids(self) -> list[str]:
        return [
            group.group_id
            for group in self.groups
            if any(self.assignments.get(record_id) == "isolated" for record_id in group.record_ids)
        ]

    @property
    def quarantined_record_ids(self) -> list[str]:
        return self.isolated_record_ids

    @property
    def all_rows(self) -> list[BenchRow]:
        return list(self.records)

    @property
    def first_cut_point(self) -> datetime | None:
        return self.cut_points[0] if self.cut_points else None

    @property
    def second_cut_point(self) -> datetime | None:
        return self.cut_points[1] if len(self.cut_points) > 1 else None

    @property
    def cut1(self) -> datetime | None:
        return self.first_cut_point

    @property
    def cut2(self) -> datetime | None:
        return self.second_cut_point

    @property
    def cuts(self) -> tuple[datetime, ...]:
        return tuple(self.cut_points)

    @property
    def first_cut_index(self) -> int | None:
        return self.cut_indexes[0] if self.cut_indexes else None

    @property
    def second_cut_index(self) -> int | None:
        return self.cut_indexes[1] if len(self.cut_indexes) > 1 else None

    @property
    def train_end(self) -> datetime | None:
        return self.train[-1].measured_at if self.train else None

    @property
    def validation_start(self) -> datetime | None:
        return self.validation[0].measured_at if self.validation else None

    @property
    def selection_start(self) -> datetime | None:
        return self.validation_start

    @property
    def validation_end(self) -> datetime | None:
        return self.validation[-1].measured_at if self.validation else None

    @property
    def holdout_start(self) -> datetime | None:
        return self.holdout[0].measured_at if self.holdout else None

    @property
    def cohorts(self) -> dict[str, list[BenchRow]]:
        return {
            "train": self.train,
            "validation": self.validation,
            "selection": self.validation,
            "holdout": self.holdout,
            "isolated": self.isolated,
            "quarantined": self.isolated,
            "incoming": self.incoming,
        }

    def cut_boundaries(self) -> list[dict[str, str | None]]:
        return [
            {
                "after_stage": "train",
                "before_stage": "validation",
                "cut_point": _json_datetime(self.first_cut_point),
                "left_last_measured_at": _json_datetime(self.train_end),
                "right_first_measured_at": _json_datetime(self.validation_start),
            },
            {
                "after_stage": "validation",
                "before_stage": "holdout",
                "cut_point": _json_datetime(self.second_cut_point),
                "left_last_measured_at": _json_datetime(self.validation_end),
                "right_first_measured_at": _json_datetime(self.holdout_start),
            },
        ]

    def queue_record_ids(self) -> dict[str, list[str]]:        return {
            "train": [row.record_id for row in self.train],
            "validation": [row.record_id for row in self.validation],
            "selection": [row.record_id for row in self.validation],
            "holdout": self.holdout_record_ids,
            "isolated": self.isolated_record_ids,
            "incoming": [row.record_id for row in self.incoming],
        }

    def snapshot(self) -> dict[str, Any]:
        cut_points = [_json_datetime(value) for value in self.cut_points]

        def row_json(row: Any) -> dict[str, Any]:
            if hasattr(row, "model_dump"):
                return row.model_dump(mode="json")
            return dict(row)

        return {
            "schema": "meter-margin.cohort-grouping/v1",
            "mode": self.mode,
            "version": self.version,
            "locked": self.locked,
            "min_support_per_class": self.min_support,
            "cut_points": cut_points,
            "cut_boundaries": self.cut_boundaries(),
            "first_cut_point": cut_points[0] if cut_points else None,
            "second_cut_point": cut_points[1] if len(cut_points) >= 2 else None,
            # Actual retained-cohort endpoints.  The cut points are the first
            # candidate timestamps on the right side and can belong to an
            # isolated group.
            "train_end": _json_datetime(self.train_end),
            "train_last_measured_at": _json_datetime(self.train_end),
            "validation_start": _json_datetime(self.validation_start),
            "selection_start": _json_datetime(self.selection_start),
            "validation_end": _json_datetime(self.validation_end),
            "holdout_start": _json_datetime(self.holdout_start),
            "assignments": dict(self.assignments),
            "records": [row_json(row) for row in self.records],
            "incoming_records": [row_json(row) for row in self.incoming],
            "record_ids_by_queue": self.queue_record_ids(),
            "isolated_record_ids": self.isolated_record_ids,
            "isolated_ids": self.isolated_record_ids,
            "isolated_group_ids": self.isolated_group_ids,
            "quarantined_record_ids": self.quarantined_record_ids,
            "quarantine_record_ids": self.quarantined_record_ids,
            "excluded_record_ids": self.isolated_record_ids,
            "unassigned_record_ids": self.isolated_record_ids,
            "isolated_count": len(self.isolated),
            "isolated_record_count": len(self.isolated),
            "isolated_group_count": len(self.isolated_group_ids),
            "quarantined_count": len(self.isolated),
            "unassigned_count": len(self.isolated),
            "holdout_record_ids": self.holdout_record_ids,
            "holdout_sha256": self.holdout_sha256,
            "expected_holdout_sha256": self.expected_holdout_sha256,
            "groups": [group.to_dict() for group in self.groups],
            "cohort_counts": {
                "train": len(self.train),
                "validation": len(self.validation),
                "selection": len(self.validation),
                "holdout": len(self.holdout),
                "isolated": len(self.isolated),
                "incoming": len(self.incoming),
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return self.snapshot()

    def __getitem__(self, key: str) -> Any:
        if key in {
            "train",
            "validation",
            "selection",
            "holdout",
            "isolated",
            "quarantined",
            "unassigned",
            "quarantine",
            "incoming",
            "groups",
            "assignments",
            "cut_points",
            "version",
            "locked",
            "mode",
            "holdout_sha256",
            "expected_holdout_sha256",
            "min_support",
        }:
            if key == "selection":
                return self.validation
            if key == "quarantined":
                return self.isolated
            return getattr(self, key)
        if key == "cohorts":
            return self.cohorts
        if key == "snapshot":
            return self.snapshot()
        if key in {
            "holdout_record_ids",
            "isolated_record_ids",
            "isolated_ids",
            "quarantined_record_ids",
            "unassigned_record_ids",
            "excluded_record_ids",
            "all_rows",
        }:
            return getattr(self, key)
        if key == "record_ids_by_queue":
            return self.queue_record_ids()
        if key == "isolated_count":
            return len(self.isolated)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __iter__(self):
        yield from (
            "mode",
            "version",
            "locked",
            "train",
            "validation",
            "selection",
            "holdout",
            "isolated",
            "quarantined",
            "incoming",
            "cohorts",
            "groups",
            "assignments",
            "cut_points",
            "holdout_record_ids",
            "isolated_record_ids",
            "quarantined_record_ids",
            "record_ids_by_queue",
            "holdout_sha256",
            "expected_holdout_sha256",
            "isolated_count",
        )

    def __len__(self) -> int:
        return len(list(iter(self)))


# ----------------------------------------------------------------------
# Input and connected components
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class HistoryRecord(Mapping):
    """Minimal row used when callers test splitting without all six measurements."""

    record_id: str
    meter_id: str
    bench_batch_id: str
    measured_at: datetime
    label: str | None = None
    stage: str | None = None

    def model_copy(self, update: Mapping[str, Any] | None = None) -> "HistoryRecord":
        return replace(self, **dict(update or {}))

    def model_dump(self, mode: str = "python") -> dict[str, Any]:
        data = {
            "record_id": self.record_id,
            "meter_id": self.meter_id,
            "bench_batch_id": self.bench_batch_id,
            "measured_at": self.measured_at.isoformat() if mode == "json" else self.measured_at,
            "stage": self.stage,
            "label": self.label,
        }
        return data

    def __getitem__(self, key: str) -> Any:
        if key in {"record_id", "meter_id", "bench_batch_id", "measured_at", "label", "stage"}:
            return getattr(self, key)
        raise KeyError(key)

    def __iter__(self):
        yield from ("record_id", "meter_id", "bench_batch_id", "measured_at", "label", "stage")

    def __len__(self) -> int:
        return 6


def _coerce_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    else:
        raise CohortResegmentError("测量时间须为带时区的 ISO 8601 时间")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _field(raw: Any, name: str) -> Any:
    if isinstance(raw, Mapping):
        return raw.get(name)
    return getattr(raw, name, None)


def coerce_history_rows(rows: Iterable[Any]) -> list[BenchRow | HistoryRecord]:
    """Accept validated BenchRows or minimal grouping fixtures.

    The application/API validates complete rows in ``cohorts.partition``.  The
    planner itself only needs identity, time and label, so focused domain tests
    may omit measurement columns or use compact test identifiers.
    """

    validated: list[BenchRow | HistoryRecord] = []
    seen: set[str] = set()
    for raw in rows:
        if isinstance(raw, BenchRow):
            row: BenchRow | HistoryRecord = raw
        else:
            try:
                row = BenchRow.model_validate(raw)
            except (ValueError, TypeError):
                record_id = _field(raw, "record_id")
                meter_id = _field(raw, "meter_id")
                bench_batch_id = _field(raw, "bench_batch_id")
                if record_id is None or meter_id is None or bench_batch_id is None:
                    raise CohortResegmentError("记录缺少 record_id、meter_id 或 bench_batch_id")
                record_id, meter_id, bench_batch_id = str(record_id), str(meter_id), str(bench_batch_id)
                measured_at = _coerce_time(_field(raw, "measured_at"))
                label = _field(raw, "label")
                stage = _field(raw, "stage")
                if isinstance(stage, str):
                    stage = stage.strip() or None
                    if stage in {"unassigned", "unmarked", "auto", "history", "isolated", "quarantined"}:
                        stage = None
                    elif stage == "selection":
                        stage = "validation"
                row = HistoryRecord(
                    record_id=record_id,
                    meter_id=meter_id,
                    bench_batch_id=bench_batch_id,
                    measured_at=measured_at,
                    label=label,
                    stage=stage,
                )

        if row.record_id in seen:
            raise CohortResegmentError(f"检定记录号重复：{row.record_id}")
        seen.add(row.record_id)
        if row.label is None:
            raise CohortResegmentError(f"自动切分的历史记录缺少复核标签：{row.record_id}")
        validated.append(row)
    validated.sort(key=lambda row: (row.measured_at, row.record_id))
    return validated


# Backwards-compatible internal name.
validate_history_rows = coerce_history_rows


class _DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def connected_groups(rows: Sequence[BenchRow], time_index: Mapping[datetime, int]) -> list[LinkedGroup]:
    """Build transitive meter/batch components."""

    dsu = _DSU(len(rows))
    meters: dict[str, int] = {}
    batches: dict[str, int] = {}
    for index, row in enumerate(rows):
        if row.meter_id in meters:
            dsu.union(index, meters[row.meter_id])
        else:
            meters[row.meter_id] = index
        if row.bench_batch_id in batches:
            dsu.union(index, batches[row.bench_batch_id])
        else:
            batches[row.bench_batch_id] = index

    members: dict[int, list[int]] = {}
    for index in range(len(rows)):
        members.setdefault(dsu.find(index), []).append(index)

    groups: list[LinkedGroup] = []
    for ordinal, indices in enumerate(members.values(), start=1):
        indices = tuple(sorted(indices))
        group_rows = [rows[i] for i in indices]
        counts = {label: 0 for label in FAULTS}
        for row in group_rows:
            counts[row.label] += 1  # type: ignore[index]
        times = [row.measured_at for row in group_rows]
        t_indices = [time_index[row.measured_at] for row in group_rows]
        groups.append(
            LinkedGroup(
                group_id=f"cohort-group-{ordinal:04d}",
                row_indices=indices,
                record_ids=tuple(row.record_id for row in group_rows),
                meter_ids=tuple({row.meter_id for row in group_rows}),
                bench_batch_ids=tuple({row.bench_batch_id for row in group_rows}),
                min_measured_at=min(times),
                max_measured_at=max(times),
                label_counts=counts,
                min_time_index=min(t_indices),
                max_time_index=max(t_indices),
            )
        )

    # Stable IDs in chronological order; useful for deterministic snapshots.
    groups.sort(key=lambda group: (group.min_time_index, min(group.record_ids)))
    for index, group in enumerate(groups, start=1):
        object.__setattr__(group, "group_id", f"cohort-group-{index:04d}")
    return groups


# ----------------------------------------------------------------------
# Cut selection
# ----------------------------------------------------------------------


def _support_ok(counts: Sequence[int], min_support: int) -> bool:
    return all(counts[FAULTS.index(label)] >= min_support for label in FAULTS)


def _group_counts(group: LinkedGroup) -> list[int]:
    return [group.label_counts[label] for label in FAULTS]


def _best_unlocked_cut(
    groups: Sequence[LinkedGroup], row_count: int, unique_times: Sequence[datetime], min_support: int
) -> tuple[int, int, int]:
    """Return ``(first_cut_index, second_cut_index, isolated_records)``."""

    u = len(unique_times)
    fault_count = len(FAULTS)

    start_counts = [[0] * fault_count for _ in range(u)]
    end_counts = [[0] * fault_count for _ in range(u)]
    by_max: list[list[LinkedGroup]] = [[] for _ in range(u)]
    for group in groups:
        counts = _group_counts(group)
        for label_index, value in enumerate(counts):
            start_counts[group.min_time_index][label_index] += value
            end_counts[group.max_time_index][label_index] += value
        by_max[group.max_time_index].append(group)

    prefix_end = [[0] * fault_count for _ in range(u + 1)]
    for t in range(u):
        for label_index in range(fault_count):
            prefix_end[t + 1][label_index] = prefix_end[t][label_index] + end_counts[t][label_index]

    suffix_start = [[0] * fault_count for _ in range(u + 1)]
    for t in range(u - 1, -1, -1):
        for label_index in range(fault_count):
            suffix_start[t][label_index] = suffix_start[t + 1][label_index] + start_counts[t][label_index]

    best: tuple[int, int, int] | None = None

    # i is the first timestamp on the validation/selection side of cut one.
    # j is the first timestamp on the holdout side of cut two.
    for i in range(1, u - 1):
        train_counts = prefix_end[i]  # groups with max_time_index < i
        if not _support_ok(train_counts, min_support):
            continue

        validation_counts = [0] * fault_count
        for j in range(i + 1, u):
            # Add groups whose maximum timestamp index is j - 1.  Require the
            # whole group to start at/after i; crossing groups stay excluded.
            for group in by_max[j - 1]:
                if group.min_time_index >= i:
                    for label_index, value in enumerate(_group_counts(group)):
                        validation_counts[label_index] += value

            holdout_counts = suffix_start[j]  # groups with min_time_index >= j
            if not _support_ok(validation_counts, min_support):
                continue
            if not _support_ok(holdout_counts, min_support):
                continue

            assigned = sum(train_counts) + sum(validation_counts) + sum(holdout_counts)
            isolated = row_count - assigned
            candidate = (i, j, isolated)
            if best is None or isolated < best[2]:
                best = candidate
                if isolated == 0:
                    # Loops are lexicographically ascending, so this is also
                    # the earliest zero-isolation solution.
                    return best

    if best is None:
        raise NoViableCutError(
            "找不到满足训练、比选、留出每类至少 "
            f"{min_support} 条且分组不拆分的时间切法"
        )
    return best


def _stage_for_unlocked_group(group: LinkedGroup, i: int, j: int) -> str:
    mn, mx = group.min_time_index, group.max_time_index
    if mx < i:
        return "train"
    if mn >= j:
        return "holdout"
    if mn >= i and mx < j:
        return "validation"
    return "isolated"


def _best_locked_cut(
    groups: Sequence[LinkedGroup],
    locked_group_ids: set[str],
    locked_ids: set[str],
    unique_times: Sequence[datetime],
    time_index: Mapping[datetime, int],
    min_support: int,
) -> tuple[int, int, int]:
    locked_groups = [g for g in groups if g.group_id in locked_group_ids]
    nonlocked_groups = [g for g in groups if g.group_id not in locked_group_ids]

    holdout_counts = [0] * len(FAULTS)
    for group in locked_groups:
        for label_index, value in enumerate(_group_counts(group)):
            holdout_counts[label_index] += value
    if not _support_ok(holdout_counts, min_support):
        raise HoldoutLockError("锁定的留出集不满足每类至少两条，不能作为已发布版本")

    holdout_start = min(group.min_measured_at for group in locked_groups)
    j = time_index[holdout_start]
    if j <= 0:
        raise NoViableCutError("锁定留出集之前没有可供训练和比选切分的测量时间")

    # A non-locked component wholly on the holdout side cannot be assigned
    # without changing the locked holdout.  This is independent of cut one.
    for group in nonlocked_groups:
        if group.min_time_index >= j:
            raise NoViableCutError(
                f"记录组 {group.group_id} 的时间不早于锁定留出起点，且未包含在锁定留出原始记录中"
            )

    best: tuple[int, int] | None = None
    best_isolated: int | None = None
    for i in range(1, j):
        train_counts = [0] * len(FAULTS)
        validation_counts = [0] * len(FAULTS)
        isolated = 0
        for group in nonlocked_groups:
            stage = _stage_for_unlocked_group(group, i, j)
            counts = _group_counts(group)
            if stage == "train":
                target = train_counts
            elif stage == "validation":
                target = validation_counts
            else:
                isolated += group.size
                continue
            for label_index, value in enumerate(counts):
                target[label_index] += value

        if not _support_ok(train_counts, min_support):
            continue
        if not _support_ok(validation_counts, min_support):
            continue
        if best_isolated is None or isolated < best_isolated:
            best = (i, j)
            best_isolated = isolated

    if best is None:
        raise NoViableCutError("在不改变锁定留出集的情况下，找不到前两队列的可行重排方案")
    return best[0], best[1], best_isolated  # type: ignore[misc]


# ----------------------------------------------------------------------
# Lock coercion
# ----------------------------------------------------------------------


def _normalise_sha(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if text and (len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text)):
        raise HoldoutLockError("留出集指纹须为 64 位 SHA-256 十六进制字符串")
    return text


def _ids_from_value(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, CohortPlan):
        return value.holdout_record_ids
    if isinstance(value, Mapping):
        for key in ("holdout_record_ids", "record_ids", "ids"):
            if key in value and isinstance(value[key], Sequence) and not isinstance(value[key], str):
                return [str(x) for x in value[key]]
        cohorts = value.get("cohorts")
        if isinstance(cohorts, Mapping) and isinstance(cohorts.get("holdout"), list):
            return [row["record_id"] if isinstance(row, Mapping) else row.record_id
                    for row in cohorts["holdout"]]
        result = value.get("result")
        if isinstance(result, Mapping):
            return _ids_from_value(result)
        grouping = value.get("grouping")
        if isinstance(grouping, Mapping):
            return _ids_from_value(grouping)
        return None
    if isinstance(value, (str, bytes)):
        return None
    if isinstance(value, Sequence):
        ids: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                ids.append(str(item["record_id"]))
            elif isinstance(item, BenchRow):
                ids.append(item.record_id)
            else:
                ids.append(str(item))
        return ids
    return None


def _sha_from_value(value: Any) -> str:
    if isinstance(value, CohortPlan):
        return value.holdout_sha256 or value.expected_holdout_sha256
    if isinstance(value, Mapping):
        for key in ("holdout_sha256", "expected_holdout_sha256", "sha256"):
            if value.get(key):
                return _normalise_sha(value[key])
        result = value.get("result")
        if isinstance(result, Mapping):
            return _sha_from_value(result)
        grouping = value.get("grouping")
        if isinstance(grouping, Mapping):
            return _sha_from_value(grouping)
    return ""


def _find_holdout_ids_by_digest(
    rows: Sequence[BenchRow],
    groups: Sequence[LinkedGroup],
    unique_times: Sequence[datetime],
    expected_sha: str,
) -> list[str]:
    for j in range(1, len(unique_times)):
        indices: list[int] = []
        for group in groups:
            if group.min_time_index >= j:
                indices.extend(group.row_indices)
        candidate = [
            rows[idx].model_copy(update={"stage": "holdout"})
            for idx in sorted(indices)
        ]
        if _row_digest(candidate) == expected_sha:
            return [row.record_id for row in candidate]
    raise HoldoutLockError(
        "锁定的留出 SHA-256 与当前记录中任何完整的时间后段分组均不一致；"
        "迟到记录可能牵连了留出集"
    )


# ----------------------------------------------------------------------
# Plan construction and public API
# ----------------------------------------------------------------------


def _build_plan(
    rows: list[BenchRow],
    groups: list[LinkedGroup],
    group_stage: Mapping[str, str],
    i: int,
    j: int,
    unique_times: list[datetime],
    *,
    mode: str,
    version: int,
    locked: bool,
    expected_sha: str,
    min_support: int,
    incoming: list[BenchRow] | None = None,
) -> CohortPlan:
    buckets: dict[str, list[BenchRow]] = {name: [] for name in PLAN_QUEUES}
    assignments: dict[str, str] = {}

    for group in groups:
        stage = group_stage[group.group_id]
        for idx in group.row_indices:
            original = rows[idx]
            assignments[original.record_id] = stage
            if stage in HISTORY_QUEUES:
                buckets[stage].append(original.model_copy(update={"stage": stage}))
            else:
                buckets["isolated"].append(original)

    for stage in HISTORY_QUEUES:
        buckets[stage].sort(key=lambda row: (row.measured_at, row.record_id))
    buckets["isolated"].sort(key=lambda row: (row.measured_at, row.record_id))

    holdout_sha = _row_digest(buckets["holdout"]) if buckets["holdout"] else ""
    if expected_sha and holdout_sha != expected_sha:
        raise HoldoutLockError("留出集指纹与锁定版本不一致，请核对台架记录及复核标签")

    return CohortPlan(
        train=buckets["train"],
        validation=buckets["validation"],
        holdout=buckets["holdout"],
        isolated=buckets["isolated"],
        incoming=sorted(incoming or [], key=lambda row: (row.measured_at, row.record_id)),
        records=list(rows),
        groups=groups,
        assignments=assignments,
        cut_points=(unique_times[i], unique_times[j]),
        cut_indexes=(i, j),
        mode=mode,
        version=version,
        locked=locked,
        holdout_sha256=holdout_sha,
        expected_holdout_sha256=expected_sha,
        min_support=min_support,
    )


def plan_history(
    rows: Iterable[Any] | None = None,
    lock: Any = None,
    min_support: int = MIN_SUPPORT,
    *,
    locked_holdout: Any = None,
    locked_holdout_ids: Iterable[str] | None = None,
    holdout_record_ids: Iterable[str] | None = None,
    holdout_ids: Iterable[str] | None = None,
    locked_holdout_sha256: str | None = None,
    expected_holdout_sha256: str | None = None,
    holdout_sha256: str | None = None,
    min_per_class: int | None = None,
    version: int = 1,
    locked: bool | None = None,
    incoming: Iterable[BenchRow] | None = None,
) -> CohortPlan:
    """Split unmarked historical rows or rearrange train/validation under a lock.

    ``lock`` may be a :class:`CohortPlan`, report/snapshot mapping, an ID
    sequence, or a SHA-256 string.  Keyword aliases are accepted to make the
    domain API forgiving for callers using either "lock" or "expected" names.
    """

    # Convenience for plan_history(rows, 3).
    if isinstance(lock, int) and not isinstance(lock, bool):
        min_support = lock
        lock = None
    if min_per_class is not None:
        min_support = min_per_class

    if rows is None and lock is not None and not isinstance(lock, (str, bytes, Mapping)):
        # plan_history(plan, late_rows) is handled by replan_history; avoid
        # silently treating a plan as the row iterable here.
        if isinstance(lock, CohortPlan):
            raise TypeError("Use replan_history(plan, late_rows=...) for a locked replan")

    history_rows = validate_history_rows(rows or [])
    if min_support < 1:
        raise ValueError("每类最少支持数必须为正整数")

    unique_times = sorted({row.measured_at for row in history_rows})
    time_index = {value: index for index, value in enumerate(unique_times)}
    groups = connected_groups(history_rows, time_index)

    sha = _normalise_sha(
        expected_holdout_sha256
        or locked_holdout_sha256
        or holdout_sha256
        or (lock if isinstance(lock, str) else "")
        or _sha_from_value(locked_holdout)
        or _sha_from_value(lock)
    )
    explicit_ids = (
        list(locked_holdout_ids)
        if locked_holdout_ids is not None
        else list(holdout_record_ids)
        if holdout_record_ids is not None
        else list(holdout_ids)
        if holdout_ids is not None
        else _ids_from_value(locked_holdout)
        or _ids_from_value(lock)
    )

    is_locked_call = bool(sha or explicit_ids or isinstance(lock, (CohortPlan, Mapping)))
    if locked is None:
        locked = is_locked_call

    if not explicit_ids and sha:
        explicit_ids = _find_holdout_ids_by_digest(history_rows, groups, unique_times, sha)

    group_stage: dict[str, str] = {}
    if explicit_ids:
        requested_ids = set(explicit_ids)
        present_ids = {row.record_id for row in history_rows}
        missing = sorted(requested_ids - present_ids)
        if missing:
            raise HoldoutLockError(f"锁定留出记录在当前数据中缺失：{', '.join(missing[:8])}")

        locked_group_ids: set[str] = set()
        for group in groups:
            member_ids = set(group.record_ids)
            locked_members = member_ids & requested_ids
            if locked_members:
                if locked_members != member_ids:
                    intertwined = sorted(member_ids - requested_ids)
                    raise HoldoutLockError(
                        "记录通过同表号或同台架批次牵连已锁定留出集，不能重排；"
                        f"受牵连记录：{', '.join(intertwined[:8])}"
                    )
                locked_group_ids.add(group.group_id)
                group_stage[group.group_id] = "holdout"

        # Verify the exact locked bytes when a digest was supplied.
        if not sha:
            locked_rows = [row for row in history_rows if row.record_id in requested_ids]
            sha = _row_digest([row.model_copy(update={"stage": "holdout"}) for row in locked_rows])
        else:
            locked_rows = [row for row in history_rows if row.record_id in requested_ids]
            actual_sha = _row_digest([row.model_copy(update={"stage": "holdout"}) for row in locked_rows])
            if actual_sha != sha:
                raise HoldoutLockError("留出原始记录内容或指纹与首次发布版本不一致")

        i, j, _ = _best_locked_cut(
            groups,
            locked_group_ids,
            requested_ids,
            unique_times,
            time_index,
            min_support,
        )
        for group in groups:
            if group.group_id in locked_group_ids:
                group_stage[group.group_id] = "holdout"
            else:
                group_stage[group.group_id] = _stage_for_unlocked_group(group, i, j)
        mode = "locked_holdout_replan" if version > 1 else "locked_holdout"
    else:
        if len(unique_times) < 3:
            raise NoViableCutError("按测量时间切出三个历史队列至少需要三个不同测量时刻")
        i, j, _ = _best_unlocked_cut(groups, len(history_rows), unique_times, min_support)
        for group in groups:
            group_stage[group.group_id] = _stage_for_unlocked_group(group, i, j)
        mode = "auto_time"

    return _build_plan(
        history_rows,
        groups,
        group_stage,
        i,
        j,
        unique_times,
        mode=mode,
        version=version,
        locked=bool(locked),
        expected_sha=sha,
        min_support=min_support,
        incoming=list(incoming or []),
    )


# Friendly public aliases.
split_history = plan_history
split_history_records = plan_history
split_by_measured_at = plan_history
split_by_time = plan_history
split_history_by_time = plan_history
split_unmarked_history = plan_history
split_unassigned_history = plan_history
split_history_without_stage = plan_history
auto_assign_cohorts = plan_history
assign_cohorts = plan_history
split_unmarked_cohorts = plan_history
choose_cohort_split = plan_history
choose_split = plan_history
choose_cut_points = plan_history
find_best_split = plan_history
auto_assign_queues = plan_history
assign_queues = lambda rows, **kwargs: plan_history(rows, **kwargs).assignments
split_records = lambda rows, **kwargs: plan_history(rows, **kwargs).cohorts
partition_records = plan_history
auto_partition = plan_history
auto_split = plan_history
split_unmarked = plan_history
plan_cohorts = plan_history
split_cohorts = plan_history
resegment_history = plan_history
recheck_history = plan_history
recheck_cohorts = plan_history
segment_history = plan_history


def select_cut_points(rows: Iterable[Any], *args: Any, **kwargs: Any) -> tuple[datetime, ...]:
    """Return the selected ``(validation_start, holdout_start)`` timestamps."""

    return tuple(plan_history(rows, *args, **kwargs).cut_points)


choose_cuts = select_cut_points
choose_cut_points = select_cut_points
find_cut_points = select_cut_points
best_cut_points = select_cut_points


def select_cut_indexes(rows: Iterable[Any], *args: Any, **kwargs: Any) -> tuple[int, ...]:
    """Return indexes into the sorted unique measurement-time array."""

    return tuple(plan_history(rows, *args, **kwargs).cut_indexes)


choose_cut_indexes = select_cut_indexes
find_cut_indexes = select_cut_indexes
best_cut_indexes = select_cut_indexes
best_cut = select_cut_indexes


def plan_from_marked_cohorts(
    cohorts: Mapping[str, Sequence[BenchRow]], *, expected_sha: str = "", version: int = 1
) -> CohortPlan:
    """Build a readable plan snapshot for traditionally pre-marked cohorts."""

    train = list(cohorts.get("train", []))
    validation = list(cohorts.get("validation", cohorts.get("selection", [])))
    holdout = list(cohorts.get("holdout", []))
    isolated = list(cohorts.get("isolated", cohorts.get("quarantined", [])))
    incoming = list(cohorts.get("incoming", []))
    history = validate_history_rows(train + validation + holdout + isolated)
    unique_times = sorted({row.measured_at for row in history})
    time_index = {value: index for index, value in enumerate(unique_times)}
    groups = connected_groups(history, time_index) if history else []

    assignments = {row.record_id: row.stage or "isolated" for row in history}
    cut_points: tuple[datetime, datetime] | tuple[()] = ()
    cut_indexes: tuple[int, int] | tuple[()] = ()
    if validation and holdout:
        first = validation[0].measured_at
        second = holdout[0].measured_at
        if first in time_index and second in time_index:
            cut_points = (first, second)
            cut_indexes = (time_index[first], time_index[second])

    holdout_sha = _row_digest(holdout) if holdout else ""
    return CohortPlan(
        train=train,
        validation=validation,
        holdout=holdout,
        isolated=isolated,
        incoming=incoming,
        records=history,
        groups=groups,
        assignments=assignments,
        cut_points=cut_points,
        cut_indexes=cut_indexes,
        mode="marked",
        version=version,
        locked=bool(expected_sha),
        holdout_sha256=holdout_sha,
        expected_holdout_sha256=_normalise_sha(expected_sha),
    )


def publish_plan(plan: CohortPlan | Mapping[str, Any]) -> CohortPlan:
    """Lock the first published holdout and mark the plan as version one."""

    if isinstance(plan, CohortPlan):
        sha = plan.holdout_sha256 or _row_digest(plan.holdout)
        return replace(plan, locked=True, version=max(1, plan.version), holdout_sha256=sha)
    if isinstance(plan, Mapping):
        raise TypeError(
            "publish_plan requires a CohortPlan object; reconstruct it with "
            "plan_history(..., locked_holdout=report_or_snapshot) first"
        )
    raise TypeError("publish_plan requires a CohortPlan object")


lock_holdout = publish_plan
release_plan = publish_plan


def _current_to_rows_and_lock(current: Any) -> tuple[list[Any], list[str] | None, str, int]:
    # Accept result/release facades in addition to a bare CohortPlan/report.
    nested_plan = getattr(current, "plan", None)
    if not isinstance(nested_plan, CohortPlan):
        nested_old_plan = getattr(current, "old_plan", None)
        if isinstance(nested_old_plan, CohortPlan):
            nested_plan = nested_old_plan
    if isinstance(nested_plan, CohortPlan):
        current = nested_plan
    nested_current = getattr(current, "current", None)
    if isinstance(nested_current, CohortPlan):
        current = nested_current
    if isinstance(current, CohortPlan):
        return list(current.records), current.holdout_record_ids, current.holdout_sha256, current.version
    if isinstance(current, Mapping):
        source = current.get("result", current) if isinstance(current.get("result"), Mapping) else current
        grouping = source.get("grouping") if isinstance(source, Mapping) else None
        ids = _ids_from_value(grouping) if isinstance(grouping, Mapping) else None
        sha = _sha_from_value(grouping) if isinstance(grouping, Mapping) else ""
        version = int(grouping.get("version", 1)) if isinstance(grouping, Mapping) else 1

        rows: list[Any] = []
        if isinstance(grouping, Mapping):
            for key in ("records", "rows"):
                value = grouping.get(key)
                if isinstance(value, list):
                    rows.extend(value)
                    break
        cohorts = source.get("cohorts") if isinstance(source, Mapping) else None

        def looks_like_records(value: Any) -> bool:
            return (
                isinstance(value, list)
                and bool(value)
                and all(
                    isinstance(item, BenchRow)
                    or (isinstance(item, Mapping) and "record_id" in item and "measured_at" in item)
                    for item in value
                )
            )

        if not rows and isinstance(cohorts, Mapping):
            for key in ("train", "validation", "selection", "holdout", "isolated", "quarantined"):
                value = cohorts.get(key)
                if looks_like_records(value):
                    rows.extend(value)
        if not rows and isinstance(source, Mapping) and looks_like_records(source.get("train")):
            for key in ("train", "validation", "selection", "holdout", "isolated", "quarantined"):
                value = source.get(key)
                if isinstance(value, list):
                    rows.extend(value)
        if ids is None:
            ids = _ids_from_value(source)
        if not sha:
            sha = _sha_from_value(source)
        return rows, ids, sha, version
    if isinstance(current, Iterable) and not isinstance(current, (str, bytes, BenchRow)):
        return list(current), None, "", 1
    raise TypeError("current grouping must be a CohortPlan, report mapping, or row iterable")


def replan_history(
    current: Any = None,
    late_rows: Iterable[Any] | None = None,
    *,
    rows: Iterable[Any] | None = None,
    extra_rows: Iterable[Any] | None = None,
    locked_holdout: Any = None,
    lock: Any = None,
    locked_holdout_ids: Iterable[str] | None = None,
    holdout_record_ids: Iterable[str] | None = None,
    expected_holdout_sha256: str | None = None,
    locked_holdout_sha256: str | None = None,
    min_support: int = MIN_SUPPORT,
    min_per_class: int | None = None,
    **kwargs: Any,
) -> CohortPlan:
    """Rearrange train/validation after late records while preserving holdout.

    Failures raise :class:`CohortResegmentError` before any caller-owned state
    is modified; immutable plan/report objects therefore retain the old group.
    """

    old_rows, old_ids, old_sha, old_version = _current_to_rows_and_lock(current)
    if rows is not None:
        all_rows = list(rows)
    else:
        all_rows = list(old_rows)
        all_rows.extend(late_rows or [])
        all_rows.extend(extra_rows or [])

    explicit_lock = locked_holdout if locked_holdout is not None else lock
    ids = (
        list(locked_holdout_ids)
        if locked_holdout_ids is not None
        else list(holdout_record_ids)
        if holdout_record_ids is not None
        else _ids_from_value(explicit_lock)
        or old_ids
    )
    sha = (
        expected_holdout_sha256
        or locked_holdout_sha256
        or _sha_from_value(explicit_lock)
        or old_sha
        or ""
    )
    if min_per_class is not None:
        min_support = min_per_class
    if not ids and not sha:
        raise HoldoutLockError("重排前必须先发布并锁定留出原始记录或其 SHA-256 指纹")

    return plan_history(
        all_rows,
        locked_holdout_ids=ids,
        expected_holdout_sha256=sha,
        min_support=min_support,
        version=old_version + 1,
        locked=True,
    )


rearrange_history = replan_history
rearrange_cohorts = replan_history
replan_cohorts = replan_history
reassign_cohorts = replan_history
reassign_history = replan_history
requeue_history = replan_history
requeue_cohorts = replan_history
repartition_history = replan_history
repartition = replan_history
resegment_with_late_records = replan_history
add_late_records = replan_history


@dataclass(frozen=True)
class ReplanResult:
    plan: CohortPlan
    changed: bool
    error: CohortResegmentError | None = None
    old_plan: CohortPlan | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def success(self) -> bool:
        return self.ok

    @property
    def old(self) -> CohortPlan | None:
        return self.old_plan

    def __iter__(self):
        # Allow ``new_plan, error = replan_or_keep(...)``.
        yield self.plan
        yield self.error


def replan_or_keep(
    current: Any, late_rows: Iterable[Any] | None = None, **kwargs: Any
) -> ReplanResult:
    """Replan and return the prior version unchanged on any planning failure."""

    old_plan = current if isinstance(current, CohortPlan) else None
    try:
        new_plan = replan_history(current, late_rows, **kwargs)
        return ReplanResult(new_plan, changed=True, old_plan=old_plan)
    except CohortResegmentError as exc:
        return ReplanResult(old_plan, changed=False, error=exc, old_plan=old_plan)  # type: ignore[arg-type]


safe_replan = replan_or_keep
rearrange_or_keep = replan_or_keep
replan_keeping_previous = replan_or_keep
repartition_or_keep = replan_or_keep


@dataclass
class Release:
    """A small versioned release aggregate useful to stateful callers."""

    current: CohortPlan
    versions: list[CohortPlan] = field(default_factory=list)

    @classmethod
    def first(cls, rows: Iterable[Any], **kwargs: Any) -> "Release":
        plan = publish_plan(plan_history(rows, version=1, **kwargs))
        return cls(current=plan, versions=[plan])

    def replan(self, late_rows: Iterable[Any], **kwargs: Any) -> CohortPlan:
        new_plan = replan_history(self.current, late_rows, **kwargs)
        self.current = new_plan
        self.versions.append(new_plan)
        return new_plan

    def try_replan(self, late_rows: Iterable[Any], **kwargs: Any) -> ReplanResult:
        old = self.current
        try:
            new = self.replan(late_rows, **kwargs)
            return ReplanResult(new, changed=True, old_plan=old)
        except CohortResegmentError as exc:
            # self.current was never assigned because replan raised first.
            return ReplanResult(old, changed=False, error=exc, old_plan=old)


class CohortSplitter:
    """Small object-oriented facade over the module-level functions."""

    plan = staticmethod(plan_history)
    split = staticmethod(split_history)
    select_cut_points = staticmethod(select_cut_points)
    replan = staticmethod(replan_history)
    replan_or_keep = staticmethod(replan_or_keep)
    publish = staticmethod(publish_plan)

    @staticmethod
    def read_grouping(report: Any) -> dict[str, Any]:
        return read_grouping(report)


class CohortReleaser(CohortSplitter):
    """Naming alias for callers modeling first release and replanning."""


CohortPlanner = CohortSplitter
HistoryPlanner = CohortSplitter


class ReleaseRegistry:
    """Named release versions for application/service-level tests."""

    def __init__(self) -> None:
        self._releases: dict[str, Release] = {}

    def publish(self, key: str, rows: Iterable[Any] | None = None, *, plan: CohortPlan | None = None, **kwargs: Any) -> CohortPlan:
        if plan is None:
            if rows is None:
                raise ValueError("Provide rows or a plan")
            release = Release.first(rows, **kwargs)
        else:
            locked = publish_plan(plan)
            release = Release(current=locked, versions=[locked])
        self._releases[key] = release
        return release.current

    def get_release(self, key: str) -> Release:
        if key not in self._releases:
            raise KeyError(key)
        return self._releases[key]

    def get(self, key: str) -> CohortPlan:
        return self.get_release(key).current

    def replan(self, key: str, late_rows: Iterable[Any], **kwargs: Any) -> CohortPlan:
        return self.get_release(key).replan(late_rows, **kwargs)

    def try_replan(self, key: str, late_rows: Iterable[Any], **kwargs: Any) -> ReplanResult:
        return self.get_release(key).try_replan(late_rows, **kwargs)


# ----------------------------------------------------------------------
# Report read-back
# ----------------------------------------------------------------------


def _canonical_legacy_cohorts(rows: Any) -> list[dict[str, Any]] | None:
    if not isinstance(rows, list):
        return None
    result: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, Mapping) and "stage" in row:
            result.append(dict(row))
    return result or None


def read_grouping(report: str | bytes | Mapping[str, Any]) -> dict[str, Any]:
    """Read a grouping from a new report or an old triage report.

    New reports contain ``result.grouping``.  Old reports only contain
    ``result.tables.cohorts``; that aggregate is returned unchanged with
    ``mode='legacy_report'`` so callers can still display the original group.
    """

    if isinstance(report, (str, bytes)):
        report = json.loads(report)
    if not isinstance(report, Mapping):
        raise ValueError("报告须为 JSON 对象")

    result = report.get("result")
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except json.JSONDecodeError:
            result = None
    if not isinstance(result, Mapping):
        # A domain result may be supplied directly.
        result = report

    grouping = result.get("grouping")
    if isinstance(grouping, Mapping):
        data = dict(grouping)
        data.setdefault("source", "grouping")
        return data

    tables = result.get("tables")
    cohorts_table = tables.get("cohorts") if isinstance(tables, Mapping) else None
    if isinstance(cohorts_table, list):
        return {
            "schema": "meter-margin.cohort-grouping/v1",
            "mode": "legacy_report",
            "source": "legacy_report",
            "locked": False,
            "cohorts": [dict(row) for row in cohorts_table if isinstance(row, Mapping)],
            # Old reports only retained aggregate counts/time windows.
            "assignments": None,
            "holdout_record_ids": None,
            "holdout_sha256": result.get("summary", {}).get("holdout_sha256")
            if isinstance(result.get("summary"), Mapping)
            else None,
        }

    # Very old/direct grouping mappings.
    for key in ("train", "validation", "holdout"):
        if key in result:
            return {
                "mode": "direct_grouping",
                "source": "direct_grouping",
                "cohorts": _canonical_legacy_cohorts(result.get(key)),
                "assignments": result.get("assignments"),
            }

    raise ValueError("报告中找不到可回读的队列分组结果")


read_report_grouping = read_grouping
load_grouping = read_grouping
grouping_from_report = read_grouping
read_original_grouping = read_grouping
read_cohort_grouping = read_grouping
read_cohort_assignment = read_grouping
read_cohort_assignments = read_grouping
read_cohorts = read_grouping
parse_grouping = read_grouping
grouping_from_legacy_report = read_grouping


def grouping_fingerprint(grouping: Mapping[str, Any] | CohortPlan) -> str:
    """Stable fingerprint for a grouping snapshot/plan."""

    payload = grouping.snapshot() if isinstance(grouping, CohortPlan) else dict(grouping)
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()
