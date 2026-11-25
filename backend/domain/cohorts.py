from collections import Counter
from collections.abc import Iterable, Mapping
import json
from hashlib import sha256
from typing import Any

from .models import FAULTS, STAGES, BenchRow, Stage, bench_violations
from .resegment import (
    CohortPlan,
    HoldoutLockError,
    NoViableCutError,
    plan_history,
    plan_from_marked_cohorts,
    publish_plan,
    read_grouping,
    replan_history,
    replan_or_keep,
    split_history,
)

RowInput = BenchRow | Mapping[str, object]
Cohorts = dict[Stage, list[BenchRow]]
MIN_LABEL_SUPPORT = 2

# Re-exported domain entry points for callers that naturally import from
# domain.cohorts.
auto_split_history = split_history
auto_split = split_history
split_unmarked = split_history
split_unmarked_history = split_history
split_unassigned_history = split_history
assign_cohorts = split_history
resegment = split_history
replan_cohorts = replan_history
reassign_cohorts = replan_history
repartition = replan_history
repartition_cohorts = replan_history
lock_holdout = publish_plan
read_cohorts = read_grouping
read_cohort_grouping = read_grouping


def _validate_input_rows(rows: Iterable[RowInput]) -> list[BenchRow]:
    validated: list[BenchRow] = []
    record_ids: set[str] = set()
    for index, raw in enumerate(rows, start=1):
        try:
            row = BenchRow.model_validate(raw)
        except ValueError as exc:
            raise ValueError(f"第 {index} 行检定记录格式错误：{exc}") from exc
        if row.record_id in record_ids:
            raise ValueError(f"检定记录号重复：{row.record_id}")
        record_ids.add(row.record_id)
        validated.append(row)
    return validated


def _assert_support(cohorts: Mapping[str, list[BenchRow]]) -> None:
    for stage in STAGES[:-1]:
        support = Counter(row.label for row in cohorts[stage])
        missing = [label for label in FAULTS if support[label] < MIN_LABEL_SUPPORT]
        if missing:
            raise ValueError(f"{stage} 队列每类至少须有 {MIN_LABEL_SUPPORT} 条记录：{', '.join(missing)}")


def _assert_chronological(cohorts: Mapping[str, list[BenchRow]]) -> None:
    previous: list[BenchRow] = []
    previous_stage: str | None = None
    for stage in STAGES:
        cohort = sorted(cohorts.get(stage, []), key=lambda row: (row.measured_at, row.record_id))
        if isinstance(cohorts, dict):
            cohorts[stage] = cohort  # type: ignore[index]
        if not cohort:
            continue
        if previous and previous[-1].measured_at >= cohort[0].measured_at:
            raise ValueError(f"队列测量时间重叠：{previous_stage} 须严格早于 {stage}")
        previous = cohort
        previous_stage = stage


def _partition_marked(rows: list[BenchRow]) -> Cohorts:
    cohorts: Cohorts = {stage: [] for stage in STAGES}
    batch_stages: dict[str, Stage] = {}
    meter_stages: dict[str, Stage] = {}
    for row in rows:
        stage = row.stage
        if stage is None:
            # partition_with_plan routes mixed/unmarked rows elsewhere.
            raise ValueError(f"记录未标队列，无法按显式队列校验：{row.record_id}")
        if batch_stages.setdefault(row.bench_batch_id, stage) != stage:
            raise ValueError(f"台架批次跨队列：{row.bench_batch_id}")
        if meter_stages.setdefault(row.meter_id, stage) != stage:
            raise ValueError(f"表号跨队列：{row.meter_id}")
        if stage == "incoming":
            if row.label is not None:
                raise ValueError(f"待分诊记录的标签须留空：{row.record_id}")
        else:
            if row.label is None:
                raise ValueError(f"历史检定记录缺少复核标签：{row.record_id}")
            violations = bench_violations(row)
            if violations:
                raise ValueError(f"历史记录超出台架适用范围：{row.record_id}，{', '.join(violations)}")
        cohorts[stage].append(row)

    _assert_support(cohorts)
    _assert_chronological(cohorts)
    cohorts["selection"] = cohorts["validation"]  # type: ignore[assignment]
    return cohorts


def _partition_unmarked(
    rows: list[BenchRow],
    *,
    expected_holdout_sha256: str = "",
    locked_holdout_ids: Iterable[str] | None = None,
    lock: Any = None,
) -> tuple[Cohorts, CohortPlan]:
    cohorts: Cohorts = {stage: [] for stage in STAGES}
    history: list[BenchRow] = []
    has_explicit_incoming = any(row.stage == "incoming" for row in rows)

    for row in rows:
        inferred_incoming = (
            row.stage is None and row.label is None and not has_explicit_incoming
        )
        if row.stage == "incoming" or inferred_incoming:
            if row.stage == "incoming" and row.label is not None:
                raise ValueError(f"待分诊记录的标签须留空：{row.record_id}")
            cohorts["incoming"].append(row)
            continue

        # A blank stage alongside an explicit incoming queue means malformed
        # history; without any stage information, a blank label is the incoming
        # signal used by CSV/JSON callers that omit the whole stage column.
        if row.label is None:
            raise ValueError(f"历史检定记录缺少复核标签：{row.record_id}")
        violations = bench_violations(row)
        if violations:
            raise ValueError(f"历史记录超出台架适用范围：{row.record_id}，{', '.join(violations)}")
        history.append(row)

    explicit_holdout = [row.record_id for row in history if row.stage == "holdout"]
    ids = list(locked_holdout_ids) if locked_holdout_ids is not None else explicit_holdout or None
    sha = expected_holdout_sha256 or None
    plan = plan_history(
        history,
        lock,
        locked_holdout_ids=ids,
        expected_holdout_sha256=sha,
        incoming=cohorts["incoming"],
    )

    cohorts["train"] = plan.train
    cohorts["validation"] = plan.validation
    cohorts["holdout"] = plan.holdout
    cohorts["incoming"] = sorted(cohorts["incoming"], key=lambda row: (row.measured_at, row.record_id))
    # Runtime extension: callers/tests can inspect quarantined components.
    cohorts["isolated"] = plan.isolated  # type: ignore[assignment]
    cohorts["selection"] = plan.validation  # type: ignore[assignment]
    cohorts["quarantined"] = plan.isolated  # type: ignore[assignment]

    # The planner guarantees history components do not cross.  Check incoming
    # against assigned history using the same meter/batch rule as marked data.
    batch_stages: dict[str, str] = {}
    meter_stages: dict[str, str] = {}
    for stage in STAGES:
        for row in cohorts[stage]:
            if batch_stages.setdefault(row.bench_batch_id, stage) != stage:
                raise ValueError(f"台架批次跨队列：{row.bench_batch_id}")
            if meter_stages.setdefault(row.meter_id, stage) != stage:
                raise ValueError(f"表号跨队列：{row.meter_id}")

    _assert_support(cohorts)
    _assert_chronological(cohorts)
    return cohorts, plan


def partition_with_plan(
    rows: Iterable[RowInput],
    lock: Any = None,
    *,
    expected_holdout_sha256: str = "",
    locked_holdout_ids: Iterable[str] | None = None,
    **options: Any,
) -> tuple[Cohorts, CohortPlan | None]:
    validated = _validate_input_rows(rows)
    if isinstance(lock, str) and not expected_holdout_sha256:
        expected_holdout_sha256 = lock
        lock = None
    sha = expected_holdout_sha256 or options.get("locked_holdout_sha256") or options.get("holdout_sha256") or ""
    ids = locked_holdout_ids or options.get("holdout_record_ids") or options.get("holdout_ids")

    if all(row.stage is not None for row in validated):
        return _partition_marked(validated), None

    return _partition_unmarked(
        validated,
        expected_holdout_sha256=sha,
        locked_holdout_ids=ids,
        lock=lock or options.get("locked_holdout"),
    )


def partition(
    rows: Iterable[RowInput],
    lock: Any = None,
    *,
    expected_holdout_sha256: str = "",
    locked_holdout_ids: Iterable[str] | None = None,
    **options: Any,
) -> Cohorts:
    # partition(rows, "<sha256>") remains a convenient positional form.
    if isinstance(lock, str) and not expected_holdout_sha256:
        expected_holdout_sha256 = lock
        lock = None
    cohorts, _ = partition_with_plan(
        rows,
        lock,
        expected_holdout_sha256=expected_holdout_sha256,
        locked_holdout_ids=locked_holdout_ids,
        **options,
    )
    return cohorts


def cohort_table(cohorts: Mapping[str, list[BenchRow]]) -> list[dict]:
    return [
        {
            "stage": stage,
            "row_count": len(cohorts[stage]),
            "meter_count": len({row.meter_id for row in cohorts[stage]}),
            "bench_batch_count": len({row.bench_batch_id for row in cohorts[stage]}),
            "first_measured_at": cohorts[stage][0].measured_at.isoformat() if cohorts[stage] else None,
            "last_measured_at": cohorts[stage][-1].measured_at.isoformat() if cohorts[stage] else None,
        }
        for stage in STAGES
    ]


def cohort_digest(rows: list[BenchRow]) -> str:
    canonical = json.dumps(
        [row.model_dump(mode="json") for row in rows],
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return sha256(canonical.encode("utf-8")).hexdigest()
