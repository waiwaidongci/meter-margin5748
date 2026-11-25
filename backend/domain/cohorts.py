from collections import Counter
from collections.abc import Iterable, Mapping
import json
from hashlib import sha256

from .models import FAULTS, STAGES, BenchRow, Stage, bench_violations

RowInput = BenchRow | Mapping[str, object]
Cohorts = dict[Stage, list[BenchRow]]
MIN_LABEL_SUPPORT = 2


def partition(rows: Iterable[RowInput]) -> Cohorts:
    cohorts: Cohorts = {stage: [] for stage in STAGES}
    record_ids: set[str] = set()
    batch_stages: dict[str, Stage] = {}
    meter_stages: dict[str, Stage] = {}
    for index, raw in enumerate(rows, start=1):
        try:
            row = BenchRow.model_validate(raw)
        except ValueError as exc:
            raise ValueError(f"第 {index} 行检定记录格式错误：{exc}") from exc
        if row.record_id in record_ids:
            raise ValueError(f"检定记录号重复：{row.record_id}")
        record_ids.add(row.record_id)
        if batch_stages.setdefault(row.bench_batch_id, row.stage) != row.stage:
            raise ValueError(f"台架批次跨队列：{row.bench_batch_id}")
        if meter_stages.setdefault(row.meter_id, row.stage) != row.stage:
            raise ValueError(f"表号跨队列：{row.meter_id}")
        if row.stage == "incoming":
            if row.label is not None:
                raise ValueError(f"待分诊记录的标签须留空：{row.record_id}")
        else:
            if row.label is None:
                raise ValueError(f"历史检定记录缺少复核标签：{row.record_id}")
            violations = bench_violations(row)
            if violations:
                raise ValueError(f"历史记录超出台架适用范围：{row.record_id}，{', '.join(violations)}")
        cohorts[row.stage].append(row)

    for stage in STAGES[:-1]:
        support = Counter(row.label for row in cohorts[stage])
        missing = [label for label in FAULTS if support[label] < MIN_LABEL_SUPPORT]
        if missing:
            raise ValueError(f"{stage} 队列每类至少须有 {MIN_LABEL_SUPPORT} 条记录：{', '.join(missing)}")

    previous: list[BenchRow] = []
    for stage in STAGES:
        cohort = cohorts[stage]
        cohort.sort(key=lambda row: (row.measured_at, row.record_id))
        if not cohort:
            continue
        if previous and previous[-1].measured_at >= cohort[0].measured_at:
            raise ValueError(f"队列测量时间重叠：{previous[-1].stage} 须严格早于 {stage}")
        previous = cohort
    return cohorts


def cohort_table(cohorts: Cohorts) -> list[dict]:
    return [
        {
            "stage": stage,
            "row_count": len(rows),
            "meter_count": len({row.meter_id for row in rows}),
            "bench_batch_count": len({row.bench_batch_id for row in rows}),
            "first_measured_at": rows[0].measured_at.isoformat() if rows else None,
            "last_measured_at": rows[-1].measured_at.isoformat() if rows else None,
        }
        for stage, rows in cohorts.items()
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
