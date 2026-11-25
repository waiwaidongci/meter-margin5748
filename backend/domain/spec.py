from collections.abc import Iterable, Mapping

from .cohorts import RowInput, cohort_digest, cohort_table, partition_with_plan
from .evidence import case_evidence, coefficient_table, scaling_table
from .margin_model import fit_scales
from .models import BenchRow, TriageOptions
from .release import MIN_BALANCED_ACCURACY, MIN_CLASS_RECALL, evaluate_holdout
from .resegment import plan_from_marked_cohorts, read_grouping
from .routing import routing_row, triage
from .seed_data import SEED_NAME, seed_rows
from .selection import candidate_class_table, candidate_table, rank_candidates

TITLE = "水表检定故障间隔分诊"
METHOD = "按表号、台架批次与时间隔离的多类线性 SVM，按决策间隔和训练支持范围分诊"
ROW_MODEL = BenchRow
OPTIONS_MODEL = TriageOptions


def record_key(row: RowInput) -> str:
    return ROW_MODEL.model_validate(row).record_id


def validate_rows(rows: Iterable[RowInput]) -> None:
    partition_with_plan(rows)


def analyze(
    rows: Iterable[RowInput], options: TriageOptions | Mapping[str, object] | None = None
) -> dict:
    settings = OPTIONS_MODEL.model_validate({} if options is None else options)
    cohorts, cohort_plan = partition_with_plan(
        rows, expected_holdout_sha256=settings.expected_holdout_sha256
    )
    if cohort_plan is None:
        cohort_plan = plan_from_marked_cohorts(
            cohorts, expected_sha=settings.expected_holdout_sha256
        )
    grouping = cohort_plan.snapshot()
    holdout_sha256 = cohort_digest(cohorts["holdout"])
    if settings.expected_holdout_sha256 and settings.expected_holdout_sha256 != holdout_sha256:
        raise ValueError("留出集指纹与锁定版本不一致，请核对台架记录及复核标签")

    scales = fit_scales(cohorts["train"])
    candidates = rank_candidates(cohorts["train"], cohorts["validation"], scales)
    chosen = candidates[0]
    model = chosen.model
    holdout = evaluate_holdout(model, cohorts["holdout"])
    incoming = cohorts["incoming"]
    decisions = [
        triage(row, scores, scales, settings, holdout.passed)
        for row, scores in zip(incoming, model.score_rows(incoming), strict=True)
    ]
    weights, intercepts = model.weights(), model.intercepts()
    accepted = sum(decision.accepted for decision in decisions)
    reviewed = len(decisions) - accepted
    warnings = list(holdout.failures)
    if reviewed:
        warnings.append(f"{reviewed} 条待分诊记录已进入人工复核队列。")
    if not decisions:
        warnings.append("本次文件仅含历史记录，暂无待分诊工单。")
    assignment_table = [
        {
            "record_id": row.record_id,
            "meter_id": row.meter_id,
            "bench_batch_id": row.bench_batch_id,
            "measured_at": row.measured_at.isoformat(),
            "stage": cohort_plan.assignments.get(row.record_id, row.stage or "isolated"),
            "label": row.label,
        }
        for row in [*cohort_plan.records, *cohort_plan.incoming]
    ]
    group_by_record = {
        record_id: group["group_id"]
        for group in grouping["groups"]
        for record_id in group["record_ids"]
    }
    isolated_table = [
        {
            "record_id": row.record_id,
            "meter_id": row.meter_id,
            "bench_batch_id": row.bench_batch_id,
            "measured_at": row.measured_at.isoformat(),
            "label": row.label,
            "group_id": group_by_record.get(row.record_id),
            "reason": "connected_meter_or_bench_batch_crosses_cut",
        }
        for row in cohort_plan.isolated
    ]
    return {
        "grouping": grouping,
        "summary": {
            "feature_set": chosen.feature_set,
            "selected_c": chosen.c,
            "candidate_count": len(candidates),
            "score_type": "linear_svm_decision_score",
            "score_is_probability": False,
            "validation_macro_f1": chosen.scores.macro_f1,
            "validation_balanced_accuracy": chosen.scores.balanced_accuracy,
            "holdout_balanced_accuracy": holdout.scores.balanced_accuracy,
            "holdout_worst_recall": min(holdout.scores.recall.values()),
            "holdout_count": len(cohorts["holdout"]),
            "isolated_count": len(cohort_plan.isolated),
            "quarantined_count": len(cohort_plan.isolated),
            "train_count": len(cohorts["train"]),
            "validation_count": len(cohorts["validation"]),
            "selection_count": len(cohorts["validation"]),
            "cohort_cut_points": [value.isoformat() for value in cohort_plan.cut_points],
            "cohort_grouping_version": cohort_plan.version,
            "cohort_grouping_locked": cohort_plan.locked,
            "holdout_sha256": holdout_sha256,
            "min_holdout_balanced_accuracy": MIN_BALANCED_ACCURACY,
            "min_holdout_class_recall": MIN_CLASS_RECALL,
            "incoming_count": len(decisions),
            "accepted_count": accepted,
            "review_count": reviewed,
            **settings.model_dump(),
        },
        "tables": {
            "cohorts": cohort_table(cohorts),
            "cohort_assignments": assignment_table,
            "cohort_groups": grouping["groups"],
            "isolated_records": isolated_table,
            "quarantined_records": isolated_table,
            "candidates": candidate_table(candidates),
            "validation_class_scores": candidate_class_table(candidates),
            "validation_confusion": chosen.scores.confusion_table(),
            "scaling": scaling_table(scales, {scale.field for scale in model.scales}),
            "coefficients": coefficient_table(weights, intercepts),
            "holdout_recall": holdout.scores.class_table(),
            "holdout_confusion": holdout.scores.confusion_table(),
            "holdout_predictions": holdout.predictions,
            "triage": [routing_row(decision, intercepts) for decision in decisions],
            "case_evidence": case_evidence(decisions, scales, weights),
        },
        "warnings": warnings,
        "publishable": holdout.passed,
    }


def seed() -> list[dict]:
    return seed_rows()
