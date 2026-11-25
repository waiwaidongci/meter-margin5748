from dataclasses import dataclass

from .margin_model import DecisionScores, Scale
from .models import FAULT_NAMES, BenchRow, Fault, TriageOptions, bench_violations

ROUTES = {
    "normal": "检定台复测并核对原始记录",
    "low_flow_drag": "机芯工位检查叶轮与轴承阻滞",
    "strainer_blockage": "过滤组件工位拆检滤网并复测压损",
    "register_slip": "计数工位核对传动与累计读数",
}
REVIEW_ROUTE = "计量复核台复测并人工派工"


@dataclass(frozen=True)
class Decision:
    row: BenchRow
    scores: DecisionScores
    bench_fields: list[str]
    training_fields: list[str]
    reasons: list[str]

    @property
    def accepted(self) -> bool:
        return not self.reasons


def triage(
    row: BenchRow, scores: DecisionScores, scales: dict[str, Scale], options: TriageOptions, released: bool
) -> Decision:
    bench_fields = bench_violations(row)
    training_fields = [field for field, scale in scales.items() if not scale.covers(row)]
    reasons = []
    if not released:
        reasons.append("holdout_failed")
    if bench_fields:
        reasons.append("bench_out_of_range")
    if training_fields:
        reasons.append("outside_training_support")
    if scores.margin == 0 or scores.margin < options.min_decision_margin:
        reasons.append("insufficient_decision_margin")
    return Decision(row, scores, bench_fields, training_fields, reasons)


def routing_row(decision: Decision, intercepts: dict[Fault, float]) -> dict:
    row, scores = decision.row, decision.scores
    leading, runner_up = scores.winner, scores.runner_up
    return {
        "record_id": row.record_id,
        "meter_id": row.meter_id,
        "bench_batch_id": row.bench_batch_id,
        "measured_at": row.measured_at.isoformat(),
        "status": "accepted" if decision.accepted else "review",
        "suggested_label": leading if decision.accepted else None,
        "leading_label": leading,
        "leading_fault": FAULT_NAMES[leading],
        "runner_up_label": runner_up,
        "leading_score": scores.values[leading],
        "runner_up_score": scores.values[runner_up],
        "decision_margin": scores.margin,
        "leading_intercept": intercepts[leading],
        "runner_up_intercept": intercepts[runner_up],
        "margin_intercept": intercepts[leading] - intercepts[runner_up],
        "reason_codes": ", ".join(decision.reasons),
        "bench_out_of_range": ", ".join(decision.bench_fields),
        "outside_training_support": ", ".join(decision.training_fields),
        "next_step": ROUTES[leading] if decision.accepted else REVIEW_ROUTE,
    }
