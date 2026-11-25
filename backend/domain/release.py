from dataclasses import dataclass

from .margin_model import MarginModel
from .models import FAULT_NAMES, BenchRow
from .scores import Scores, score

MIN_BALANCED_ACCURACY = 0.80
MIN_CLASS_RECALL = 0.67


@dataclass(frozen=True)
class HoldoutReport:
    scores: Scores
    predictions: list[dict]
    failures: list[str]

    @property
    def passed(self) -> bool:
        return not self.failures


def evaluate_holdout(model: MarginModel, holdout: list[BenchRow]) -> HoldoutReport:
    decisions = model.score_rows(holdout)
    metrics = score(holdout, decisions)
    failures = []
    if metrics.balanced_accuracy < MIN_BALANCED_ACCURACY:
        failures.append(f"留出集平衡准确率 {metrics.balanced_accuracy:.1%} 低于 {MIN_BALANCED_ACCURACY:.0%}")
    for label, recall in metrics.recall.items():
        if recall < MIN_CLASS_RECALL:
            failures.append(f"留出集「{FAULT_NAMES[label]}」召回率 {recall:.1%} 低于 {MIN_CLASS_RECALL:.0%}")
    predictions = [
        {
            "record_id": row.record_id,
            "meter_id": row.meter_id,
            "bench_batch_id": row.bench_batch_id,
            "actual": row.label,
            "predicted": decision.winner,
            "leading_score": decision.values[decision.winner],
            "runner_up_label": decision.runner_up,
            "runner_up_score": decision.values[decision.runner_up],
            "decision_margin": decision.margin,
            "correct": row.label == decision.winner,
        }
        for row, decision in zip(holdout, decisions, strict=True)
    ]
    return HoldoutReport(metrics, predictions, failures)
