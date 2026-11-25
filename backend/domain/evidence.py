from .margin_model import Scale
from .models import FEATURES, Fault
from .routing import Decision


def scaling_table(scales: dict[str, Scale], selected_fields: set[str]) -> list[dict]:
    return [
        {
            "feature": field,
            "feature_name": FEATURES[field].name,
            "unit": FEATURES[field].unit,
            "used_by_model": field in selected_fields,
            "train_mean": scale.mean,
            "train_stddev": scale.stddev,
            "scale_used": scale.divisor,
            "train_min": scale.train_min,
            "train_max": scale.train_max,
            "support_min": scale.bounds[0],
            "support_max": scale.bounds[1],
            "bench_min": FEATURES[field].bench_min,
            "bench_max": FEATURES[field].bench_max,
        }
        for field, scale in scales.items()
    ]


def coefficient_table(weights: dict[Fault, dict[str, float]], intercepts: dict[Fault, float]) -> list[dict]:
    return [
        {"label": label, "feature": field, "coefficient": coefficient, "intercept": intercepts[label]}
        for label, fields in weights.items() for field, coefficient in fields.items()
    ]


def case_evidence(
    decisions: list[Decision], scales: dict[str, Scale], weights: dict[Fault, dict[str, float]]
) -> list[dict]:
    evidence = []
    for decision in decisions:
        row = decision.row
        leading = weights[decision.scores.winner]
        runner_up = weights[decision.scores.runner_up]
        for field, scale in scales.items():
            used = field in leading
            z = scale.transform(row)
            support_min, support_max = scale.bounds
            evidence.append({
                "record_id": row.record_id,
                "feature": field,
                "feature_name": FEATURES[field].name,
                "unit": FEATURES[field].unit,
                "measured_value": getattr(row, field),
                "train_min": scale.train_min,
                "train_max": scale.train_max,
                "support_min": support_min,
                "support_max": support_max,
                "inside_training_support": field not in decision.training_fields,
                "inside_bench_range": field not in decision.bench_fields,
                "used_by_model": used,
                "standardized_value": z if used else None,
                "leading_contribution": leading[field] * z if used else None,
                "runner_up_contribution": runner_up[field] * z if used else None,
                "margin_contribution": (leading[field] - runner_up[field]) * z if used else None,
            })
    return evidence
