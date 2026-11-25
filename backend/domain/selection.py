from dataclasses import dataclass

from .margin_model import MarginModel, Scale
from .models import FEATURES, BenchRow
from .scores import Scores, score

C_VALUES = (0.01, 0.1, 1.0, 10.0)
ERRORS = ("q1_error_pct", "q2_error_pct", "q3_error_pct")
FEATURE_SETS = {
    "flow_errors": ERRORS,
    "flow_hydraulics": (*ERRORS, "start_flow_lph", "pressure_loss_kpa"),
    "full_bench": tuple(FEATURES),
}


@dataclass(frozen=True)
class Candidate:
    feature_set: str
    c: float
    model: MarginModel
    scores: Scores

    def rank_key(self) -> tuple[float, float, float, int, float, str]:
        return (
            -self.scores.macro_f1,
            -min(self.scores.recall.values()),
            -self.scores.balanced_accuracy,
            len(self.model.scales),
            self.c,
            self.feature_set,
        )


def rank_candidates(train: list[BenchRow], validation: list[BenchRow], scales: dict[str, Scale]) -> list[Candidate]:
    candidates = []
    for name, fields in FEATURE_SETS.items():
        selected_scales = tuple(scales[field] for field in fields)
        for c in C_VALUES:
            model = MarginModel.fit(train, selected_scales, c)
            candidates.append(Candidate(name, c, model, score(validation, model.score_rows(validation))))
    return sorted(candidates, key=Candidate.rank_key)


def candidate_table(candidates: list[Candidate]) -> list[dict]:
    return [
        {
            "rank": rank,
            "selected": rank == 1,
            "feature_set": candidate.feature_set,
            "features": ", ".join(scale.field for scale in candidate.model.scales),
            "c": candidate.c,
            "validation_macro_f1": candidate.scores.macro_f1,
            "validation_worst_recall": min(candidate.scores.recall.values()),
            "validation_balanced_accuracy": candidate.scores.balanced_accuracy,
        }
        for rank, candidate in enumerate(candidates, start=1)
    ]


def candidate_class_table(candidates: list[Candidate]) -> list[dict]:
    return [
        {"feature_set": candidate.feature_set, "c": candidate.c, **row}
        for candidate in candidates
        for row in candidate.scores.class_table()
    ]
