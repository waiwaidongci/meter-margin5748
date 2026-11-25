from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import cast

from sklearn.svm import LinearSVC

from .models import FEATURES, BenchRow, Fault

MAX_ITERATIONS = 10_000
TRAIN_RANGE_PAD = 0.5


@dataclass(frozen=True)
class Scale:
    field: str
    mean: float
    stddev: float
    divisor: float
    train_min: float
    train_max: float

    def transform(self, row: BenchRow) -> float:
        return (getattr(row, self.field) - self.mean) / self.divisor

    @property
    def bounds(self) -> tuple[float, float]:
        padding = TRAIN_RANGE_PAD * self.divisor
        return self.train_min - padding, self.train_max + padding

    def covers(self, row: BenchRow) -> bool:
        lower, upper = self.bounds
        return lower <= getattr(row, self.field) <= upper


def fit_scales(train: list[BenchRow]) -> dict[str, Scale]:
    scales = {}
    for field, feature in FEATURES.items():
        values = [getattr(row, field) for row in train]
        stddev = pstdev(values)
        scales[field] = Scale(field, fmean(values), stddev, max(stddev, feature.resolution), min(values), max(values))
    return scales


def design_matrix(rows: list[BenchRow], scales: tuple[Scale, ...]) -> list[list[float]]:
    return [[scale.transform(row) for scale in scales] for row in rows]


@dataclass(frozen=True)
class DecisionScores:
    ranking: tuple[Fault, ...]
    values: dict[Fault, float]

    @classmethod
    def from_values(cls, values: dict[Fault, float]) -> "DecisionScores":
        return cls(tuple(sorted(values, key=lambda label: (-values[label], label))), values)

    @property
    def winner(self) -> Fault:
        return self.ranking[0]

    @property
    def runner_up(self) -> Fault:
        return self.ranking[1]

    @property
    def margin(self) -> float:
        return self.values[self.winner] - self.values[self.runner_up]


@dataclass(frozen=True)
class MarginModel:
    scales: tuple[Scale, ...]
    estimator: LinearSVC

    @classmethod
    def fit(cls, rows: list[BenchRow], scales: tuple[Scale, ...], c: float) -> "MarginModel":
        estimator = LinearSVC(
            C=c, penalty="l2", loss="squared_hinge", dual=False, multi_class="ovr",
            max_iter=MAX_ITERATIONS, tol=1e-8,
        )
        estimator.fit(design_matrix(rows, scales), [row.label for row in rows])
        if estimator.n_iter_ >= MAX_ITERATIONS:
            raise ValueError(f"线性 SVM 未在迭代上限内收敛（C={c:g}）")
        return cls(scales, estimator)

    @property
    def labels(self) -> tuple[Fault, ...]:
        return tuple(cast(Fault, str(label)) for label in self.estimator.classes_)

    def score_rows(self, rows: list[BenchRow]) -> list[DecisionScores]:
        if not rows:
            return []
        values = self.estimator.decision_function(design_matrix(rows, self.scales)).tolist()
        labels = self.labels
        return [DecisionScores.from_values(dict(zip(labels, scores, strict=True))) for scores in values]

    def intercepts(self) -> dict[Fault, float]:
        return dict(zip(self.labels, self.estimator.intercept_.tolist(), strict=True))

    def weights(self) -> dict[Fault, dict[str, float]]:
        fields = [scale.field for scale in self.scales]
        return {
            label: dict(zip(fields, values, strict=True))
            for label, values in zip(self.labels, self.estimator.coef_.tolist(), strict=True)
        }
