from collections import Counter
from dataclasses import dataclass
from statistics import fmean
from typing import cast

from .margin_model import DecisionScores
from .models import FAULTS, FAULT_NAMES, BenchRow, Fault


@dataclass(frozen=True)
class Scores:
    confusion: Counter[tuple[Fault, Fault]]
    support: Counter[Fault]
    predicted: Counter[Fault]
    recall: dict[Fault, float]
    precision: dict[Fault, float]
    f1: dict[Fault, float]

    @property
    def balanced_accuracy(self) -> float:
        return fmean(self.recall.values())

    @property
    def macro_f1(self) -> float:
        return fmean(self.f1.values())

    def class_table(self) -> list[dict]:
        return [
            {
                "label": fault,
                "fault_name": FAULT_NAMES[fault],
                "support": self.support[fault],
                "predicted": self.predicted[fault],
                "correct": self.confusion[fault, fault],
                "recall": self.recall[fault],
                "precision": self.precision[fault],
                "f1": self.f1[fault],
            }
            for fault in FAULTS
        ]

    def confusion_table(self) -> list[dict]:
        return [
            {"actual": actual, "predicted": predicted, "count": self.confusion[actual, predicted]}
            for actual in FAULTS for predicted in FAULTS
        ]


def score(rows: list[BenchRow], decisions: list[DecisionScores]) -> Scores:
    actual = [cast(Fault, row.label) for row in rows]
    winners = [decision.winner for decision in decisions]
    support = Counter(actual)
    predicted = Counter(winners)
    confusion = Counter(zip(actual, winners, strict=True))
    recall = {label: confusion[label, label] / support[label] for label in FAULTS}
    precision = {
        label: confusion[label, label] / predicted[label] if predicted[label] else 0.0 for label in FAULTS
    }
    f1 = {label: 2 * confusion[label, label] / (support[label] + predicted[label]) for label in FAULTS}
    return Scores(confusion, support, predicted, recall, precision, f1)
