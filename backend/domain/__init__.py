"""水表检定故障间隔分诊。"""

from .resegment import (
    CohortPlan,
    CohortResegmentError,
    HoldoutLockError,
    NoViableCutError,
    ReleaseRegistry,
    ReplanResult,
    plan_history,
    publish_plan,
    read_grouping,
    replan_history,
    replan_or_keep,
    split_history,
)

ReplanError = CohortResegmentError
GroupingError = CohortResegmentError
NoViableSplitError = NoViableCutError
LockedHoldoutError = HoldoutLockError

__all__ = [
    "CohortPlan",
    "CohortResegmentError",
    "HoldoutLockError",
    "NoViableCutError",
    "ReleaseRegistry",
    "ReplanResult",
    "plan_history",
    "publish_plan",
    "read_grouping",
    "replan_history",
    "replan_or_keep",
    "split_history",
]
