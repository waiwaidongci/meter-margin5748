"""Compatibility entry point for time/contiguity split selection."""

from .resegment import *  # noqa: F401,F403
from .resegment import choose_cut_points, plan_history, split_cohorts, split_history
