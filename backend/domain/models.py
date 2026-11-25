from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Literal, get_args

from pydantic import AwareDatetime, BaseModel, BeforeValidator, ConfigDict, Field, field_validator

Stage = Literal["train", "validation", "holdout", "incoming"]
Fault = Literal["normal", "low_flow_drag", "strainer_blockage", "register_slip"]
STAGES: tuple[Stage, ...] = get_args(Stage)
FAULTS: tuple[Fault, ...] = get_args(Fault)
FAULT_NAMES = {
    "normal": "未见故障",
    "low_flow_drag": "低流量阻滞",
    "strainer_blockage": "滤网堵塞",
    "register_slip": "计数机构打滑",
}


def reject_boolean(value: object) -> object:
    if isinstance(value, bool):
        raise ValueError("测量值和阈值须为数值")
    return value


Number = Annotated[float, BeforeValidator(reject_boolean)]
Identifier = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Z0-9][A-Z0-9_-]*$")]


@dataclass(frozen=True)
class Feature:
    name: str
    unit: str
    bench_min: float
    bench_max: float
    resolution: float


FEATURES = {
    "q1_error_pct": Feature("Q1 示值误差", "%", -18.0, 8.0, 0.01),
    "q2_error_pct": Feature("Q2 示值误差", "%", -10.0, 6.0, 0.01),
    "q3_error_pct": Feature("Q3 示值误差", "%", -6.0, 6.0, 0.01),
    "start_flow_lph": Feature("始动流量", "L/h", 1.0, 32.0, 0.1),
    "pressure_loss_kpa": Feature("Q3 压损", "kPa", 0.0, 65.0, 0.1),
    "repeatability_pct": Feature("Q2 误差极差", "百分点", 0.0, 6.0, 0.01),
}


class BenchRow(BaseModel):
    model_config = ConfigDict(
        extra="forbid", allow_inf_nan=False, str_strip_whitespace=True, revalidate_instances="always"
    )

    record_id: Identifier = Field(title="检定记录号")
    meter_id: Identifier = Field(title="表号")
    bench_batch_id: Identifier = Field(title="台架批次号")
    measured_at: AwareDatetime = Field(title="测量时间（带时区）")
    stage: Stage = Field(title="数据队列")
    label: Fault | None = Field(default=None, title="复核标签")
    q1_error_pct: Number = Field(ge=-100, le=1_000, title="Q1 示值误差（%）")
    q2_error_pct: Number = Field(ge=-100, le=1_000, title="Q2 示值误差（%）")
    q3_error_pct: Number = Field(ge=-100, le=1_000, title="Q3 示值误差（%）")
    start_flow_lph: Number = Field(ge=0, le=10_000, title="始动流量（L/h）")
    pressure_loss_kpa: Number = Field(ge=0, le=10_000, title="Q3 压损（kPa）")
    repeatability_pct: Number = Field(ge=0, le=1_000, title="Q2 误差极差（百分点）")

    @field_validator("label", mode="before")
    @classmethod
    def empty_csv_label(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @field_validator("measured_at", mode="before")
    @classmethod
    def iso_timestamp(cls, value: object) -> object:
        if isinstance(value, str):
            return datetime.fromisoformat(value.strip())
        if not isinstance(value, datetime):
            raise ValueError("测量时间须使用带时区的 ISO 8601 格式")
        return value

    @field_validator("measured_at")
    @classmethod
    def utc_timestamp(cls, value: datetime) -> datetime:
        try:
            return value.astimezone(timezone.utc)
        except OverflowError as exc:
            raise ValueError("测量时间超出可表示范围") from exc


class TriageOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, revalidate_instances="always")

    min_decision_margin: Number = Field(default=0.20, ge=0, title="首选与次选最低 SVM 决策分差（分值）")
    expected_holdout_sha256: str = Field(
        default="", pattern=r"^([0-9a-f]{64})?$", title="已锁定的留出集 SHA-256（可空）"
    )


def bench_violations(row: BenchRow) -> list[str]:
    return [
        field for field, feature in FEATURES.items()
        if not feature.bench_min <= getattr(row, field) <= feature.bench_max
    ]
