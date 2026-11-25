from datetime import datetime, timedelta, timezone
from random import Random

from .models import FEATURES

SEED_NAME = "ningbo-dn15-bench-20261125.csv"
LOCAL_TIME = timezone(timedelta(hours=8))

# Each tuple follows the six CSV measurement columns in FEATURES order.
PROFILES = {
    "normal": (-0.8, -0.4, 0.1, 5.8, 15.0, 0.18),
    "low_flow_drag": (-9.3, -2.7, -0.4, 22.0, 20.0, 0.45),
    "strainer_blockage": (-8.9, -2.9, -0.6, 13.0, 48.0, 0.55),
    "register_slip": (-4.5, -4.1, -3.6, 6.8, 15.5, 2.1),
}
SPREAD = (1.1, 0.75, 0.42, 1.4, 2.8, 0.12)
HISTORICAL_BATCHES = (
    ("train", "T01", 2, 4),
    ("train", "T02", 4, 4),
    ("train", "T03", 6, 4),
    ("validation", "V01", 12, 3),
    ("validation", "V02", 13, 3),
    ("holdout", "H01", 19, 3),
    ("holdout", "H02", 20, 3),
)
INCOMING_CASES = (
    ("W-1125-01", (-0.9, -0.2, 0.2, 6.0, 15.4, 0.2)),
    ("W-1125-02", (-9.4, -2.9, -0.5, 22.3, 20.2, 0.48)),
    ("W-1125-03", (-9.0, -3.0, -0.7, 13.2, 48.5, 0.58)),
    ("W-1125-04", (-4.6, -4.2, -3.7, 6.9, 16.0, 2.15)),
    ("W-1125-05", (-9.1, -2.8, -0.5, 17.5, 33.0, 0.50)),
    ("W-1125-06", (-10.0, -3.7, -0.8, 15.0, 72.0, 0.61)),
    ("W-1125-07", (-9.4, -2.9, -0.5, 27.0, 20.2, 0.48)),
)


def seed_rows() -> list[dict]:
    rng = Random(85)
    rows: list[dict] = []
    for stage, batch, day, count in HISTORICAL_BATCHES:
        shift = datetime(2026, 11, day, 8, 30, tzinfo=LOCAL_TIME)
        for profile_index, (label, center) in enumerate(PROFILES.items()):
            for sample_index in range(count):
                ordinal = profile_index * count + sample_index + 1
                meter_ordinal = ordinal - int(stage == "train" and sample_index == 1)
                measured_at = shift + timedelta(minutes=(ordinal - 1) * 11)
                values = tuple(
                    round(base + rng.uniform(-spread, spread), 2)
                    for base, spread in zip(center, SPREAD, strict=True)
                )
                rows.append({
                    "record_id": f"MR-{batch}-{ordinal:02d}",
                    "meter_id": f"NB-DN15-{batch}-{meter_ordinal:02d}",
                    "bench_batch_id": f"NB-B2-1126-{batch}",
                    "measured_at": measured_at.isoformat(),
                    "stage": stage,
                    "label": label,
                    **dict(zip(FEATURES, values, strict=True)),
                })
    for index, (record_id, values) in enumerate(INCOMING_CASES, start=1):
        measured_at = datetime(2026, 11, 25, 8, 20, tzinfo=LOCAL_TIME) + timedelta(minutes=(index - 1) * 8)
        rows.append({
            "record_id": record_id,
            "meter_id": f"NB-DN15-N-{index:02d}",
            "bench_batch_id": "NB-B2-1125-AM",
            "measured_at": measured_at.isoformat(),
            "stage": "incoming",
            "label": None,
            **dict(zip(FEATURES, values, strict=True)),
        })
    return rows
