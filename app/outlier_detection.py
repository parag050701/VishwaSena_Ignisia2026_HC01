"""Lab outlier detection utilities used before LLM synthesis."""

from __future__ import annotations

import math
from typing import Any, Dict, List


def detect_lab_outliers(
    labs: Dict[str, Any],
    z_threshold: float = 2.5,
    min_points: int = 3,
) -> List[Dict[str, Any]]:
    outliers: List[Dict[str, Any]] = []

    for name, arr in (labs or {}).items():
        if not isinstance(arr, list) or len(arr) < min_points:
            continue

        try:
            values = [float(x["v"] if isinstance(x, dict) else x) for x in arr]
        except (KeyError, TypeError, ValueError):
            continue

        history = values[:-1]
        latest_v = values[-1]
        if not history:
            continue

        mean = sum(history) / len(history)
        variance = sum((x - mean) ** 2 for x in history) / len(history)
        std = math.sqrt(variance)
        if std < 1e-6:
            continue

        z = abs((latest_v - mean) / std)
        if z > z_threshold:
            last = arr[-1] if isinstance(arr[-1], dict) else {}
            outliers.append(
                {
                    "lab": name,
                    "value": latest_v,
                    "mean": round(mean, 2),
                    "std": round(std, 2),
                    "z": round(z, 2),
                    "timepoint": last.get("t", "?") if isinstance(last, dict) else "?",
                    "action": "HOLD DIAGNOSIS — Recommend confirmed redraw",
                }
            )

    return outliers
