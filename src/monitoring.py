"""Data and model quality monitoring helpers."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from src.config import RAW_NUMERIC_FEATURES, TARGET_COL


def compute_data_quality_report(df: pd.DataFrame, label: str = "train") -> dict:
    """Basic data quality metrics for monitoring."""
    report = {
        "dataset": label,
        "rows": len(df),
        "null_counts": df.isnull().sum().to_dict(),
        "target_rate": float(df[TARGET_COL].mean()) if TARGET_COL in df.columns else None,
        "type_distribution": df["Type"].value_counts().to_dict() if "Type" in df.columns else {},
    }
    for col in RAW_NUMERIC_FEATURES:
        if col in df.columns:
            report[f"mean_{col}"] = float(df[col].mean())
            report[f"std_{col}"] = float(df[col].std())
    return report


def population_stability_index(
    expected: pd.Series, actual: pd.Series, bins: int = 10
) -> float:
    """Simple PSI for drift detection on a numeric feature."""
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1,
    )
    expected_pct = pd.cut(expected, breakpoints, duplicates="drop").value_counts(
        normalize=True
    )
    actual_pct = pd.cut(actual, breakpoints, duplicates="drop").value_counts(
        normalize=True
    )
    aligned = pd.concat([expected_pct, actual_pct], axis=1, join="outer").fillna(0.0001)
    aligned.columns = ["expected", "actual"]
    psi = ((aligned["actual"] - aligned["expected"]) * np.log(
        aligned["actual"] / aligned["expected"]
    )).sum()
    return float(psi)


def compare_distributions(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """Compare reference vs current batch means and PSI."""
    drift = {}
    for col in RAW_NUMERIC_FEATURES:
        if col not in reference.columns or col not in current.columns:
            continue
        drift[col] = {
            "mean_shift": float(current[col].mean() - reference[col].mean()),
            "psi": population_stability_index(reference[col], current[col]),
        }
    return drift


def infrastructure_snapshot() -> dict:
    """CPU/RAM snapshot for infrastructure monitoring."""
    mem = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_total_gb": round(mem.total / (1024**3), 2),
        "ram_used_percent": mem.percent,
    }


def save_monitoring_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
