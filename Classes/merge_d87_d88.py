from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_csv(path: Path) -> pd.DataFrame:
    """Load telemetry CSV and normalize numeric columns."""
    df = pd.read_csv(path)
    expected = ["Tick", "Throttle", "Speed", "Current", "Voltage"]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    for col in expected:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["Tick"].isna().any():
        raise ValueError(f"{path} contains non-numeric Tick values")

    return df[expected].copy()


def _positive_diffs(series: pd.Series) -> np.ndarray:
    diffs = series.diff().dropna().to_numpy(dtype=float)
    return diffs[diffs > 0]


def _interval_stats(diffs: np.ndarray) -> tuple[float, float, float]:
    if diffs.size == 0:
        raise ValueError("Not enough tick data to calculate intervals")
    median = float(np.median(diffs))
    p05 = float(np.percentile(diffs, 5))
    p95 = float(np.percentile(diffs, 95))
    return median, p05, p95


def _remap_ticks_like_reference(ticks_ref: pd.Series, ticks_to_adjust: pd.Series) -> np.ndarray:
    """Scale d_88 tick intervals so they match d_87 interval pattern, then append."""
    diffs_ref = _positive_diffs(ticks_ref)
    diffs_adj = _positive_diffs(ticks_to_adjust)

    median_ref, p05_ref, p95_ref = _interval_stats(diffs_ref)
    median_adj, _, _ = _interval_stats(diffs_adj)

    scale = median_ref / median_adj if median_adj > 0 else 1.0

    # Use interval distances first (as requested), then clamp outliers for smoother cadence.
    raw_diffs = np.diff(ticks_to_adjust.to_numpy(dtype=float))
    scaled_diffs = raw_diffs * scale
    smooth_diffs = np.clip(np.round(scaled_diffs), max(1.0, p05_ref), p95_ref)

    start_tick = float(ticks_ref.iloc[-1]) + median_ref
    out = np.empty(len(ticks_to_adjust), dtype=np.int64)
    out[0] = int(round(start_tick))

    for i, delta in enumerate(smooth_diffs, start=1):
        out[i] = out[i - 1] + int(round(delta))

    return out


def merge_runs(run87: pd.DataFrame, run88: pd.DataFrame) -> pd.DataFrame:
    run88_adj = run88.copy()
    run88_adj["Tick"] = _remap_ticks_like_reference(run87["Tick"], run88["Tick"])

    merged = pd.concat([run87, run88_adj], ignore_index=True)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge d_87 and d_88 with smooth Tick continuity")
    parser.add_argument("--run87", type=Path, default=Path("../Data/d_87.csv"))
    parser.add_argument("--run88", type=Path, default=Path("../Data/d_88.csv"))
    parser.add_argument("--out", type=Path, default=Path("../Data/d_89.csv"))
    args = parser.parse_args()

    df87 = _load_csv(args.run87)
    df88 = _load_csv(args.run88)

    merged = merge_runs(df87, df88)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)

    diffs = _positive_diffs(merged["Tick"])
    median_dt, p05_dt, p95_dt = _interval_stats(diffs)
    print(f"Merged rows: {len(merged)}")
    print(f"Output: {args.out}")
    print(f"Tick interval stats (ms): median={median_dt:.1f}, p05={p05_dt:.1f}, p95={p95_dt:.1f}")


if __name__ == "__main__":
    main()
