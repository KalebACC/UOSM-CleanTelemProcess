"""Telemetry preprocessing and track-mapping helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TelemetryConfig:
    """Configuration used for telemetry preprocessing."""

    wheel_diam_m: float = 0.55
    encoder_ticks_per_rev: int = 4
    track_len_m: float = 3832.5
    speed_scale: float = 1.06
    chop_seconds: int = 0
    start_coords: tuple[float, float] = (39.792255, -86.238697)


def prepare_telemetry(
    telemetry_df: pd.DataFrame,
    config: TelemetryConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Normalize telemetry columns and compute distance metrics."""
    if telemetry_df is None or telemetry_df.empty:
        raise ValueError("Telemetry dataframe is empty. Load data before preprocessing.")

    df = telemetry_df.copy()

    if config.chop_seconds > 0:
        chop_ms = config.chop_seconds * 1000
        df = df[df["Tick"] >= chop_ms].copy()
        if df.empty:
            raise ValueError("No data remains after chop_seconds filtering.")
        df["Tick"] = df["Tick"] - df["Tick"].iloc[0]
        df["TimePlot"] = df["Tick"] / 60_000

    circumference = np.pi * config.wheel_diam_m
    dist_per_tick = circumference / config.encoder_ticks_per_rev

    df["Speed_kph"] = df["Speed"] * 0.001 * config.speed_scale
    df["Speed_mps"] = df["Speed_kph"] / 3.6
    df["Voltage_V"] = df["Voltage"] / 1000.0
    df["Current_A"] = df["Current"] / 1000.0
    df["dt"] = df["Tick"].diff().fillna(0) / 1000.0
    df["Distance_m"] = np.cumsum(df["Speed_mps"].values * df["dt"].values)

    total_dist = float(df["Distance_m"].iloc[-1])
    n_laps = total_dist / config.track_len_m if config.track_len_m else 0.0
    moving = df.loc[df["Speed_kph"] > 1, "Speed_kph"]

    summary = {
        "circumference_m": float(circumference),
        "dist_per_tick_m": float(dist_per_tick),
        "total_dist_m": total_dist,
        "total_dist_km": total_dist / 1000.0,
        "n_laps": float(n_laps),
        "max_speed_kph": float(df["Speed_kph"].max()),
        "avg_moving_speed_kph": float(moving.mean()) if not moving.empty else 0.0,
        "min_voltage_v": float(df["Voltage_V"].min()),
        "max_voltage_v": float(df["Voltage_V"].max()),
        "min_current_a": float(df["Current_A"].min()),
        "max_current_a": float(df["Current_A"].max()),
    }
    return df, summary


def load_track_centerline(track_csv_path: str) -> dict[str, np.ndarray | float]:
    """Load GPS centreline and build local XY + arc-length arrays."""
    track_gps = pd.read_csv(track_csv_path)
    gps_lat = track_gps["Latitude"].values
    gps_lon = track_gps["Longitude"].values

    lat_ref = gps_lat.mean()
    lon_ref = gps_lon.mean()
    m_per_deg_lat = 110_540
    m_per_deg_lon = 111_320 * np.cos(np.radians(lat_ref))

    track_x = (gps_lon - lon_ref) * m_per_deg_lon
    track_y = (gps_lat - lat_ref) * m_per_deg_lat
    track_xy = np.column_stack([track_x, track_y])

    seg_lens = np.sqrt(np.sum(np.diff(track_xy, axis=0) ** 2, axis=1))
    arc = np.zeros(len(track_xy))
    arc[1:] = np.cumsum(seg_lens)
    total_arc = float(arc[-1])

    return {
        "gps_lat": gps_lat,
        "gps_lon": gps_lon,
        "track_xy": track_xy,
        "arc": arc,
        "total_arc": total_arc,
    }


def add_track_position(
    telemetry_df: pd.DataFrame,
    track: dict[str, np.ndarray | float],
    config: TelemetryConfig,
) -> tuple[pd.DataFrame, float, int]:
    """Compute TrackPos based on distance and start coordinate offset."""
    gps_lat = track["gps_lat"]
    gps_lon = track["gps_lon"]
    arc = track["arc"]
    total_arc = float(track["total_arc"])

    dists_to_start = (gps_lat - config.start_coords[0]) ** 2 + (gps_lon - config.start_coords[1]) ** 2
    start_idx = int(np.argmin(dists_to_start))
    start_arc = arc[start_idx]
    start_offset = (start_arc / total_arc) * config.track_len_m

    df = telemetry_df.copy()
    df["TrackPos"] = (df["Distance_m"] + start_offset) % config.track_len_m
    return df, float(start_offset), start_idx


def dist_to_xy_factory(
    track: dict[str, np.ndarray | float],
    track_len_m: float,
):
    """Return a function that maps track distance to local XY coordinate."""
    arc = track["arc"]
    total_arc = float(track["total_arc"])
    track_xy = track["track_xy"]

    def dist_to_xy(distance_m: float) -> np.ndarray:
        arc_pos = (distance_m / track_len_m) * total_arc
        idx = min(np.searchsorted(arc, arc_pos % total_arc), len(track_xy) - 1)
        return track_xy[idx]

    return dist_to_xy


def prepare_telemetry_with_track(
    telemetry_df: pd.DataFrame,
    track_csv_path: str,
    config: TelemetryConfig,
) -> tuple[pd.DataFrame, dict[str, np.ndarray | float], dict[str, float], object]:
    """End-to-end helper returning prepared telemetry, track, summary, and mapper."""
    df, summary = prepare_telemetry(telemetry_df=telemetry_df, config=config)
    track = load_track_centerline(track_csv_path=track_csv_path)
    df, start_offset, start_idx = add_track_position(telemetry_df=df, track=track, config=config)
    summary["start_offset_m"] = float(start_offset)
    summary["start_idx"] = int(start_idx)
    dist_to_xy = dist_to_xy_factory(track=track, track_len_m=config.track_len_m)
    return df, track, summary, dist_to_xy
