"""Reusable telemetry analysis plots and track-map helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


def plot_speed_vs_time(df: pd.DataFrame) -> None:
    """Plot vehicle speed over time."""
    plt.figure(figsize=(12, 4))
    plt.plot(df["TimePlot"], df["Speed_kph"], alpha=0.6, linewidth=0.8)
    plt.xlabel("Time (min)")
    plt.ylabel("Speed (km/h)")
    plt.title("Speed vs Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_acceleration_profile(
    df: pd.DataFrame,
    smooth_window: int = 21,
) -> pd.Series:
    """Plot acceleration profile and return acceleration series."""
    speed_smooth = df["Speed_mps"].rolling(smooth_window, center=True, min_periods=1).mean()
    accel = speed_smooth.diff() / df["dt"]
    accel = accel.fillna(0).clip(-5, 5)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [2, 1]})

    ax = axes[0]
    ax.plot(df["TimePlot"], accel, alpha=0.5, linewidth=0.6, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.fill_between(df["TimePlot"], accel, 0, where=accel > 0, alpha=0.15, color="green", label="Accelerating")
    ax.fill_between(df["TimePlot"], accel, 0, where=accel < 0, alpha=0.15, color="red", label="Decelerating")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Acceleration (m/s^2)")
    ax.set_title("Acceleration Profile")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.hist(accel[accel != 0], bins=80, orientation="horizontal", color="steelblue", alpha=0.7, edgecolor="none")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Count")
    ax2.set_ylabel("Acceleration (m/s^2)")
    ax2.set_title("Distribution")
    ax2.grid(True, alpha=0.3)

    moving = df["Speed_kph"] > 1
    accel_moving = accel[moving]
    t_accel = (accel_moving > 0.05).sum()
    t_decel = (accel_moving < -0.05).sum()
    t_cruise = ((accel_moving >= -0.05) & (accel_moving <= 0.05)).sum()
    total = t_accel + t_decel + t_cruise
    if total > 0:
        fig.text(
            0.5,
            -0.04,
            f"While moving:  Accelerating {t_accel / total * 100:.1f}%  |  "
            f"Cruising {t_cruise / total * 100:.1f}%  |  "
            f"Decelerating {t_decel / total * 100:.1f}%",
            ha="center",
            fontsize=12,
            style="italic",
        )

    plt.tight_layout()
    plt.show()
    return accel


def plot_power_vs_speed(df: pd.DataFrame) -> None:
    """Plot power-speed scatter and efficiency-speed relationship."""
    df = df.copy()
    df["Power_W"] = df["Voltage_V"] * df["Current_A"]

    mask = df["Speed_kph"] > 1
    spd_mov = df.loc[mask, "Speed_kph"]
    pwr_mov = df.loc[mask, "Power_W"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.scatter(spd_mov, pwr_mov, s=1, alpha=0.15, color="steelblue", rasterized=True)

    speed_bins = np.arange(0, spd_mov.max() + 2, 2)
    bin_idx = np.digitize(spd_mov, speed_bins) - 1
    bin_means = pd.DataFrame({"spd": spd_mov, "pwr": pwr_mov, "bin": bin_idx}).groupby("bin").agg(
        spd_mean=("spd", "mean"),
        pwr_mean=("pwr", "mean"),
        pwr_std=("pwr", "std"),
        count=("pwr", "count"),
    )
    bin_means = bin_means[bin_means["count"] >= 10]

    ax.plot(bin_means["spd_mean"], bin_means["pwr_mean"], "o-", color="red", linewidth=2, markersize=5, label="Bin average")
    ax.fill_between(
        bin_means["spd_mean"],
        bin_means["pwr_mean"] - bin_means["pwr_std"],
        bin_means["pwr_mean"] + bin_means["pwr_std"],
        color="red",
        alpha=0.12,
        label="+-1 std",
    )
    ax.set_xlabel("Speed (km/h)")
    ax.set_ylabel("Power (W)")
    ax.set_title("Power vs Speed")
    ax.legend()
    ax.grid(True, alpha=0.3)

    whpkm = pwr_mov / spd_mov
    ax2 = axes[1]
    ax2.scatter(spd_mov, whpkm, s=1, alpha=0.15, color="steelblue", rasterized=True)

    bin_eff = pd.DataFrame({"spd": spd_mov, "eff": whpkm, "bin": bin_idx}).groupby("bin").agg(
        spd_mean=("spd", "mean"),
        eff_mean=("eff", "mean"),
        count=("eff", "count"),
    )
    bin_eff = bin_eff[bin_eff["count"] >= 10]
    ax2.plot(bin_eff["spd_mean"], bin_eff["eff_mean"], "o-", color="red", linewidth=2, markersize=5, label="Bin average")

    if len(bin_eff) > 0:
        best = bin_eff.loc[bin_eff["eff_mean"].idxmin()]
        ax2.axvline(best["spd_mean"], color="green", linestyle="--", alpha=0.7)
        ax2.annotate(
            f"  Best: {best['spd_mean']:.0f} km/h\n  ({best['eff_mean']:.1f} Wh/km)",
            xy=(best["spd_mean"], best["eff_mean"]),
            fontsize=10,
            fontweight="bold",
            color="green",
        )

    ax2.set_xlabel("Speed (km/h)")
    ax2.set_ylabel("Energy use (Wh/km)")
    ax2.set_title("Efficiency vs Speed - lower is better")
    ax2.set_ylim(0, np.percentile(whpkm, 95) * 1.3)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_speed_variance_per_lap(
    df: pd.DataFrame,
    track_len_m: float,
) -> pd.DataFrame:
    """Plot lap overlays and return lap statistics dataframe."""
    df = df.copy()
    df["Lap"] = (df["Distance_m"] / track_len_m).astype(int)
    completed_laps = [lap for lap in df["Lap"].unique() if lap < df["Lap"].max()]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    colors = plt.cm.tab10.colors
    stats_rows: list[dict[str, float]] = []

    for j, lap_num in enumerate(completed_laps):
        lap_df = df[df["Lap"] == lap_num]
        pos = lap_df["TrackPos"]
        spd = lap_df["Speed_kph"]
        c = colors[j % len(colors)]
        ax.plot(pos, spd, alpha=0.5, linewidth=0.8, color=c, label=f"Lap {lap_num + 1}")

        moving = spd[spd > 1]
        stats_rows.append(
            {
                "Lap": lap_num + 1,
                "Avg (kph)": float(moving.mean()),
                "Std (kph)": float(moving.std()),
                "CV (%)": float((moving.std() / moving.mean() * 100) if moving.mean() > 0 else 0),
                "Min (kph)": float(moving.min()) if len(moving) > 0 else 0,
                "Max (kph)": float(moving.max()) if len(moving) > 0 else 0,
            }
        )

    if stats_rows:
        avg_all = np.mean([row["Avg (kph)"] for row in stats_rows])
        ax.axhline(avg_all, color="black", linestyle="--", linewidth=1.5, alpha=0.6, label=f"Ideal constant {avg_all:.1f} kph")

    ax.set_xlabel("Track Position (m)")
    ax.set_ylabel("Speed (km/h)")
    ax.set_title("Speed vs Track Position - Lap Overlay")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    if stats_rows:
        lap_labels = [f"Lap {row['Lap']}" for row in stats_rows]
        cvs = [row["CV (%)"] for row in stats_rows]
        bar_colors = [colors[j % len(colors)] for j in range(len(stats_rows))]
        ax2.bar(lap_labels, cvs, color=bar_colors, alpha=0.8, edgecolor="white")
        ax2.set_ylabel("Speed CV (%)")
        ax2.set_title("Speed Consistency\n(lower = more efficient)")
        ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.show()

    stats_df = pd.DataFrame(stats_rows).round(1)
    if not stats_df.empty:
        print("Lap-by-lap speed stats (while moving):")
        print(stats_df.to_string(index=False))
        print("\nA lower CV% means steadier speed -> less energy wasted on drag variation.")
    return stats_df


def compute_binned_track_metrics(
    df: pd.DataFrame,
    accel: pd.Series,
    track_len_m: float,
    n_bins: int = 500,
    smooth_window: int = 15,
) -> dict[str, np.ndarray | float]:
    """Compute binned telemetry arrays over track distance."""
    bin_edges = np.linspace(0, track_len_m, n_bins + 1)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2

    def bin_on_track(positions: np.ndarray, values: np.ndarray) -> np.ndarray:
        bin_idx = np.clip(np.digitize(positions, bin_edges) - 1, 0, n_bins - 1)
        binned = pd.Series(values).groupby(bin_idx).mean().reindex(range(n_bins))
        binned = binned.interpolate().bfill().ffill()
        return binned.rolling(smooth_window, center=True, min_periods=1).mean().values

    def bin_std_on_track(positions: np.ndarray, values: np.ndarray) -> np.ndarray:
        bin_idx = np.clip(np.digitize(positions, bin_edges) - 1, 0, n_bins - 1)
        binned = pd.Series(values).groupby(bin_idx).std().reindex(range(n_bins))
        binned = binned.interpolate().bfill().ffill()
        return binned.rolling(smooth_window, center=True, min_periods=1).mean().values

    positions = df["TrackPos"].values
    v_bin = bin_on_track(positions, df["Voltage_V"].values)
    c_bin = bin_on_track(positions, df["Current_A"].values)
    s_bin = bin_on_track(positions, df["Speed_kph"].values)
    a_bin = bin_on_track(positions, accel.values)
    sv_bin = bin_std_on_track(positions, df["Speed_kph"].values)

    return {
        "n_bins": float(n_bins),
        "bin_edges": bin_edges,
        "bin_mids": bin_mids,
        "v_bin": v_bin,
        "c_bin": c_bin,
        "s_bin": s_bin,
        "a_bin": a_bin,
        "sv_bin": sv_bin,
    }


def plot_three_panel_track_map(
    binned: dict[str, np.ndarray | float],
    dist_to_xy,
    wheel_diam_m: float,
    encoder_ticks_per_rev: int,
    dist_per_tick_m: float,
    track_len_m: float,
    n_laps: float,
    output_path: str = "ims_telemetry_map.png",
    turn_info: list[tuple[str, int]] | None = None,
) -> None:
    """Render and save three telemetry overlays on the track map."""
    if turn_info is None:
        turn_info = [
            ("T1", 700),
            ("T2", 950),
            ("T3", 1100),
            ("T4", 1250),
            ("T5", 1400),
            ("T6", 1550),
            ("T7", 1700),
            ("T8", 2650),
            ("T9", 2850),
            ("T10", 3000),
            ("T11", 3100),
            ("T12", 3250),
            ("T13", 3400),
            ("T14", 3550),
        ]

    bin_mids = binned["bin_mids"]
    v_bin = binned["v_bin"]
    c_bin = binned["c_bin"]
    s_bin = binned["s_bin"]

    bin_xy = np.array([dist_to_xy(d) for d in bin_mids])
    points = bin_xy.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(27, 9))
    panels = [
        (v_bin, "Voltage", "V", "RdYlGn"),
        (c_bin, "Current Draw", "A", "YlOrRd"),
        (s_bin, "Car Speed", "km/h", "cool"),
    ]

    for ax, (vals, title, unit, cmap) in zip(axes, panels):
        norm = Normalize(vmin=np.nanmin(vals), vmax=np.nanmax(vals))
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=6, capstyle="round")
        lc.set_array(vals[:-1])
        ax.add_collection(lc)

        cbar = fig.colorbar(lc, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label(unit, fontsize=11)

        sf = dist_to_xy(0)
        ax.plot(*sf, "k^", ms=14, zorder=5)
        ax.annotate("S/F", sf, textcoords="offset points", xytext=(8, 8), fontsize=9, fontweight="bold")

        a1, a2 = dist_to_xy(150), dist_to_xy(250)
        ax.annotate("", xy=a2, xytext=a1, arrowprops=dict(arrowstyle="->", color="black", lw=2.5))

        for lbl, d in turn_info:
            txy = dist_to_xy(d)
            ax.annotate(
                lbl,
                txy,
                fontsize=7,
                ha="center",
                va="center",
                color="dimgray",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="silver", alpha=0.8),
            )

        ax.set_aspect("equal")
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_xlabel("East - West (m)")
        ax.set_ylabel("North - South (m)")
        ax.grid(True, alpha=0.15)
        ax.autoscale()
        ax.margins(0.08)

    fig.suptitle(
        "IMS Grand Prix Road Course  -  Telemetry Map\n"
        f"Wheel dia {wheel_diam_m} m  |  {encoder_ticks_per_rev} encoder ticks/rev  |  "
        f"{dist_per_tick_m:.4f} m/tick  |  Track {track_len_m / 1000:.3f} km  |  "
        f"{n_laps:.1f} laps averaged",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {output_path}")
