"""Interactive Plotly telemetry map helpers."""

from __future__ import annotations

import ipywidgets as widgets
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import HTML, display


def build_interactive_track_map(
    df: pd.DataFrame,
    track_len_m: float,
    bin_mids: np.ndarray,
    s_bin: np.ndarray,
    c_bin: np.ndarray,
    v_bin: np.ndarray,
    a_bin: np.ndarray,
    sv_bin: np.ndarray,
    gps_track_csv: str = "sem-us-2022-track_coordinates.csv",
    enable_widgets: bool = False,
) -> None:
    """Display telemetry map.

    By default this renders a static HTML plot to avoid widget-related
    JavaScript issues in some notebook environments. Set enable_widgets=True
    to use the slider/playback widget view.
    """
    track_gps = pd.read_csv(gps_track_csv)
    track_lat = track_gps["Latitude"].values
    track_lon = track_gps["Longitude"].values

    r_earth = 6_371_000
    dlat = np.radians(np.diff(track_lat))
    dlon = np.radians(np.diff(track_lon))
    hav = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(track_lat[:-1]))
        * np.cos(np.radians(track_lat[1:]))
        * np.sin(dlon / 2) ** 2
    )
    track_arc = np.zeros(len(track_lat))
    track_arc[1:] = np.cumsum(r_earth * 2 * np.arctan2(np.sqrt(hav), np.sqrt(1 - hav)))
    gps_len = float(track_arc[-1])

    print(f"GPS track: {len(track_lat)} pts, {gps_len:.0f} m")

    gps_d = (df["TrackPos"].values / track_len_m) * gps_len
    gps_idx = np.clip(np.searchsorted(track_arc, gps_d), 0, len(track_lat) - 1)
    car_lat = track_lat[gps_idx]
    car_lon = track_lon[gps_idx]

    bin_gps_d = (bin_mids / track_len_m) * gps_len
    bin_gps_idx = np.clip(np.searchsorted(track_arc, bin_gps_d), 0, len(track_lat) - 1)
    ov_lat = track_lat[bin_gps_idx]
    ov_lon = track_lon[bin_gps_idx]

    print(f"Mapped {len(car_lat)} telemetry samples to GPS coords")

    use_widget = enable_widgets
    if use_widget:
        # FigureWidget requires anywidget in recent plotly versions.
        # Fall back to a normal Figure to avoid hard failures in notebooks where
        # anywidget is not installed or where widget front-end support is missing.
        try:
            fig = go.FigureWidget()
        except ImportError:
            use_widget = False
            fig = go.Figure()
            print("FigureWidget is unavailable (install anywidget for sliders). Showing static map instead.")
    else:
        fig = go.Figure()
    fig.add_trace(
        go.Scattermap(
            lat=track_lat,
            lon=track_lon,
            mode="lines",
            line=dict(color="#444", width=6),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scattermap(
            lat=ov_lat,
            lon=ov_lon,
            mode="markers",
            marker=dict(
                size=6,
                color=s_bin,
                colorscale="Viridis",
                cmin=0,
                cmax=float(df["Speed_kph"].max()),
                showscale=True,
                colorbar=dict(title=dict(text="km/h"), x=1.0, len=0.5, y=0.75),
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scattermap(
            lat=[float(car_lat[0])],
            lon=[float(car_lon[0])],
            mode="markers",
            marker=dict(size=18, color="red"),
            showlegend=False,
        )
    )

    fig.update_layout(
        map=dict(
            style="open-street-map",
            center=dict(lat=float(track_lat.mean()), lon=float(track_lon.mean())),
            zoom=14.5,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
    )

    if not use_widget:
        if enable_widgets:
            print("Widget mode unavailable in this environment. Rendering static HTML map.")
        display(HTML(fig.to_html(full_html=False, include_plotlyjs="cdn")))
        return

    slider = widgets.IntSlider(
        value=0,
        min=0,
        max=len(df) - 1,
        step=1,
        description="Sample:",
        continuous_update=True,
        layout=widgets.Layout(width="85%"),
        style={"description_width": "55px"},
    )
    play_btn = widgets.Play(value=0, min=0, max=len(df) - 1, step=20, interval=30, description="Play")
    widgets.jslink((play_btn, "value"), (slider, "value"))

    metric_dd = widgets.Dropdown(
        options=["Speed", "Current", "Voltage", "Acceleration", "Speed Variance"],
        value="Speed",
        description="Overlay:",
        style={"description_width": "55px"},
        layout=widgets.Layout(width="180px"),
    )
    info_html = widgets.HTML()

    metric_cfg = {
        "Speed": (s_bin, "Viridis", 0, float(df["Speed_kph"].max()), "km/h"),
        "Current": (c_bin, "YlOrRd", 0, float(df["Current_A"].max()), "A"),
        "Voltage": (v_bin, "RdYlGn", float(df["Voltage_V"].min()), float(df["Voltage_V"].max()), "V"),
        "Acceleration": (a_bin, "RdBu_r", float(np.nanmin(a_bin)), float(np.nanmax(a_bin)), "m/s^2"),
        "Speed Variance": (sv_bin, "hot_r", 0, float(np.nanmax(sv_bin)), "sigma km/h"),
    }

    def on_metric(change: dict) -> None:
        vals, colorscale, lo, hi, unit = metric_cfg[change["new"]]
        with fig.batch_update():
            fig.data[1].marker.color = vals
            fig.data[1].marker.colorscale = colorscale
            fig.data[1].marker.cmin = lo
            fig.data[1].marker.cmax = hi
            fig.data[1].marker.colorbar.title.text = unit

    metric_dd.observe(on_metric, names="value")

    def on_tick(change: dict) -> None:
        i = change["new"]
        with fig.batch_update():
            fig.data[2].lat = [float(car_lat[i])]
            fig.data[2].lon = [float(car_lon[i])]

        spd = df["Speed_kph"].iloc[i]
        volt = df["Voltage_V"].iloc[i]
        cur = df["Current_A"].iloc[i]
        pwr = volt * cur
        t_min = df["TimePlot"].iloc[i]
        dist = df["Distance_m"].iloc[i]
        lap = int(dist / track_len_m) + 1
        thr = df["Throttle"].iloc[i]

        info_html.value = (
            "<div style=\"font-family:'Courier New',monospace;font-size:15px;"
            "padding:12px;background:linear-gradient(135deg,#1a1a2e,#16213e);"
            "color:#eee;border-radius:10px;display:flex;gap:24px;flex-wrap:wrap;"
            "align-items:center;\">"
            f"<div><span style=\"color:#888\">TIME</span><br>"
            f"<b style=\"font-size:18px\">{t_min:.2f} min</b></div>"
            f"<div><span style=\"color:#888\">DISTANCE</span><br>"
            f"<b style=\"font-size:18px\">{dist:.0f} m</b></div>"
            f"<div><span style=\"color:#888\">LAP</span><br>"
            f"<b style=\"font-size:18px\">{lap}</b></div>"
            "<div style=\"border-left:1px solid #444;padding-left:20px\">"
            f"<span style=\"color:#4CAF50\">VOLTAGE</span><br>"
            f"<b style=\"font-size:18px\">{volt:.1f} V</b></div>"
            f"<div><span style=\"color:#FF9800\">CURRENT</span><br>"
            f"<b style=\"font-size:18px\">{cur:.1f} A</b></div>"
            f"<div><span style=\"color:#E91E63\">POWER</span><br>"
            f"<b style=\"font-size:18px\">{pwr:.1f} W</b></div>"
            "<div style=\"border-left:1px solid #444;padding-left:20px\">"
            f"<span style=\"color:#2196F3\">SPEED</span><br>"
            f"<b style=\"font-size:18px\">{spd:.1f} km/h</b></div>"
            f"<div><span style=\"color:#9C27B0\">THROTTLE</span><br>"
            f"<b style=\"font-size:18px\">{thr}</b></div>"
            "</div>"
        )

    slider.observe(on_tick, names="value")
    on_tick({"new": 0})

    controls = widgets.HBox([play_btn, slider, metric_dd], layout=widgets.Layout(align_items="center", gap="8px"))
    display(widgets.VBox([fig, controls, info_html]))
