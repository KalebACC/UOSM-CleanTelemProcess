import pandas as pd
import matplotlib.pyplot as plt

MS_PER_MINUTE = 60_000
DEFAULT_START_SPEED_KPH = 8.0
DEFAULT_START_CURRENT_A = 5.0
DEFAULT_START_THROTTLE = 1.0
DEFAULT_END_CURRENT_A = 1.0
DEFAULT_END_THROTTLE = 1.0
DEFAULT_MIN_ACTIVE_SECONDS = 1.0


def ms_to_minutes(ms):
    """Convert milliseconds to minutes."""
    return ms / MS_PER_MINUTE

def speed_to_kph(csv_speed):
    """Convert raw CSV speed to km/h."""
    return csv_speed * 0.001

def speed_to_mps(csv_speed):
    """Convert raw CSV speed to m/s."""
    return speed_to_kph(csv_speed) / 3.6


def _seconds_to_samples(seconds: float, tick_ms: pd.Series) -> int:
    """Convert seconds to an estimated number of telemetry samples."""
    dt = tick_ms.diff().dropna()
    dt = dt[dt > 0]
    if dt.empty:
        return 1

    median_dt_seconds = float(dt.median()) / 1000.0
    if median_dt_seconds <= 0:
        return 1
    return max(1, int(round(seconds / median_dt_seconds)))


def detect_official_run_bounds(
    data: pd.DataFrame,
    start_speed_kph: float = DEFAULT_START_SPEED_KPH,
    start_current_a: float = DEFAULT_START_CURRENT_A,
    start_throttle: float = DEFAULT_START_THROTTLE,
    end_current_a: float = DEFAULT_END_CURRENT_A,
    end_throttle: float = DEFAULT_END_THROTTLE,
    min_active_seconds: float = DEFAULT_MIN_ACTIVE_SECONDS,
) -> tuple[int, int] | None:
    """Detect the official run start/end rows in telemetry data.

    The default logic is intentionally conservative:
    - Start requires sustained activity (throttle or loaded movement).
    - End is the last point where the car still appears under power.
    This helps ignore push-on/push-off periods where speed may be non-zero
    while throttle/current are low.
    """
    required = ["Tick", "Throttle", "Speed", "Current"]
    for col in required:
        if col not in data.columns:
            raise ValueError(f"Missing column: {col}")

    tick = pd.to_numeric(data["Tick"], errors="coerce").fillna(0)
    throttle = pd.to_numeric(data["Throttle"], errors="coerce").fillna(0).abs()
    speed_kph = pd.to_numeric(data["Speed"], errors="coerce").fillna(0).apply(speed_to_kph)
    current_a = pd.to_numeric(data["Current"], errors="coerce").fillna(0).abs() / 1000.0

    # Start is either throttle engagement or sustained speed under meaningful load.
    start_signal = (throttle >= start_throttle) | (
        (speed_kph >= start_speed_kph) & (current_a >= start_current_a)
    )
    if not start_signal.any():
        return None

    start_window = _seconds_to_samples(min_active_seconds, tick)
    sustained_start = start_signal.rolling(window=start_window, min_periods=start_window).sum() >= start_window

    if sustained_start.any():
        first_confirmed = int(sustained_start[sustained_start].index[0])
        start_idx = max(0, first_confirmed - start_window + 1)
    else:
        start_idx = int(start_signal[start_signal].index[0])

    # End is the last sample where the car still appears under driver/motor control.
    end_signal = (throttle >= end_throttle) | (current_a >= end_current_a)
    end_candidates = end_signal.iloc[start_idx:]
    if end_candidates.any():
        end_idx = int(end_candidates[end_candidates].index[-1])
    else:
        end_idx = int(start_idx)

    if end_idx < start_idx:
        return None

    return start_idx, end_idx


def calculate_distance_km(data: pd.DataFrame) -> pd.Series:
    """Calculate cumulative distance in kilometers from telemetry speed and time."""
    time_s = (data["Tick"] - data["Tick"].iloc[0]) / 1000.0
    speed_mps = speed_to_mps(data["Speed"])

    dt = time_s.diff().fillna(0)
    avg_speed_mps = (speed_mps + speed_mps.shift(1)).fillna(0) / 2
    return (avg_speed_mps * dt).cumsum() / 1000

class DataPlotter:
    """Read telemetry CSV data and render plots, with optional smoothing, speed conversion, and time filtering."""

    def __init__(
        self,
        file_path: str,
        auto_trim_to_official_run: bool = True,
        run_detection_options: dict | None = None,
    ) -> None:
        self.file_path: str = file_path
        self.data: pd.DataFrame | None = None
        self.auto_trim_to_official_run: bool = auto_trim_to_official_run
        self.run_detection_options: dict = run_detection_options or {}
        self.official_run_bounds: tuple[int, int] | None = None
        self.official_run_time_window_min: tuple[float, float] | None = None
        self.load_data()

    def load_data(self, convert_speed: bool = True) -> None:
        """Load CSV, prepare TimePlot in minutes, and optionally convert speed."""
        self.data = pd.read_csv(self.file_path)
        expected_cols = ["Tick", "Throttle", "Speed", "Current", "Voltage"]
        for col in expected_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing column: {col}")

        # Normalize core telemetry columns to numeric values.
        for col in expected_cols:
            self.data[col] = pd.to_numeric(self.data[col], errors="coerce")
        self.data = self.data.dropna(subset=expected_cols).reset_index(drop=True)

        capture_start_tick = int(self.data["Tick"].iloc[0])
        self.official_run_bounds = None
        self.official_run_time_window_min = None

        if self.auto_trim_to_official_run:
            bounds = detect_official_run_bounds(self.data, **self.run_detection_options)
            if bounds is not None:
                start_idx, end_idx = bounds
                start_tick = int(self.data["Tick"].iloc[start_idx])
                end_tick = int(self.data["Tick"].iloc[end_idx])
                self.official_run_bounds = (start_idx, end_idx)
                self.official_run_time_window_min = (
                    ms_to_minutes(start_tick - capture_start_tick),
                    ms_to_minutes(end_tick - capture_start_tick),
                )
                self.data = self.data.iloc[start_idx : end_idx + 1].reset_index(drop=True)

        # Convert Tick to TimePlot in minutes starting at 0
        self.data["Tick"] = self.data["Tick"].astype(int)
        self.data["Tick"] -= self.data["Tick"].iloc[0]
        self.data["TimePlot"] = ms_to_minutes(self.data["Tick"])

        # Convert Speed column if requested
        if convert_speed:
            self.data["Speed_kph"] = self.data["Speed"].apply(speed_to_kph)
            self.data["Speed_mps"] = self.data["Speed"].apply(speed_to_mps)

        self.data["Distance_km"] = calculate_distance_km(self.data)

    def plot_single(self, column: str, smooth_window: int = 0, start_time: float | None = None, end_time: float | None = None,speed_unit: str = "kph") -> None:
        """Plot a single column with optional smoothing, minute range filter, and speed unit conversion."""
        if self.data is None:
            raise ValueError("Load data first")

        # Map speed column if needed
        if column.lower() == "speed":
            col_map = {"raw": "Speed", "kph": "Speed_kph", "mps": "Speed_mps"}
            if speed_unit not in col_map:
                raise ValueError(f"Invalid speed_unit: {speed_unit}")
            column = col_map[speed_unit]

        if column not in self.data.columns:
            raise ValueError(f"{column} not found")

        df = self.data
        if start_time is not None:
            df = df[df["TimePlot"] >= start_time]
        if end_time is not None:
            df = df[df["TimePlot"] <= end_time]

        x = df["TimePlot"]
        y = df[column]

        ylabel_map = {
            "Speed": "Speed",
            "Speed_kph": "Speed (kph)",
            "Speed_mps": "Speed (m/s)",
            "Distance_km": "Distance (km)",
        }
        ylabel = ylabel_map.get(column, column)

        plt.figure(figsize=(10, 5))
        plt.plot(x, y, label="Raw", alpha=0.7)
        if smooth_window > 1:
            y_smooth = y.rolling(window=smooth_window, min_periods=1, center=True).mean()
            plt.plot(x, y_smooth, label=f"Smoothed ({smooth_window})", linewidth=2)

        plt.xlabel("Time (min)")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs Time")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_all(self,smooth_window: int = 0,start_time: float | None = None,end_time: float | None = None, speed_unit: str = "kph") -> None:
        """Plot all columns with optional smoothing, speed unit conversion, and minute filter."""
        if self.data is None:
            raise ValueError("Load data first")
        self.plot_single("Throttle",smooth_window,start_time,end_time,speed_unit)
        self.plot_single("Speed",smooth_window,start_time,end_time,speed_unit)
        self.plot_single("Distance_km",smooth_window,start_time,end_time,speed_unit)
        self.plot_single("Current",smooth_window,start_time,end_time,speed_unit)
        self.plot_single("Voltage",smooth_window,start_time,end_time,speed_unit)

    def plot_compare(self,file: str,column: str,smooth_window: int = 0,start_time: float | None = None, end_time: float | None = None, speed_unit: str = "kph",) -> None:
        """Compare a column from this plotter against the same column from another CSV file."""
        if self.data is None:
            raise ValueError("Load data first")

        # Load the comparison file into a temporary DataPlotter
        other = DataPlotter(file)
        other.load_data(convert_speed=True)

        # Resolve speed column alias for both datasets
        col = column
        if column.lower() == "speed":
            col_map = {"raw": "Speed", "kph": "Speed_kph", "mps": "Speed_mps"}
            if speed_unit not in col_map:
                raise ValueError(f"Invalid speed_unit: {speed_unit}")
            col = col_map[speed_unit]

        for label, df in [("Self", self.data), ("Other", other.data)]:
            if col not in df.columns:
                raise ValueError(f"{col} not found in {label} dataset")

        def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
            if start_time is not None:
                df = df[df["TimePlot"] >= start_time]
            if end_time is not None:
                df = df[df["TimePlot"] <= end_time]
            return df

        self_df = apply_filters(self.data)
        other_df = apply_filters(other.data)

        self_label = self.file_path.split("/")[-1]
        other_label = file.split("/")[-1]

        plt.figure(figsize=(10, 5))

        for df, label in [(self_df, self_label), (other_df, other_label)]:
            x = df["TimePlot"]
            y = df[col]
            line, = plt.plot(x, y, alpha=0.4, label=f"{label} Raw")
            if smooth_window > 1:
                y_smooth = y.rolling(window=smooth_window, min_periods=1, center=True).mean()
                plt.plot(x, y_smooth, color=line.get_color(), linewidth=2, label=f"{label} Smoothed ({smooth_window})")

        plt.xlabel("Time (min)")
        plt.ylabel(col)
        plt.title(f"{col} vs Time — Comparison")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_two(self,column1: str,column2: str,smooth_window: int = 0,start_time: float | None = None,end_time: float | None = None) -> None:
        """Plot two columns on the same graph."""

        if self.data is None:
            raise ValueError("Load data first")

        for col in [column1, column2]:
            if col not in self.data.columns:
                raise ValueError(f"{col} not found")

        df = self.data
        if start_time is not None:
            df = df[df["TimePlot"] >= start_time]
        if end_time is not None:
            df = df[df["TimePlot"] <= end_time]

        x = df["TimePlot"]

        plt.figure(figsize=(10, 5))

        for col in [column1, column2]:
            y = df[col]
            plt.plot(x, y, label=f"{col} Raw", alpha=0.7)

            if smooth_window > 1:
                y_s = y.rolling(window=smooth_window, min_periods=1, center=True).mean()
                plt.plot(x, y_s, linewidth=2, label=f"{col} Smoothed")

        plt.xlabel("Time (min)")
        plt.ylabel("Values")
        plt.title(f"{column1} vs {column2}")
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_official_run_window_minutes(self) -> tuple[float, float] | None:
        """Return detected official run start/end times in capture-relative minutes."""
        return self.official_run_time_window_min

    def give_averages(self) -> None:
        """Print averages for all numeric columns."""
        if self.data is None:
            raise ValueError("Load data first")

        for col in self.data.columns:
            avg_val = self.data[col].mean()
            print(f"{col} average is {avg_val:.2f}")
            
    def plot_compare_all(self,file: str,smooth_window: int = 0,start_time: float | None = None, end_time: float | None = None, speed_unit: str = "kph",) -> None:
        self.plot_compare(file=file,column="Throttle",smooth_window=smooth_window,start_time=start_time,end_time=end_time,speed_unit=speed_unit)
        self.plot_compare(file=file,column="Speed",smooth_window=smooth_window,start_time=start_time,end_time=end_time,speed_unit=speed_unit)
        self.plot_compare(file=file,column="Current",smooth_window=smooth_window,start_time=start_time,end_time=end_time,speed_unit=speed_unit)
        self.plot_compare(file=file,column="Voltage",smooth_window=smooth_window,start_time=start_time,end_time=end_time,speed_unit=speed_unit)