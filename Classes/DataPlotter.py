import pandas as pd
import matplotlib.pyplot as plt

def speed_to_kph(csv_speed):
    """Convert raw CSV speed to km/h."""
    return csv_speed * 0.0011

def speed_to_mps(csv_speed):
    """Convert raw CSV speed to m/s."""
    return speed_to_kph(csv_speed) / 3.6

class DataPlotter:
    """Read telemetry CSV data and render plots, with optional smoothing, speed conversion, and time filtering."""

    def __init__(self, file_path: str) -> None:
        self.file_path: str = file_path
        self.data: pd.DataFrame | None = None

    def load_data(self, convert_speed: bool = True) -> None:
        """Load CSV, prepare TimePlot column, and optionally convert speed."""
        self.data = pd.read_csv(self.file_path)
        expected_cols = ["Tick", "Throttle", "Speed", "Current", "Voltage"]
        for col in expected_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing column: {col}")

        # Convert Tick to TimePlot in ms starting at 0
        self.data["Tick"] = self.data["Tick"].astype(int)
        self.data["Tick"] -= self.data["Tick"].iloc[0]
        self.data["TimePlot"] = self.data["Tick"]

        # Convert Speed column if requested
        if convert_speed:
            self.data["Speed_kph"] = self.data["Speed"].apply(speed_to_kph)
            self.data["Speed_mps"] = self.data["Speed"].apply(speed_to_mps)

    def plot_single(self, column: str, smooth_window: int = 0, start_time: int | None = None, end_time: int | None = None,speed_unit: str = "kph") -> None:
        """Plot a single column with optional smoothing, time range filter, and speed unit conversion."""
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

        plt.figure(figsize=(10, 5))
        plt.plot(x, y, label="Raw", alpha=0.7)
        if smooth_window > 1:
            y_smooth = y.rolling(window=smooth_window, min_periods=1, center=True).mean()
            plt.plot(x, y_smooth, label=f"Smoothed ({smooth_window})", linewidth=2)

        plt.xlabel("Time (ms)")
        plt.ylabel(column)
        plt.title(f"{column} vs Time")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_all(self,smooth_window: int = 0,start_time: int | None = None,end_time: int | None = None, speed_unit: str = "kph") -> None:
        """Plot all columns with optional smoothing, speed unit conversion, and time filter."""
        if self.data is None:
            raise ValueError("Load data first")
        self.plot_single("Throttle",smooth_window,start_time,end_time,speed_unit)
        self.plot_single("Speed",smooth_window,start_time,end_time,speed_unit)
        self.plot_single("Current",smooth_window,start_time,end_time,speed_unit)
        self.plot_single("Voltage",smooth_window,start_time,end_time,speed_unit)