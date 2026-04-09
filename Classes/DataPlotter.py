import pandas as pd
import matplotlib.pyplot as plt

MS_PER_MINUTE = 60_000


def ms_to_minutes(ms):
    """Convert milliseconds to minutes."""
    return ms / MS_PER_MINUTE

def speed_to_kph(csv_speed):
    """Convert raw CSV speed to km/h."""
    return csv_speed * 0.001

def speed_to_mps(csv_speed):
    """Convert raw CSV speed to m/s."""
    return speed_to_kph(csv_speed) / 3.6


def calculate_distance_km(data: pd.DataFrame) -> pd.Series:
    """Calculate cumulative distance in kilometers from telemetry speed and time."""
    time_s = (data["Tick"] - data["Tick"].iloc[0]) / 1000.0
    speed_mps = speed_to_mps(data["Speed"])

    dt = time_s.diff().fillna(0)
    avg_speed_mps = (speed_mps + speed_mps.shift(1)).fillna(0) / 2
    return (avg_speed_mps * dt).cumsum() / 1000

class DataPlotter:
    """Read telemetry CSV data and render plots, with optional smoothing, speed conversion, and time filtering."""

    #TODO TRUNCATE THE DF TO WHEN THE RUN OFFICIALY STARTS

    def __init__(self, file_path: str) -> None:
        self.file_path: str = file_path
        self.data: pd.DataFrame | None = None
        self.load_data()

    def load_data(self, convert_speed: bool = True) -> None:
        """Load CSV, prepare TimePlot in minutes, and optionally convert speed."""
        self.data = pd.read_csv(self.file_path)
        expected_cols = ["Tick", "Throttle", "Speed", "Current", "Voltage"]
        for col in expected_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing column: {col}")

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