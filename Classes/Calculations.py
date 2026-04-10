import numpy as np
import pandas as pd


def get_efficiency(distance_km: float, energy_joules: float) -> float:
    """Get the efficiency of the car in km/kWh.

    Args:
        distance_km (float): Total distance covered in kilometers.
        energy_joules (float): Total energy consumed in joules.

    Returns:
        float: Efficiency in km/kWh.
    """
    energy_kwh = energy_joules / 3_600_000
    if energy_kwh == 0:
        raise ValueError("Energy must be greater than 0 to calculate efficiency")

    efficiency = distance_km / energy_kwh
    print(f"Efficiency: {efficiency:.2f} km/kWh")
    return efficiency


def get_power(data: pd.DataFrame) -> None:
    """Returns power used in a run

    Args:
        data (pd.DataFrame): dataframe holding the track csv results
    """
    data['Power'] = (data['Current'] / 1e3) * (data['Voltage'] / 1e3)


def get_total_energy(data: pd.DataFrame) -> float:
    if 'Power' not in data.columns:
        get_power(data=data)

    energy_joules = np.trapezoid(data['Power'], x=data['Tick'] / 1e3)
    energy_kWh = energy_joules / 3600000

    print(f"\nTotal energy consumed: {energy_joules} J or {energy_kWh} kWh")
    return energy_joules


def get_distance_from_speed(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate cumulative distance from speed and tick data in kilometers."""
    if "Tick" not in data.columns or "Speed" not in data.columns:
        raise ValueError("Data must contain Tick and Speed columns")

    result = data.copy()
    time_s = (result["Tick"] - result["Tick"].iloc[0]) / 1000.0
    speed_mps = (result["Speed"] * 0.001) / 3.6

    dt = time_s.diff().fillna(0)
    avg_speed_mps = (speed_mps + speed_mps.shift(1)).fillna(0) / 2

    # avg_speed_mps * dt gives meters, so divide by 1000 to store kilometers.
    result["Distance_km"] = (avg_speed_mps * dt).cumsum() / 1000

    return result
