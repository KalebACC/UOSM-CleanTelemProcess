import numpy as np

def get_efficiency(num_laps: int, energy_joules: float, track_len_km:float) -> None:
    """gets the efficiency of the car in km/kWh

    Args:
        num_laps (int): _description_
        energy_joules (float): _description_
        track_len_km (float): _description_
    """
    # convert joules to kWh
    energy_kwh = energy_joules / 3_600_000
    
    # compute efficiency (km per kWh)
    efficiency = (num_laps * track_len_km) / energy_kwh
    
    print(f"Efficiency: {efficiency:.2f} km/kWh")

def get_power(data):
    data['Power'] = (data['Current'] / 1e3) * (data['Voltage']/1e3)

def get_total_energy(data):
    """Prints total energy used by the car

    Args:
        data (df): dataframe holding the track csv results
    """
    if 'Power' not in data.columns:
        get_power(data=data)
    
    energy_joules = np.trapezoid(data['Power'], x=data['Tick'] / 1e3)
    energy_kWh = energy_joules / 3600000  # Convert from J to kWh
    print(f"\nTotal energy consumed: {energy_joules} J or {energy_kWh} kWh")