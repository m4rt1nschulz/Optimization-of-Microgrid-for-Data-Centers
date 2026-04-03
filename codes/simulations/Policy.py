import vessim as vs
from datetime import datetime, timedelta
import pandas as pd

# --- THE SINGLE-OBJECTIVE POLICY (Focus 100% on Cost/Rate) ---
class HeuresCreusesPolicy(vs.Policy):
    def __init__(self, sim_start):
        # We synchronize the internal clock of the policy
        self.current_time = pd.to_datetime(sim_start)

    def apply(self, p_delta, duration, storage=None):
        # We advance the time
        hour = self.current_time.hour
        self.current_time += timedelta(seconds=duration)

        if storage:
            # Logic of "Heures Creuses" (Off-peak Hours in France: 22:00 to 06:00)
            is_heures_creuses = (hour >= 22 or hour < 6)

            if is_heures_creuses and storage.soc() < 0.9:
                # It's late at night and energy is cheap! 
                # THE ONLY objective is to fill the battery as fast as possible (e.g., 2500W from the grid).
                p_batt_charge = storage.update(power=2500, duration=duration) / duration
                return p_delta - p_batt_charge
            else:
                # Heures Pleines (Expensive energy): Try not to buy anything from the grid!
                # Uses the sun, wind, and battery to cover the server.
                p_batt_discharge = storage.update(power=p_delta, duration=duration) / duration
                return p_delta - p_batt_discharge
        
        return p_delta

def sim_mono_objective_policy():
    sim_start = "2022-06-15 00:00:00"
    env = vs.Environment(sim_start=sim_start, step_size=300)

    # 1. ACTORS: Server, Solar and Wind
    datacenter = vs.Actor(name="Server", signal=vs.StaticSignal(value=-1500))
    solar = vs.Actor(name="Solar", signal=vs.Trace.load(
        dataset="solcast2022_global", column="Berlin", params={"scale": 4000}
    ))
    
    try:
        df_wind = pd.read_csv("../../data/vessim_results/vessim_wind_gif.csv", index_col=0, parse_dates=True)
        df_wind.index = df_wind.index + (pd.to_datetime(sim_start) - df_wind.index[0])
        wind = vs.Actor(name="Wind", signal=vs.Trace(actual=df_wind["power_W"]))
    except FileNotFoundError:
        wind = vs.Actor(name="Wind", signal=vs.StaticSignal(value=400))

    # 2. BATTERY
    battery = vs.SimpleBattery(capacity=10000, initial_soc=0.2, min_soc=0.1)

    # 3. APPLYING THE SINGLE-OBJECTIVE POLICY
    policy = HeuresCreusesPolicy(sim_start=sim_start)

    # 4. CREATING THE MICROGRID
    env.add_microgrid(
        name="Mono_Objective_Cost_Policy",
        actors=[datacenter, solar, wind],
        storage=battery,
        policy=policy
    )

    logger = vs.MemoryLogger()
    env.add_controller(logger)

    print("Running Single-Objective simulation (Cost Policy - Off-peak Hours)...")
    env.run(until=24 * 3600)

    # 5. VISUALIZATION
    fig = vs.plot_result_df(logger.to_df())
    fig.show()

if __name__ == "__main__":
    sim_mono_objective_policy()