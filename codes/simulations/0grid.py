import vessim as vs
import pandas as pd
import os

def sim_100_percent_clean():
    sim_start = "2022-06-15 00:00:00"
    env = vs.Environment(sim_start=sim_start, step_size=300)

    # 1. CONSUMER (Data Center)
    datacenter = vs.Actor(name="Server", signal=vs.StaticSignal(value=-1500))

    # 2. OVERSIZED GENERATORS (To cope with the lockdown)
    # We increased the solar panel capacity to 8000W to charge the giant battery quickly
    solar = vs.Actor(name="Solar", signal=vs.Trace.load(
        dataset="solcast2022_global", column="Berlin", params={"scale": 8000}
    ))

    # Gif-sur-Yvette wind
    try:
        df_wind = pd.read_csv("../../data/vessim_results/vessim_wind_gif.csv", index_col=0, parse_dates=True)
        df_wind.index = df_wind.index + (pd.to_datetime(sim_start) - df_wind.index[0])
        wind = vs.Actor(name="Wind", signal=vs.Trace(actual=df_wind["power_W"]))
    except FileNotFoundError:
        wind = vs.Actor(name="Wind", signal=vs.StaticSignal(value=1000))

    # 3. GIANT BATTERY (To survive the night and cloudy days)
    # Starting with 80% SOC to have some energy in the beginning, and a minimum SOC of 10% to avoid deep discharges that could damage the battery.
    battery = vs.SimpleBattery(capacity=25000, initial_soc=0.8, min_soc=0.1)

    # 4. POLICY (Islanded mode to forcelocal generation and storage)
    policy = vs.DefaultPolicy(mode="islanded")

    # 5. CREATE MICROGRID
    env.add_microgrid(
        name="100_Percent_Clean_Offgrid",
        actors=[datacenter, solar, wind],
        storage=battery,
        policy=policy
    )

    logger = vs.MemoryLogger()
    env.add_controller(logger)

    print("Running simulation 100% Clean Energy (Islanded Mode)...")
    try:
        env.run(until=24 * 3600)
        print("Success! The Data Center survived 24h with clean energy only.")
    except Exception as e:
        print(f"Blackout! The system failed: {e}")

    # 6. VISUALIZATION
    fig = vs.plot_result_df(logger.to_df())
    fig.show()

if __name__ == "__main__":
    sim_100_percent_clean()