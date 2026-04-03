import vessim as vs
import pandas as pd

def sim_no_battery_with_renewables():
    sim_start = "2022-06-15 00:00:00"
    env = vs.Environment(sim_start=sim_start, step_size=300)

    # 1. CONSUMER (Data Center)
    datacenter = vs.Actor(name="DataCenter", signal=vs.StaticSignal(value=-1500))

    # 2. PRODUCERS (Solar and Wind)
    solar = vs.Actor(name="Solar", signal=vs.Trace.load(
        dataset="solcast2022_global", column="Berlin", params={"scale": 4000}
    ))

    # Gif-sur-Yvette wind
    try:
        df_wind = pd.read_csv("../../data/vessim_results/vessim_wind_gif.csv", index_col=0, parse_dates=True)
        df_wind.index = df_wind.index + (pd.to_datetime(sim_start) - df_wind.index[0])
        wind = vs.Actor(name="Wind", signal=vs.Trace(actual=df_wind["power_W"]))
    except FileNotFoundError:
        wind = vs.Actor(name="Wind", signal=vs.StaticSignal(value=400))

    # 3. BATTERY (Removed for this scenario)

    # 4. POLICY (Using default grid-connected policy)
    policy = vs.DefaultPolicy(mode="grid-connected")

    # 5. CREATING MICROGRID WITHOUT BATTERY
    env.add_microgrid(
        name="No_Battery_Renewables_Grid",
        actors=[datacenter, solar, wind],
        # storage=battery  <-- REMOVED for this scenario
        policy=policy
    )

    logger = vs.MemoryLogger()
    env.add_controller(logger)

    print("Running simulation: 100% Off-Grid (Solar + Wind + Grid Only)...")
    env.run(until=24 * 3600)

    # 6. VISUALIZATION
    fig = vs.plot_result_df(logger.to_df())
    fig.show()

if __name__ == "__main__":
    sim_no_battery_with_renewables()