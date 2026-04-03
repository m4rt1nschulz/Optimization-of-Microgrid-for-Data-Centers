import vessim as vs

def sim_mono_objective_solar():
    sim_start = "2022-06-15 00:00:00"
    env = vs.Environment(sim_start=sim_start, step_size=300)

    # 1. CONSUMER (Data Center)
    datacenter = vs.Actor(name="Server", signal=vs.StaticSignal(value=-1500))

    # 2. PRODUCTION (Solar)
    # Solar data from Solcast for Berlin, scaled to represent a 8 kW solar installation.
    solar = vs.Actor(name="Solar", signal=vs.Trace.load(
        dataset="solcast2022_global", column="Berlin", params={"scale": 8000}
    ))

    # No wind and no battery to isolate the impact of solar generation on the system.

    # 3. POLICY (Grid-Connected)
    policy = vs.DefaultPolicy(mode="grid-connected")

    # 4. CREATE MICROGRID
    env.add_microgrid(
        name="Mono_Objective_Solar_Only",
        actors=[datacenter, solar], # Only solar production, no wind.
        # storage=battery <-- No battery in this scenario
        policy=policy
    )

    logger = vs.MemoryLogger()
    env.add_controller(logger)

    print("Running a single-objective simulation (focusing entirely on Solaire)...")
    env.run(until=24 * 3600)

    # 5. VISUALIZATION
    fig = vs.plot_result_df(logger.to_df())
    fig.show()

if __name__ == "__main__":
    sim_mono_objective_solar()