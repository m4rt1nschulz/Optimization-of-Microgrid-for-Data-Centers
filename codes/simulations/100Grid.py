import vessim as vs

def sim_100_percent_grid():
    sim_start = "2022-06-15 00:00:00"
    env = vs.Environment(sim_start=sim_start, step_size=300)

    # 1. ONLY THE CONSUMER (1500W Server)
    datacenter = vs.Actor(name="Server", signal=vs.StaticSignal(value=-1500))

    # 2. NO PRODUCERS AND NO BATTERY
    # We load the carbon signal only for observation (it won't control anything here)
    carbon_signal = vs.Trace.load(dataset="watttime2023_caiso-north", params={"start_time": sim_start})

    # 3. DEFAULT POLICY (Grid-connected)
    # Since there is no battery or generation, the DefaultPolicy will simply buy 1500W from the grid all the time
    policy = vs.DefaultPolicy(mode="grid-connected")

    # 4. CREATING THE MICROGRID (Empty, only with the load)
    env.add_microgrid(
        name="100_Percent_Grid_Baseline",
        actors=[datacenter],
        # storage=battery  <-- REMOVED!
        policy=policy
    )

    # 5. CONTROLLER & EXECUTION
    logger = vs.MemoryLogger()
    env.add_controller(logger)

    print("Running 100% Electric Grid simulation (Baseline)...")
    env.run(until=24 * 3600)

    # 6. VISUALIZATION
    fig = vs.plot_result_df(logger.to_df())
    fig.show()

if __name__ == "__main__":
    sim_100_percent_grid()