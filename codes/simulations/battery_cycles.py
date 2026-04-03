import vessim as vs
from datetime import datetime, timedelta
import pandas as pd

def sim_multi_day_degradation():
    sim_start = "2022-06-15 00:00:00"
    env = vs.Environment(sim_start=sim_start, step_size=300)

    # =================================================================
    # SETTING UP THE DURATION (Ex: 7 days)
    # =================================================================
    sim_duration_days = 7
    duration_seconds = sim_duration_days * 24 * 3600

    # 1. ACTORS (Server, Solar and Wind)
    datacenter = vs.Actor(name="Server", signal=vs.StaticSignal(value=-1500))
    solar = vs.Actor(name="Solar", signal=vs.Trace.load(
        dataset="solcast2022_global", column="Berlin", params={"scale": 4000}
    ))
    
    # Wind
    try:
        df_wind = pd.read_csv("../../data/vessim_results/vessim_wind_gif.csv", index_col=0, parse_dates=True)
        df_wind.index = df_wind.index + (pd.to_datetime(sim_start) - df_wind.index[0])
        wind = vs.Actor(name="Wind", signal=vs.Trace(actual=df_wind["power_W"]))
    except FileNotFoundError:
        wind = vs.Actor(name="Wind", signal=vs.StaticSignal(value=400))

    # 2. BATTERY
    battery = vs.SimpleBattery(capacity=10000, initial_soc=0.5, min_soc=0.2)

    # 3. DEFAULT OR SMART POLICY (I'll use the default here to simplify)
    policy = vs.DefaultPolicy(mode="grid-connected")

    env.add_microgrid(
        name="Gif_DataCenter_LongTerm", 
        actors=[datacenter, solar, wind], 
        storage=battery, 
        policy=policy
    )

    logger = vs.MemoryLogger()
    env.add_controller(logger)

    print(f"Starting simulation for {sim_duration_days} days...")
    env.run(until=duration_seconds)

    # =================================================================
    # BATTERY DEGRADATION AND CYCLE CALCULATION
    # =================================================================
    df_results = logger.to_df()
    
    soc_columns = [col for col in df_results.columns if 'soc' in col.lower() or 'state of charge' in col.lower()]
    
    if soc_columns:
        soc_series = df_results[soc_columns[0]]
        
        # Sum of absolute charge variations
        soc_diffs = soc_series.diff().abs().dropna()
        
        # Calculate cycles (if SoC is from 0.0 to 1.0, divide by 2)
        if soc_series.max() <= 1.0:
            total_cycles = soc_diffs.sum() / 2.0
        else:
            total_cycles = soc_diffs.sum() / 200.0

        # DEGRADATION MATHEMATICS
        limite_ciclos_bateria = 3000  # Standard for Lithium (LFP) batteries
        ciclos_por_ano = total_cycles * (365 / sim_duration_days)
        
        if ciclos_por_ano > 0:
            anos_de_vida = limite_ciclos_bateria / ciclos_por_ano
        else:
            anos_de_vida = float('inf')
            
        degradacao_periodo = (total_cycles / limite_ciclos_bateria) * 100

        # BATTERY DEGRADATION REPORT
        print("\n" + "="*55)
        print("📊 BATTERY DEGRADATION REPORT (VESSIM)")
        print("="*55)
        print(f"🔹 Simulation Duration      : {sim_duration_days} days")
        print(f"🔹 Total Capacity           : {battery.capacity} Wh")
        print(f"🔹 Cycles Completed         : {total_cycles:.2f} equivalent cycles")
        print(f"🔹 Degradation During Period: {degradacao_periodo:.4f} %")
        print("-" * 55)
        print(f"📈 Annual Projection         : ~{ciclos_por_ano:.0f} cycles / year")
        print(f"⏳ Estimated Lifespan        : ~{anos_de_vida:.1f} years (for {limite_ciclos_bateria} cycles)")
        print("="*55 + "\n")

    # =================================================================
    
    fig = vs.plot_result_df(df_results)
    fig.show()

if __name__ == "__main__":
    sim_multi_day_degradation()