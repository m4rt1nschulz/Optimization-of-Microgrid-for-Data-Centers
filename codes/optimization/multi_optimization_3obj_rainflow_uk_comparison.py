import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
import vessim as vs
from CustomBatteries import BoundedSimpleBattery
import time
import matplotlib.pyplot as plt
import rainflow

# SIMULATION PARAMETERS (EDIT HERE):
POP_SIZE = 50
N_GEN = 30

class SystemSizingProblem(ElementwiseProblem):
    def __init__(self, data_df):
        # We pass the dataframe into the problem so it can access it during evaluation
        # We have 3 decision variables (wind size, solar size, battery capacity) and 3 objectives (OPEX CO2, CAPEX CO2, and Grid Cost)
        super().__init__(n_var=3, n_obj=3, n_constr=0, 
                         xl=np.array([0, 0, 500]),
                         xu=np.array([5000, 3000, 10000]))
        self.df = data_df
        self.eval_count = 0

    def _evaluate(self, x, out, *args, **kwargs):
        # Start tracker for number of evaluations (useful for debugging and performance monitoring)
        self.eval_count += 1
        start_time = time.time()
        
        # x[0] = Wind power in kW
        # x[1] = Total Solar kWp
        # x[2] = Battery Capacity in kWh
        wind_size, solar_size, bat_cap = x

        # 2. Setup Vessim Environment
        # We use a 1-hour step size (3600 seconds) because the dataset is hourly
        environment = vs.Environment(sim_start="2022-01-01 00:00:00", step_size=3600)

        # 3. Define Actors
        datacenter_load = vs.Actor(
            name="main_server", 
            signal=vs.StaticSignal(value=-1_000_000) # 1 MW expressed in Watts
        )
        
        # We create the wind and solar actors using the data from the CSV, scaled by the Pymoo variables.
        wind_plant = vs.Actor(
            name="wind_turbine",
            signal=vs.Trace(actual=(self.df["Wind_Power_1kW"] * wind_size).fillna(0))
        )
        
        solar_plant = vs.Actor(
            name="solar_panel",
            signal=vs.Trace(actual=(self.df["Solar_Power_1kW"] * solar_size).fillna(0))
        )

        # 4. Define Storage (Battery)
        # Capacity in Watt-hours (Wh) -> so we multiply the Pymoo kWh variable by 1000
        battery = BoundedSimpleBattery(
            capacity=bat_cap * 1000, 
            initial_soc=0.5,     # Start half-full so it doesn't immediately hit the bottom limit
            min_soc=0.2,         # Hard lower bound (20%)
            max_soc=0.8          # Hard upper bound (80%)
        )

        # 5. Create Microgrid and Policy
        policy = vs.DefaultPolicy(mode="grid-connected") # Use grid when needed, but we will track how much we draw for cost calculation
        
        environment.add_microgrid(
            name="my_datacenter",
            actors=[datacenter_load, wind_plant, solar_plant],
            storage=battery,
            policy=policy
        )

        # 6. Logger to extract results
        logger = vs.MemoryLogger()
        environment.add_controller(logger)

        # 7. Run Simulation
        # Run for exactly 1 year (8760 hours * 3600 seconds)
        environment.run(until=8760 * 3600)

        # 8. Analyze Results for Pymoo
        res_df = logger.to_df()
        
        # Locate the column that contains the grid power (p_delta). We use a flexible search to avoid issues with column naming.
        p_delta_cols = [col for col in res_df.columns if 'p_delta' in col.lower()]
        if not p_delta_cols:
             raise ValueError(f"Could not find 'p_delta' in Vessim logger output. Available columns are: {res_df.columns.tolist()}")    
        p_delta_col_name = p_delta_cols[0]
        
        # Vessim logs the grid power. Negative values usually mean we are drawing from the grid (buying power)
        grid_draw_W = res_df[p_delta_col_name].clip(upper=0).abs()
        
        # Convert grid draw from Watts to MWh for price calculation
        # 1 Watt over 1 hour = 1 Wh. Divide by 1,000,000 to get MWh
        grid_draw_MWh = grid_draw_W / 1_000_000
        
        # Calculate Total Cost
        # Note: We align the hourly prices with the length of the simulation results
        hourly_prices = self.df["Price_EUR_MWh"].values[:len(grid_draw_MWh)]
        total_cost_grid = (grid_draw_MWh * hourly_prices).sum()

        # Calculate Average CO2 Constraint
        grid_draw_kWh = grid_draw_MWh * 1000  # Convert MWh to kWh for CO2 calculation
        hourly_co2 = self.df["CO2_Intensity_g_kWh"].values[:len(grid_draw_kWh)]
        
        # Calculate total grams of CO2 emitted over the whole year
        total_co2_emitted = (grid_draw_kWh * hourly_co2).sum()
        
        # Calculate total energy bought from the grid over the whole year
        total_kwh_bought = grid_draw_kWh.sum()
        
        # Divide total emissions by total energy to get the weighted average
        avg_co2 = (total_co2_emitted / total_kwh_bought) if total_kwh_bought > 0 else 0
        
        # RAINFLOW ALGORITHM
        soc_cols = [col for col in res_df.columns if 'soc' in col.lower()]
        if soc_cols:
            soc_series = res_df[soc_cols[0]].values
            cycles = rainflow.count_cycles(soc_series)
            total_damage = 0.0
            for rng, count in cycles: 
                if rng > 0.01: # Ignore micro-cycles smaller than 1% DoD
                    # Simplified Palmgren-Miner rule for Li-Ion
                    cycles_to_failure = 3000 * (0.8 / rng)**1.5
                    total_damage += count / cycles_to_failure
            
            raw_life = 1.0 / total_damage if total_damage > 0 else 20.0
            # FIX: Use min/max instead of np.clip for scalar floats
            estimated_life_years = max(1.0, min(raw_life, 20.0))
        else:
            estimated_life_years = 5.0
        
        # OBJECTIVE 1: Operational Emissions (OPEX CO2)
        # Convert total grams to kilograms for the year
        opex_co2_kg = total_co2_emitted / 1000
        
        # OBJECTIVE 2: Embedded Emissions (CAPEX CO2)
        capex_wind = wind_size * 200 / 20      # 200 kg CO2 per kW for wind turbines divided by 20 years of turbine life to get annualized emissions
        capex_solar = solar_size * 600 / 20  # 600 kg CO2 per kWp divided by 20 years of panel life to get annualized emissions
        capex_battery = bat_cap * 100 / estimated_life_years # 100 kg CO2 per kWh divided by the estimated life years of the battery to get annualized emissions
        capex_co2_kg = capex_wind + capex_solar + capex_battery
        
        # CAPEX economic cost (annualized)
        wind_capex_cost = wind_size * 1700 / 20    # €1700 per kW divided by 20 years
        solar_capex_cost = solar_size * 900 / 20  # €900 per kWp divided by 20 years
        battery_capex_cost = bat_cap * 450 / estimated_life_years     # €450 per kWh divided by estimated life years
        total_capex_cost = wind_capex_cost + solar_capex_cost + battery_capex_cost
        
        # Carbon cost (using a carbon price of €70 per tonne)
        carbon_price_per_kg = 70 / 1000  # Convert €70 per tonne to € per kg
        carbon_cost = opex_co2_kg * carbon_price_per_kg # Only OPEX CO2 (CAPEX CO2 to be included if desired)
        
        # OBJECTIVE 3: Total Cost = Grid Cost (OPEX) + Annualized CAPEX Cost + Carbon Cost
        total_cost = total_cost_grid + total_capex_cost + carbon_cost
        
        # Pass all three objectives to PyMoo (Minimize them)
        out["F"] = [opex_co2_kg, capex_co2_kg, total_cost]
        
        # End tracker and print results for this evaluation
        exec_time = time.time() - start_time
        print(f"Sim {self.eval_count} | OPEX CO2: {opex_co2_kg:,.0f} kg | CAPEX CO2: {capex_co2_kg:,.0f} kg | Coût total: {total_cost:,.0f} €")

# Helper function to get the best solution's battery cycle count using Rainflow (for analysis after optimization)
def get_best_solution_cycles(x, df):
    wind_size, solar_size, bat_cap = x
    env = vs.Environment(sim_start="2022-01-01 00:00:00", step_size=3600)
    
    datacenter_load = vs.Actor(name="main_server", signal=vs.StaticSignal(value=-1_000_000))
    wind_plant = vs.Actor(name="wind_turbine", signal=vs.Trace(actual=(df["Wind_Power_1kW"] * wind_size).fillna(0)))
    solar_plant = vs.Actor(name="solar_panel", signal=vs.Trace(actual=(df["Solar_Power_1kW"] * solar_size).fillna(0)))

    battery = BoundedSimpleBattery(capacity=bat_cap * 1000, initial_soc=0.5, min_soc=0.2, max_soc=0.8)
    
    env.add_microgrid(name="final", actors=[datacenter_load, wind_plant, solar_plant], storage=battery, policy=vs.DefaultPolicy(mode="grid-connected"))
    logger = vs.MemoryLogger()
    env.add_controller(logger)
    env.run(until=8760 * 3600)
    
    res_df = logger.to_df()
    soc_col = [col for col in res_df.columns if 'soc' in col.lower()][0]
    return list(rainflow.count_cycles(res_df[soc_col].values))

# 9. Run the Optimization
if __name__ == "__main__":
    print("Initializing Pymoo Multi-Objective Sizing Optimization...")
    # Datasets to test - Make sure the file names match your actual CSV files
    scenarios = [
        {"name": "FR", "file": "vessim_unified_data_2022.csv"},
        {"name": "UK", "file": "vessim_unified_data_london_2022.csv"}
    ]
    
    # We loop through each scenario (dataset) and run the optimization separately, printing results for each
    for scenario in scenarios:
        loc = scenario["name"]
        file_path = scenario["file"]
        
        print(f"\n{'='*60}")
        print(f"STARTING OPTIMIZATION FOR: {loc} ({file_path})")
        print(f"{'='*60}\n")

        try:
            data = pd.read_csv(file_path, parse_dates=['Datetime'], index_col='Datetime')
            if 'Consommation' not in data.columns:
                data.columns = ['Price_EUR_MWh', 'CO2_Intensity_g_kWh', 'Consommation', 'Solar_Power_1kW', 'Wind_Power_1kW']
        except FileNotFoundError:
            print(f" [!] Erreur : Fichier {file_path} introuvable. On passe au suivant.")
            continue    
    
        problem = SystemSizingProblem(data)
    
        # We use a very small population and generation size for initial testing
        # Increase these to 20/50 when you are ready for the final run
        algorithm = NSGA2(pop_size=POP_SIZE)
        res = minimize(problem, algorithm, ('n_gen', N_GEN), verbose=True)

        print(f"\n--- PARETO FRONT FOUND FOR {loc}---")
    
        if res.X is not None:
            # Combine the Results (OPEX, CAPEX, Wind, Solar, Battery)
            front = np.column_stack((res.F, res.X))
        
            # Sort by Operational CO2 (Lowest to Highest)
            front = front[front[:, 0].argsort()]
        
            # Save the Pareto front to a CSV for further analysis
            csv_name = f"pareto_front_{loc}.csv"
            header = "OPEX_CO2_kg,CAPEX_CO2_kg,Total_Cost_EUR,Wind_kW,Solar_kWp,Battery_kWh"
            np.savetxt(csv_name, front, delimiter=",", header=header, comments="")
            print(f"Data saved to {csv_name}")
            print(f"{'OPEX CO2 (kg)':>15} | {'CAPEX CO2 (kg)':>15} | {'Total Cost (€)':>15} || {'Wind kW':>6} | {'Solar kWp':>9} | {'Battery kWh':>11}")
            print("-" * 85)
        
            for solution in front:
                opex, capex, total_cost, wind, solar, bat = solution
                print(f"{opex:15,.0f} | {capex:15,.0f} | {total_cost:15,.0f} || {wind:6.1f} | {solar:9.0f} | {bat:11.0f}")
            
            # Plotting the Pareto Front in 2D with Grid Cost as a color dimension
            fig, ax = plt.subplots(figsize=(10, 7))
            # Convert kg to tonnes for better visualization
            opex_tonnes = front[:, 0] / 1000 # Convert to tonnes
            capex_tonnes = front[:, 1] / 1000 # Convert to tonnes
            grid_cost = front[:, 2] / 1_000_000  # Convert to millions of euros for better visualization
            # cmap='viridis' => gradient from purple (low cost) to yellow (high cost)
            sc = ax.scatter(capex_tonnes, opex_tonnes, c=grid_cost, cmap='viridis', 
                            s=80, edgecolors='black', alpha=0.9, zorder=5)
            ax.set_xlabel('Embedded Emissions / CAPEX (Tonnes CO2)', fontsize=16)
            ax.set_ylabel('Yearly Operational Emissions / OPEX (Tonnes CO2)', fontsize=16)
            ax.set_title(f'Pareto Front ({loc}): Carbon Trade-offs vs. Total Annual Cost', fontsize=16, fontweight='bold')
            # Add the color bar on the side
            cbar = plt.colorbar(sc)
            cbar.set_label('Yearly Total Cost (M€)', fontsize=16)
            ax.grid(True, linestyle='--', alpha=0.6, zorder=0)
            plt.tight_layout()
            # Save the plot
            plot_name = f"pareto_plot_{loc}.png"
            plt.savefig(plot_name)
            plt.close()
            print(f"Plot saved to {plot_name}")
        
            # Plot the cycle depth histogram for the most cost-effective  solution (lowest total cost)
            print("\nGenerating Cycle Depth histogram for the most cost-effective {loc} solution...")
            best_idx = np.argmin(res.F[:, 2]) # Find the row with the lowest total cost (index 2)
            best_x = res.X[best_idx]
            cycles = get_best_solution_cycles(best_x, data)
        
            if cycles:
                depths = [c[0] for c in cycles if c[0] > 0.01]
                counts = [c[1] for c in cycles if c[0] > 0.01]
                plt.figure(figsize=(10, 6))
                # Create a weighted histogram showing how many times the battery hit a specific DoD
                plt.hist(depths, weights=counts, bins=20, color='teal', edgecolor='black', alpha=0.7)
                plt.title(f"Battery Cycle Depth ({loc}) - Most Cost-Effective System\nWind: {best_x[0]:.0f}kW | Solar: {best_x[1]:.0f}kW | Bat: {best_x[2]:.0f}kWh", fontweight='bold')
                plt.xlabel("Cycle Depth (Depth of Discharge %)")
                plt.ylabel("Number of Cycles per Year")
                plt.grid(axis='y', alpha=0.6, linestyle='--')
                plt.tight_layout()
                # Save the plot
                rainflow_name = f"rainflow_plot_{loc}.png"
                plt.savefig(rainflow_name)
                plt.close()
                print(f"Rainflow histogram saved to {rainflow_name}")
  
        else:
            print(f"No valid configurations found for {loc}")
    print("\nALL SIMULATIONS COMPLETED SUCCESSFULLY!")