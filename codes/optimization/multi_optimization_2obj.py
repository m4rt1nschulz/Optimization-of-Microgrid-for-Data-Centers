import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
import vessim as vs
from CustomBatteries import BoundedSimpleBattery
import time
import matplotlib.pyplot as plt

# 1. Load Data
# For this to run, the CSV must contain columns named: 
# 'Price_EUR_MWh', 'CO2_Intensity_g_kWh', 'Wind_Power_1kW', and 'Solar_Power_1kW'.
data = pd.read_csv("vessim_unified_data_2022.csv", parse_dates=['Datetime'], index_col='Datetime')
data.columns = ['Price_EUR_MWh', 'CO2_Intensity_g_kWh', 'Consommation', 'Solar_Power_1kW', 'Wind_Power_1kW']

class SystemSizingProblem(ElementwiseProblem):
    def __init__(self, data_df):
        # We pass the dataframe into the problem so it can access it during evaluation
        # We have 3 decision variables (wind size, solar size, battery capacity) and 2 objectives (OPEX CO2 and CAPEX CO2)
        super().__init__(n_var=3, n_obj=2, n_constr=0, 
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
        total_cost = (grid_draw_MWh * hourly_prices).sum()

        # Calculate Average CO2 Constraint
        grid_draw_kWh = grid_draw_MWh * 1000  # Convert MWh to kWh for CO2 calculation
        hourly_co2 = self.df["CO2_Intensity_g_kWh"].values[:len(grid_draw_kWh)]
        
        # Calculate total grams of CO2 emitted over the whole year
        total_co2_emitted = (grid_draw_kWh * hourly_co2).sum()
        
        # Calculate total energy bought from the grid over the whole year
        total_kwh_bought = grid_draw_kWh.sum()
        
        # Divide total emissions by total energy to get the weighted average
        avg_co2 = (total_co2_emitted / total_kwh_bought) if total_kwh_bought > 0 else 0
        
        # OBJECTIVE 1: Operational Emissions (OPEX CO2)
        # Convert total grams to kilograms for the year
        opex_co2_kg = total_co2_emitted / 1000
        
        # OBJECTIVE 2: Embedded Emissions (CAPEX CO2)
        capex_wind = wind_size * 200       # 200 kg CO2 per kW for wind turbines
        capex_solar = solar_size * 600   # 600 kg CO2 per kWp
        capex_battery = bat_cap * 100    # 100 kg CO2 per kWh
        capex_co2_kg = capex_wind + capex_solar + capex_battery
        
        # Pass both objectives to PyMoo (Minimize both)
        out["F"] = [opex_co2_kg, capex_co2_kg]
        
        # End tracker and print results for this evaluation
        exec_time = time.time() - start_time
        print(f"Sim {self.eval_count} | OPEX: {opex_co2_kg:,.0f} kg | CAPEX: {capex_co2_kg:,.0f} kg")

# 9. Run the Optimization
if __name__ == "__main__":
    print("Initializing Pymoo Multi-Objective Sizing Optimization...")
    problem = SystemSizingProblem(data)
    
    # We use a very small population and generation size for initial testing
    # Increase these to 20/50 when you are ready for the final run
    algorithm = NSGA2(pop_size=20)
    res = minimize(problem, algorithm, ('n_gen', 10), verbose=True)

    print("\n--- PARETO FRONT FOUND ---")
    
    if res.X is not None:
        # Combine the Results (OPEX, CAPEX, Wind, Solar, Battery)
        front = np.column_stack((res.F, res.X))
        
        # Sort by Operational CO2 (Lowest to Highest)
        front = front[front[:, 0].argsort()]
        
        print(f"{'OPEX CO2 (kg)':>15} | {'CAPEX CO2 (kg)':>15} || {'Wind kW':>6} | {'Solar kWp':>9} | {'Battery kWh':>11}")
        print("-" * 70)
        
        for solution in front:
            opex, capex, wind, solar, bat = solution
            print(f"{opex:15,.0f} | {capex:15,.0f} || {wind:6.1f} | {solar:9.0f} | {bat:11.0f}")
            
        # Plot the Pareto Front (OPEX vs CAPEX)
        plt.figure(figsize=(10, 6))
        # Convert kg to tonnes for better visualization
        capex_tonnes = front[:, 1] / 1000
        opex_tonnes = front[:, 0] / 1000
        plt.scatter(capex_tonnes, opex_tonnes, color='blue', edgecolors='black', s=50)
        plt.plot(capex_tonnes, opex_tonnes, color='gray', linestyle='--', alpha=0.5)
        plt.title("Pareto Front: Embedded vs. Operational Carbon Emissions")
        plt.xlabel("Embedded Emissions / CAPEX (kg CO2)")
        plt.ylabel("Yearly Operational Emissions / OPEX (kg CO2)")
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
    else:
        print("No valid configurations found.")