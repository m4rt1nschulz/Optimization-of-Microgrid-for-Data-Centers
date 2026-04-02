import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2
import vessim
from datetime import datetime

# Load the Typical Energy Year dataset
data = pd.read_csv("vessim_unified_data_2022.csv", parse_dates=['Datetime'], index_col='Datetime')
signals = vessim.signal.HistoricalSignal.from_df(data)

class MyProblem(ElementwiseProblem):
    def __init__(self):
        # n_var=3 (Wind, Solar, Battery), n_obj=1 (Cost), n_constr=1 (CO2)
        # Bounds: [0 turbines, 0kW solar, 100kWh battery] to [10, 2000, 5000]
        super().__init__(n_var=3, n_obj=1, n_constr=1, 
                         xl=np.array([0, 0, 100]), 
                         xu=np.array([10, 2000, 5000]))

    def _evaluate(self, x, out, *args, **kwargs):
        # x[0] = Wind Turbines, x[1] = Solar kWp, x[2] = Battery Capacity
        n_wind, solar_size, bat_cap = x

        # 1. Setup Vessim components with the Pymoo variables
        wind_gen = vessim.core.Generator(name="wind", p=data["Wind_Power_1kW"] * n_wind)
        solar_gen = vessim.core.Generator(name="solar", p=data["Solar_Power_1kW"] * solar_size) 
        battery = vessim.storage.SimpleBattery(capacity=bat_cap, charge_level=bat_cap*0.5) # Charge limits to be set
        datacenter = vessim.core.Consumer(name="datacenter", p=-500) # Constant 500kW consumption

        # 2. Run the simulation
        # Using a simple "Self-Consumption" policy: Use renewables first, then battery, then grid.
        microgrid = vessim.core.Microgrid(
            actors=[datacenter, wind_gen, solar_gen, battery],
            policy=vessim.policy.DefaultPolicy(mode="grid-connected"), # Simple built-in policy for sizing
            signals=signals
        )
        
        # Run for the whole year (or a representative month to save time)
        results = microgrid.run(until=datetime(2022, 12, 31))
        
        # 3. Calculate Objective (Total Grid Cost during the year)
        # Sum of (Power from Grid * Price at that hour)
        grid_energy = results.p_delta[results.p_delta < 0].abs() # Only multiply the hours where we drew from the grid
        hourly_prices = data.loc[results.index, "Price_EUR_MWh"]
        total_cost = (grid_energy * hourly_prices / 1000).sum() # Convert MWh to kWh for cost calculation

        # 4. Calculate Constraint (CO2 Threshold)
        # We check the grid CO2 only during hours where we actually used the grid.
        grid_hours_co2 = data.loc[grid_energy > 0, "CO2_Intensity_g_kWh"]
        max_co2 = grid_hours_co2.max() if not grid_hours_co2.empty else 0 # If we never used the grid, max CO2 is 0
        
        # Constraint: Max CO2 - 72 <= 0
        out["F"] = [total_cost]
        out["G"] = [max_co2 - 72]

# Run the optimization
problem = MyProblem()
algorithm = NSGA2(pop_size=20)
res = minimize(problem, algorithm, ('n_gen', 50), verbose=True)

print(f"Best Configuration: Wind: {res.X[0]:.1f}, Solar: {res.X[1]:.0f}kWp, Battery: {res.X[2]:.0f}kWh")