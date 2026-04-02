import PySAM.Pvwattsv8 as pvwatts
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Initialize the PVWatts model
    pv = pvwatts.default("PVWattsNone")
    # Point to the weather file from the NREL TMY dataset 
    pv.SolarResource.solar_resource_file = "gif_weather_tmy-2022.csv"
    # Define the solar plant size
    pv.SystemDesign.system_capacity = 4.0 
    pv.SystemDesign.tilt = 35.0      # Typical roof angle
    pv.SystemDesign.azimuth = 180.0  # Facing South
    pv.SystemDesign.losses = 14.0    # Standard system losses (%)
    
    # Run the physics simulation
    print("Running PySAM simulation...")
    pv.execute()
    # Extract the hourly AC power output (in Watts)
    power_output_watts = pv.Outputs.ac
    # Attach date to the PySAM outputs
    dates = pd.date_range(start="2022-01-01 00:00", periods=8760, freq="h")
    df = pd.DataFrame({"power_W": power_output_watts}, index=dates)
    # Clean up any negative values (inverters sometimes draw tiny power at night)
    df["power_W"] = df["power_W"].clip(lower=0)
    # Save to CSV so Vessim can use it later
    df.to_csv("vessim_solar_gif.csv")
    print("Saved simulation data to 'vessim_solar_gif.csv'")

    # Plot Seasonal Average Daily Curves
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    # Define seasons based on months
    seasons = {
        "Winter": [12, 1, 2],
        "Spring": [3, 4, 5],
        "Summer": [6, 7, 8],
        "Autumn": [9, 10, 11]
    }
    # Plot the average daily curve for each season
    plt.figure(figsize=(10, 5))
    for season_name, months in seasons.items():
        # Filter data by season, then group by hour and calculate the mean
        season_data = df[df["month"].isin(months)]
        hourly_avg = season_data.groupby("hour")["power_W"].mean()
        
        plt.plot(hourly_avg.index, hourly_avg.values, label=season_name, linewidth=2)
    # Plot formatting
    plt.title("Average Daily Solar Generation in Gif-sur-Yvette (4kW System)", fontsize=14)
    plt.xlabel("Hour of the Day", fontsize=12)
    plt.ylabel("Power Output (Watts)", fontsize=12)
    plt.xticks(range(0, 24, 2)) # Show every 2 hours
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()