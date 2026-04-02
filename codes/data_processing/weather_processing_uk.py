import PySAM.Pvwattsv8 as pvwatts
import PySAM.Windpower as windpower
import pandas as pd
import matplotlib.pyplot as plt

def main():
    # Point to the weather file from the NREL TMY dataset 
    weather_file = "codes-martin\data\london_weather_tmy-2022.csv"
    # Create a date range for the entire year (8760 hours)
    dates = pd.date_range(start="2022-01-01 00:00", periods=8760, freq="h")

    # SOLAR SIMULATION
    print("Running PySAM Solar simulation...")
    # Initialize the PVWatts model
    pv = pvwatts.default("PVWattsNone")
    pv.SolarResource.solar_resource_file = weather_file
    # Define the solar plant size
    pv.SystemDesign.system_capacity = 1.0 # Scale to 1kW for better comparison
    pv.SystemDesign.tilt = 40.0      # Typical roof angle (London)
    pv.SystemDesign.azimuth = 180.0  # Facing South
    pv.SystemDesign.losses = 14.0    # Standard system losses (%)  
    pv.execute()
    # Save solar results to CSV
    df_solar = pd.DataFrame({"power_W": pv.Outputs.ac}, index=dates)
    df_solar["power_W"] = df_solar["power_W"].clip(lower=0)
    df_solar.to_csv("vessim_solar_london.csv")

    # WIND SIMULATION
    print("Translating Solar CSV to Wind SRW format...")
    # Read the NREL file, skip the 2 metadata rows and ensure 8760 hours
    df_nrel = pd.read_csv(weather_file, skiprows=2).head(8760)
    # We apply the Hellmann Wind Profile Power Law (alpha = 0.143 for onshore)
    df_nrel['Wind Speed 80m'] = df_nrel['Wind Speed'] * ((80.0 / 10.0) ** 0.143)
    # Prepare the .srw file for PySAM Windpower
    wind_ready_file = "vessim_wind_input.srw"
    # Write the file manually to ensure correct formatting
    with open(wind_ready_file, 'w', newline='\n') as f:
        # Row 1: Metadata (Location ID, City, State, Country, Year, Lat, Lon, Elev)
        f.write("355378,London,ENG,UK,2022,51.49,-0.14,11\n")
        # Row 2: Description (PySAM completely ignores this, but it MUST be here to push Row 3 down!)
        f.write("NREL UK Weather Data Extrapolated to 80m by Pandas\n")
        # Row 3: LABELS (PySAM looks exactly here. Must be Speed, Direction, Temperature, Pressure)
        f.write("Speed,Direction,Temperature,Pressure\n")
        # Row 4: UNITS 
        f.write("m/s,deg,C,atm\n")
        # Row 5: HEIGHTS in meters
        f.write("80,80,2,0\n")
        # Row 6 to 8765: The pure data (4 columns)
        total_rows = len(df_nrel)
        for i, row in df_nrel.iterrows():
            wspd = row['Wind Speed 80m']
            wdir = row['Wind Direction']
            temp = row['Temperature']
            pres_atm = row['Pressure'] / 1013.25 # Convert hPa to atm
            line = f"{wspd},{wdir},{temp},{pres_atm}"
            # Write avoiding ghost trailing newlines
            if i < total_rows - 1:
                f.write(line + "\n")
            else:
                f.write(line)
    # Run the PySAM Wind simulation
    print("Running PySAM Wind simulation...")
    # Initialize the Windpower model
    wind = windpower.default("WindpowerNone")
    wind.Resource.wind_resource_filename = wind_ready_file
    wind.Turbine.wind_turbine_hub_ht = 80 # Height of the turbine
    wind.execute() 
    print("Wind simulation successful!")
    # Scale the raw wind power output to a 1kW system
    raw_wind_kw = wind.Outputs.gen
    max_wind_kw = max(raw_wind_kw)
    scaled_wind_watts = [(x / max_wind_kw) * 1000 for x in raw_wind_kw]
    # Save wind results to CSV
    df_wind = pd.DataFrame({"power_W": scaled_wind_watts}, index=dates)
    df_wind["power_W"] = df_wind["power_W"].clip(lower=0)
    df_wind.to_csv("vessim_wind_london.csv")

    # COMPARATIVE PLOTS
    # Group by hour and month
    df_solar["hour"] = df_solar.index.hour
    df_solar["month"] = df_solar.index.month
    df_wind["hour"] = df_wind.index.hour
    df_wind["month"] = df_wind.index.month
    # Define seasons based on months
    seasons = {
        "Winter": [12, 1, 2],
        "Spring": [3, 4, 5],
        "Summer": [6, 7, 8],
        "Autumn": [9, 10, 11]
    }
    # Create side-by-side plots for solar and wind
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    # Add a main title for the entire figure
    fig.suptitle("Seasonal Renewable Energy Profiles - London, UK", 
                 fontsize=18, fontweight='bold')
    # Loop through seasons to plot both average daily curves
    for season_name, months in seasons.items():
        # Solar Curve
        solar_season = df_solar[df_solar["month"].isin(months)]
        solar_avg = solar_season.groupby("hour")["power_W"].mean()
        solar_avg.loc[24] = solar_avg.loc[0] # Add hour 24 for better plotting continuity
        ax1.plot(solar_avg.index, solar_avg.values, label=season_name, linewidth=2)
        # Wind Curve
        wind_season = df_wind[df_wind["month"].isin(months)]
        wind_avg = wind_season.groupby("hour")["power_W"].mean()
        wind_avg.loc[24] = wind_avg.loc[0] # Add hour 24 for better plotting continuity
        ax2.plot(wind_avg.index, wind_avg.values, label=season_name, linewidth=2)
    # Format the solar plot
    ax1.set_title("Average Daily Solar Generation (1kWp)", fontsize=14)
    ax1.set_xlabel("Hour of the Day", fontsize=12)
    ax1.set_ylabel("Power Output (Watts)", fontsize=12)
    ax1.set_xticks(range(0, 25))
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()
    # Format the wind plot
    ax2.set_title("Average Daily Wind Generation (1kW Scaled, 80m Height)", fontsize=14)
    ax2.set_xlabel("Hour of the Day", fontsize=12)
    ax2.set_ylabel("Power Output (Watts)", fontsize=12)
    ax2.set_xticks(range(0, 25))
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()
    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()
    
# Run the main function
if __name__ == "__main__":
    main()