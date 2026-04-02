import pandas as pd
import numpy as np
import os

def main():
    # We only process the year 2022
    year = 2022
    print(f"Starting processing for the unified {year} dataset...\n")

    # 1. Define Paths
    price_file = f'codes-martin/data/ember-energy/uk_energy_prices.csv'     
    co2_file = f'codes-martin/data/NESO/uk_generation_mix_and_emissions.csv'
    
    # Paths for your local solar and wind data
    solar_file = 'codes-martin/data/vessim_solar_london.csv'
    wind_file = 'codes-martin/data/vessim_wind_london.csv'

    # Safety check
    if not (os.path.exists(price_file) and os.path.exists(co2_file)):
        print(f" [!] Missing one or more RTE/Energy-Charts files for {year}.")
        return
    if not (os.path.exists(solar_file) and os.path.exists(wind_file)):
        print(f" [!] Missing solar or wind CSV files. Please check paths.")
        return

    # 2. Load and Clean Pricing Data (Ember)
    print("Loading and cleaning Ember Price Data...")
    df_prices = pd.read_csv(price_file)
    
    # We use 'Datetime (Local)' to align with the sun/wind, avoiding UTC shift issues.
    # We parse it, remove the timezone awareness so it matches PySAM, and set it as index.
    df_prices['Datetime'] = pd.to_datetime(df_prices['Datetime (Local)'], utc=True).dt.tz_localize(None)
    df_prices.set_index('Datetime', inplace=True)
    
    # Filter strictly for the year 2022
    df_prices = df_prices.loc[str(year)]
    
    # Keep only the price column and rename it
    df_prices = df_prices[['Price (EUR/MWhe)']].rename(columns={'Price (EUR/MWhe)': 'Price_EUR_MWh'})
    
    # Ensure it's strictly hourly (fills any missing single hours to prevent shape mismatches)
    df_prices = df_prices.resample('1h').mean().ffill()

    # 3. Load and Clean NESO Carbon Data
    print("Loading and cleaning NESO Carbon Intensity Data...")
    # Read only the two columns we care about to save memory (the file has data from 2009!)
    df_co2 = pd.read_csv(co2_file, usecols=['DATETIME', 'CARBON_INTENSITY'])
    
    df_co2['DATETIME'] = pd.to_datetime(df_co2['DATETIME'], utc=True).dt.tz_localize(None)
    df_co2.set_index('DATETIME', inplace=True)
    
    # Filter strictly for the year 2022
    df_co2 = df_co2.loc[str(year)]
    
    # UK data is 30 mins. We must resample it to 1-hour steps by taking the mean of every 2 half-hours!
    df_co2_hourly = df_co2.resample('1h').mean().ffill()
    df_co2_hourly.rename(columns={'CARBON_INTENSITY': 'CO2_Intensity_g_kWh'}, inplace=True)

    # 4. Merge Price and CO2
    print("Merging Prices and Carbon...")
    # Combine them on their shared hourly Datetime index
    df_base = pd.merge(df_prices, df_co2_hourly, left_index=True, right_index=True, how='inner')
    
    # Add a empty "Consommation" column because the Vessim script expects it
    df_base['Consommation'] = 0 

    # 5. Load PySAM Wind and Solar Data
    print("Loading Local PySAM Weather Data...")
    df_solar = pd.read_csv(solar_file, index_col=0, parse_dates=True)
    df_solar.rename(columns={'power_W': 'Solar_Power_1kW'}, inplace=True)
    
    df_wind = pd.read_csv(wind_file, index_col=0, parse_dates=True)
    df_wind.rename(columns={'power_W': 'Wind_Power_1kW'}, inplace=True)

    # Ensure indices are timezone-naive to match df_base perfectly
    if df_solar.index.tz is not None:
        df_solar.index = df_solar.index.tz_localize(None)
    if df_wind.index.tz is not None:
        df_wind.index = df_wind.index.tz_localize(None)

    # 6. Final Merge
    print("Merging Grid Data with local Solar and Wind profiles...")
    master_df = df_base.join(df_solar, how='inner').join(df_wind, how='inner')
    
    # Safety Check: Did we get exactly 1 year of hourly data?
    if len(master_df) != 8760:
        print(f" [!] Warning: Expected 8760 hours, but got {len(master_df)} rows. Check for missing data in the raw CSVs.")

    # 7. Output
    output_name = f'vessim_unified_data_london_{year}.csv'
    
    # Reset index so 'Datetime' becomes a normal column again before saving
    master_df.reset_index(inplace=True)
    master_df.rename(columns={'index': 'Datetime'}, inplace=True) # Just in case reset_index names it 'index'
    
    master_df.to_csv(output_name, index=False)
    
    print(f"\nSUCCESS! Created '{output_name}' with {len(master_df)} rows.")
    print("Columns included: Datetime, Price_EUR_MWh, CO2_Intensity_g_kWh, Consommation, Solar_Power_1kW, Wind_Power_1kW")

if __name__ == "__main__":
    main()
    