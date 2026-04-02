import pandas as pd
import numpy as np
import os

def calculate_local_emissions(df):
    """Applies ADEME Life Cycle Assessment (LCA) factors."""
    return (df['Nucléaire'] * 6 + 
            df['Eolien'] * 14 + 
            df['Solaire'] * 43 + 
            df['Hydraulique'] * 6 + 
            df['Bioénergies'] * 130 + 
            df['Thermique'] * 418)

def clean_datetime_index(df):
    """Converts 'Date' and 'Heures' columns into a single clean Datetime index."""
    df = df.dropna(subset=['Date', 'Heures']).copy()
    df['Consommation'] = pd.to_numeric(df['Consommation'], errors='coerce')
    df = df.dropna(subset=['Consommation'])
    df['Heures'] = df['Heures'].astype(str).str.replace('24:00', '00:00')
    df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Heures'])
    return df.set_index('Datetime').fillna(0)

def main():
    # We only process the year 2022
    year = 2022
    print(f"Starting processing for the unified {year} dataset...\n")

    # 1. Define Paths
    price_file = f'codes-martin/data/energy-charts/energy-charts_Electricity_production_and_spot_prices_in_France_in_{year}.xlsx'     
    idf_file = f'codes-martin/data/eCO2_idf/eCO2mix_RTE_Ile-de-France_Annuel-Definitif_{year}.xls'
    fr_file = f'codes-martin/data/eCO2_france/eCO2mix_RTE_Annuel-Definitif_{year}.xls'
    
    # Paths for your local solar and wind data
    solar_file = 'codes-martin/data/vessim_solar_gif.csv'
    wind_file = 'codes-martin/data/vessim_wind_gif.csv'

    # Safety check
    if not (os.path.exists(price_file) and os.path.exists(idf_file) and os.path.exists(fr_file)):
        print(f" [!] Missing one or more RTE/Energy-Charts files for {year}.")
        return
    if not (os.path.exists(solar_file) and os.path.exists(wind_file)):
        print(f" [!] Missing solar or wind CSV files. Please check paths.")
        return

    # 2. Load and Clean Pricing Data
    if price_file.endswith('.csv'):
        df_prices = pd.read_csv(price_file)
    else:
        df_prices = pd.read_excel(price_file)
        
    df_prices = df_prices.iloc[:, [0, 4]].copy()
    col_price_date = df_prices.columns[0]
    df_prices = df_prices.rename(columns={df_prices.columns[1]: 'Price_EUR_MWh'})
    df_prices['Price_EUR_MWh'] = pd.to_numeric(df_prices['Price_EUR_MWh'], errors='coerce').fillna(0)
    df_prices[col_price_date] = pd.to_datetime(
        df_prices[col_price_date], errors='coerce', format='mixed', utc=True
    ).dt.tz_localize(None)
    df_prices = df_prices.dropna(subset=[col_price_date])
    
    # 3. Load and Clean RTE Data
    df_idf = pd.read_csv(idf_file, sep='\t', encoding='latin1', low_memory=False, index_col=False)
    df_fr = pd.read_csv(fr_file, sep='\t', encoding='latin1', low_memory=False, index_col=False)

    df_idf = clean_datetime_index(df_idf)
    df_fr = clean_datetime_index(df_fr)

    generation_cols = ['Nucléaire', 'Eolien', 'Solaire', 'Hydraulique', 'Bioénergies', 'Thermique']
    for col in generation_cols:
        if col in df_idf.columns:
            df_idf[col] = pd.to_numeric(df_idf[col], errors='coerce').fillna(0)
            
    col_co2_fr = [col for col in df_fr.columns if 'Co2' in col or 'CO2' in col][0]
    df_fr[col_co2_fr] = pd.to_numeric(df_fr[col_co2_fr], errors='coerce').fillna(0)

    # 4. Calculate Mass Balance
    national_intensity_fr = df_fr[col_co2_fr]
    total_gen_idf = (df_idf['Nucléaire'] + df_idf['Eolien'] + df_idf['Solaire'] + 
                     df_idf['Hydraulique'] + df_idf['Bioénergies'] + df_idf['Thermique'])
    local_co2_idf = calculate_local_emissions(df_idf)
    imports_idf = np.maximum(0, df_idf['Consommation'] - total_gen_idf)
    total_consumed_co2_idf = local_co2_idf + (imports_idf * national_intensity_fr)
    
    df_idf['CO2_Intensity_g_kWh'] = np.where(
        df_idf['Consommation'] > 0, 
        total_consumed_co2_idf / df_idf['Consommation'], 
        0
    )

    df_idf_hourly = df_idf[['CO2_Intensity_g_kWh', 'Consommation']].resample('1h').mean().reset_index()

    # 5. Merge Price and CO2
    df_base = pd.merge(
        df_prices, 
        df_idf_hourly, 
        left_on=col_price_date, 
        right_on='Datetime', 
        how='inner'
    )
    df_base = df_base[['Datetime', 'Price_EUR_MWh', 'CO2_Intensity_g_kWh', 'Consommation']]
    df_base.set_index('Datetime', inplace=True)

    # 6. Load Wind and Solar Data
    df_solar = pd.read_csv(solar_file, index_col=0, parse_dates=True)
    df_solar.rename(columns={'power_W': 'Solar_Power_1kW'}, inplace=True)
    
    df_wind = pd.read_csv(wind_file, index_col=0, parse_dates=True)
    df_wind.rename(columns={'power_W': 'Wind_Power_1kW'}, inplace=True)

    # Ensure indices are timezone-naive to match df_base
    if df_solar.index.tz is not None:
        df_solar.index = df_solar.index.tz_localize(None)
    if df_wind.index.tz is not None:
        df_wind.index = df_wind.index.tz_localize(None)

    # 7. Final Merge
    print("Merging Grid Data with local Solar and Wind profiles...")
    # Join everything together on the Datetime index
    master_df = df_base.join(df_solar, how='inner').join(df_wind, how='inner')
    
    # 8. Output
    output_name = f'vessim_unified_data_{year}.csv'
    master_df.reset_index(inplace=True)
    master_df.to_csv(output_name, index=False)
    
    print(f"\nSUCCESS! Created '{output_name}' with {len(master_df)} rows.")
    print("Columns included: Datetime, Price_EUR_MWh, CO2_Intensity_g_kWh, Consommation, Solar_Power_1kW, Wind_Power_1kW")

if __name__ == "__main__":
    main()