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
    years = range(2016, 2024) # Loops 2016 through 2023
    all_years_data = []

    print("Starting processing to create the Typical Energy Year (Average 2016-2023)...\n")

    for year in years:
        # Paths based on your new folder structure
        # Note: Energy Charts usually downloads as .csv, but we check for .xlsx just in case
        price_file = f'codes-martin/data/energy-charts/energy-charts_Electricity_production_and_spot_prices_in_France_in_{year}.xlsx'     
        idf_file = f'codes-martin/data/eCO2_idf/eCO2mix_RTE_Ile-de-France_Annuel-Definitif_{year}.xls'
        fr_file = f'codes-martin/data/eCO2_france/eCO2mix_RTE_Annuel-Definitif_{year}.xls'

        # Check if files exist for this specific year
        if not (os.path.exists(price_file) and os.path.exists(idf_file) and os.path.exists(fr_file)):
            print(f"  [!] Missing one or more files for {year}. Skipping...")
            continue
            
        print(f"  -> Processing Year {year}...")

        # 1. Load data
        if price_file.endswith('.csv'):
            df_prices = pd.read_csv(price_file)
        else:
            df_prices = pd.read_excel(price_file)
            
        # RTE files are often tab-separated and encoded in latin1, so we use read_csv with those parameters
        df_idf = pd.read_csv(idf_file, sep='\t', encoding='latin1', low_memory=False, index_col=False)
        df_fr = pd.read_csv(fr_file, sep='\t', encoding='latin1', low_memory=False, index_col=False)
        
        # Extract only Index 0 (Column A - Date) and Index 4 (Column E - Price)
        df_prices = df_prices.iloc[:, [0, 4]].copy()
        
        # Standardize the column names so the rest of the code works
        col_price_date = df_prices.columns[0]
        df_prices = df_prices.rename(columns={df_prices.columns[1]: 'Price_EUR_MWh'})
        
        # Force the price to be numeric (converts any text/garbage to NaN, then to 0)
        df_prices['Price_EUR_MWh'] = pd.to_numeric(df_prices['Price_EUR_MWh'], errors='coerce').fillna(0)
        
        # Clean the dates (format='mixed' stops that annoying red UserWarning)
        df_prices[col_price_date] = pd.to_datetime(
            df_prices[col_price_date], errors='coerce', format='mixed', utc=True
        ).dt.tz_localize(None)
        
        # Drop rows that don't have a valid date (like header/metadata rows)
        df_prices = df_prices.dropna(subset=[col_price_date])
        
        # 2. Format dates
        df_idf = clean_datetime_index(df_idf)
        df_fr = clean_datetime_index(df_fr)

        # 2.5 FORCE NUMERIC DATA
        # Define the columns we need to do math on
        generation_cols = ['Nucléaire', 'Eolien', 'Solaire', 'Hydraulique', 'Bioénergies', 'Thermique']
        
        # Force them to numeric in IDF dataframe, replacing errors (like '-' or 'ND') with NaN, then fill with 0
        for col in generation_cols:
            if col in df_idf.columns:
                df_idf[col] = pd.to_numeric(df_idf[col], errors='coerce').fillna(0)
                
        # Do the same for the France dataframe (since we need its CO2 column to be numeric too)
        # We also force the 'Taux de Co2' column to be numeric just in case
        col_co2_fr = [col for col in df_fr.columns if 'Co2' in col or 'CO2' in col][0]
        df_fr[col_co2_fr] = pd.to_numeric(df_fr[col_co2_fr], errors='coerce').fillna(0)


        # 3. Extract National CO2
        national_intensity_fr = df_fr[col_co2_fr]

        # 4. Mass Balance for IDF
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

        # 5. Resample to 1-hour
        df_idf_hourly = df_idf[['CO2_Intensity_g_kWh', 'Consommation']].resample('1h').mean().reset_index()
        
        # 6. Prepare Prices Date
        col_price_date = df_prices.columns[0] 
        # ADDED errors='coerce' to handle extra header strings hidden in the data
        df_prices[col_price_date] = pd.to_datetime(df_prices[col_price_date], errors='coerce', utc=True).dt.tz_localize(None)
        
        # ADDED: Drop the rows that contained those invalid text strings
        df_prices = df_prices.dropna(subset=[col_price_date])

        # 7. Merge datasets for the year
        df_final_year = pd.merge(
            df_prices, 
            df_idf_hourly, 
            left_on=col_price_date, 
            right_on='Datetime', 
            how='inner'
        )

        # Clean up column names (find the price column which may have different names like 'Price' or 'Day Ahead Price')
        price_col_name = [col for col in df_final_year.columns if 'Price' in col or 'Day Ahead' in col][0]
        df_final_year = df_final_year.rename(columns={price_col_name: 'Price_EUR_MWh'})
        
        # Keep only what we need
        df_final_year = df_final_year[['Datetime', 'Price_EUR_MWh', 'CO2_Intensity_g_kWh', 'Consommation']]
        all_years_data.append(df_final_year)

    # ==========================================
    # FINAL STEP: AVERAGE EVERYTHING INTO ONE YEAR
    # ==========================================
    if not all_years_data:
        print("\n[!] Error: No data processed. Check file paths and names.")
        return

    print("\nAll years processed. Calculating the 8760-hour average...")
    master_df = pd.concat(all_years_data, ignore_index=True)

    # Extract time components to group by
    master_df['Month'] = master_df['Datetime'].dt.month
    master_df['Day'] = master_df['Datetime'].dt.day
    master_df['Hour'] = master_df['Datetime'].dt.hour

    # Drop Leap Day (Feb 29) to ensure a perfect 8760-hour typical year
    master_df = master_df[~((master_df['Month'] == 2) & (master_df['Day'] == 29))]

    # Group by the exact calendar hour and take the mathematical mean
    typical_year = master_df.groupby(['Month', 'Day', 'Hour']).mean().reset_index()

    # Recreate a clean Datetime index (we use 2023 arbitrarily as the base year since it isn't a leap year)
    typical_year['Datetime'] = pd.to_datetime(
        '2023-' + typical_year['Month'].astype(str) + '-' + typical_year['Day'].astype(str) + ' ' + typical_year['Hour'].astype(str) + ':00:00'
    )

    # Final cleanup
    typical_year = typical_year[['Datetime', 'Price_EUR_MWh', 'CO2_Intensity_g_kWh', 'Consommation']]
    typical_year = typical_year.iloc[:, [0, 2, 3, 4]]
    typical_year.columns = ['Datetime', 'Price_EUR_MWh', 'CO2_Intensity_g_kWh', 'Consommation']
    output_name = 'vessim_typical_energy_year_2016_2023.csv'
    typical_year.to_csv(output_name, index=False)
    
    print(f"\nSUCCESS! Created '{output_name}' with {len(typical_year)} exact hours.")
    print("Your Typical Energy Year is ready for the Vessim simulation.")

if __name__ == "__main__":
    main()